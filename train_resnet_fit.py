import argparse
import json
import os
import os.path as osp
import time

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL.ImageFile import ImageFile
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DataParallel
from utils import set_gpu, ensure_path
from models.resnet import ResNet
from datasets.image_folder import ImageFolder

ImageFile.LOAD_TRUNCATED_IMAGES = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred',default='save/gcn-dense-att/epoch-3000.pred')
    parser.add_argument('--train-dir',default='../data_mxl/ImageNet-1k')#materials/datasets/awa2/AwA2/Animals_with_Attributes2/JPEGImages
    parser.add_argument('--save-path', default='save/resnet-fit')
    # parser.add_argument('--gpu', default='0')
    parser.add_argument('--device_ids',type=str,default='0')
    parser.add_argument('--local_rank',type=int,default=-1)
    args = parser.parse_args()

    print(args.local_rank)
    device_ids=list(map(int,args.device_ids.split(',')))
    dist.init_process_group(backend='nccl')
    device=torch.device('cuda:{}'.format(device_ids[args.local_rank]))
    # set_gpu(args.gpu)
    # save_path = args.save_path
    # ensure_path(save_path)

    pred = torch.load(args.pred)

    # path = '../data_mxl/ImageNet-1k'
    # file_name = os.listdir(path)
    train_winds = os.listdir(args.train_dir)#排序之前是乱序的
    train_wnids = sorted(os.listdir(args.train_dir))#os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表


    train_dir = args.train_dir
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(train_dir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),#随机对图片进行水平翻转
        transforms.ToTensor(),
        normalize]))
    #数据增强策略：对图片进行随机缩放裁剪或随机水平翻转，可进行数据增强

    train_sampler=DistributedSampler(train_dataset)

    loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256,
        num_workers=16, pin_memory=True, sampler=train_sampler)

    assert pred['wnids'][:1000] == train_wnids

    model = ResNet('resnet50',1000)#输入是2048，输出是50out_features,50传到resnet中是num_class,应该就是输出多少类
    #这个里面的参数是为了控制最后的分类数量
    sd = model.resnet_base.state_dict()
    sd.update(torch.load('materials/resnet50-base.pth'))
    model.resnet_base.load_state_dict(sd)

    #由于我们只需要将fc的最后一层的参数进行替换，所以我们不需要改变网络其他的参数，而之前我们将fc的最后一层的参数
    #进行了保存（保存在了resnet50-base中，所以这里使用update只对相应层进行了更改）
    #在之后的嵌入中我们考虑应该要在resnet50的bn层进行嵌入，首先考虑要将dgp的高层嵌入，这里就涉及到改变输出的矩阵的大小
    #的问题，因为bn层每一层的输入通道数都不同
    #由于考虑到ImageNet上进行训练时间过长，我们可以写一个awa2的训练模型进行尝试操作
    #如果在awa2上有效果的话，在考虑传输方式的改变，可以在GCN或者resnet50后面加入transformer进行尝试
    #随后我们可以尝试加入方法，如t2t-vit
    #存在问题：我们自己写的resnet50对ImageNet的训练可能和作者写的有所差异，在效果上可能有所差距，在训练的时候也可以加入注意力机制进行测试

    fcw = pred['pred'][:1000].cpu()#这里取得是前1000个分类器（预测的权重数据）,总共fcw是[1000,2049]
    model.fc.weight = nn.Parameter(fcw[:, :-1])#将fcw的前50行2048列取出来付给fc.weight
    model.fc.bias = nn.Parameter(fcw[:, -1])#将fcw的最后一列赋给fc.bias
    # a=model.fc.weight
    # b=model.fc.bias

    torch.cuda.set_device(device)
    model = model.to(device)
    model=DistributedDataParallel(model,device_ids=[device_ids[args.local_rank]],output_device=device_ids[args.local_rank])
    model=model.module
    model.train()

    optimizer = torch.optim.SGD(model.resnet_base.parameters(), lr=0.0001, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss().to(device)

    keep_ratio = 0.9975
    trlog = {}
    trlog['loss'] = []
    trlog['acc'] = []

    for epoch in range(1, 9999):
        train_sampler.set_epoch(epoch)

        ave_loss = None
        ave_acc = None
        for i, (data, label) in enumerate(loader, 1):
            data = data.to(device)
            label = label.to(device)

            logits = model(data)
            loss = loss_fn(logits, label)

            _, pred = torch.max(logits, dim=1)
            acc = torch.eq(pred, label).type(torch.FloatTensor).mean().item()#item（）取出单元素张量的元素值并返回该值，保持原元素类型不变
            #torch.eq用来比对相应位置上的两个值是否系相等，相等为1否则为0。type转换数据类型,mean求均值

            if i == 1:
                ave_loss = loss.item()
                ave_acc = acc
            else:
                ave_loss = ave_loss * keep_ratio + loss.item() * (1 - keep_ratio)
                ave_acc = ave_acc * keep_ratio + acc * (1 - keep_ratio)

            if args.local_rank == 0:

                print('Train Time:{} epoch {}, {}/{}, loss={:.4f} ({:.4f}), acc={:.4f} ({:.4f})'
                      .format(time.strftime("%Y-%m-%d - %H:%M:%S"),epoch, i, len(loader), loss.item(), ave_loss, acc, ave_acc))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        trlog['loss'].append(ave_loss)
        trlog['acc'].append(ave_acc)

        #
        torch.save(trlog, osp.join('save/resnet-fit', 'trlog'))

        if args.local_rank == 0:
            torch.save(model.resnet_base.state_dict(),
                   osp.join('save/resnet-fit', 'epoch-{}.pth'.format(epoch)))