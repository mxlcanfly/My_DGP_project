import argparse
import json
import random
import os.path as osp

import torch
import torch.nn.functional as F

from utils import ensure_path, set_gpu, l2_loss
from models.gcn_dense_att import GCN_Dense_Att


def save_checkpoint(name):
    torch.save(gcn.state_dict(), osp.join(save_path, name + '.pth'))
    torch.save(pred_obj, osp.join(save_path, name + '.pred'))


def mask_l2_loss(a, b, mask):
    return l2_loss(a[mask], b[mask])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #创建对象
    parser.add_argument('--max-epoch', type=int, default=3000)#添加参数，这些调用指定 ArgumentParser 如何获取命令行字符串并将其转换为对象。这些信息在 parse_args() 调用时被存储和使用
    parser.add_argument('--trainval', default='10,0')#训练集和测试集占得比例
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    #weight-decay权值衰减，在损失函数中，他是放在正则项前面的系数，为了调节模型复杂度对损失函数的影响
    parser.add_argument('--save-epoch', type=int, default=300)
    parser.add_argument('--save-path', default='save/gcn-dense-att')

    parser.add_argument('--gpu', default='0')

    parser.add_argument('--no-pred', action='store_true')#action命令行遇到参数时默认store
    args = parser.parse_args()
    #解析添加的参数，把每个参数转换为适当的类型然后调用相应的操作
    set_gpu(args.gpu)

    save_path = args.save_path
    ensure_path(save_path)

    graph = json.load(open('materials/imagenet-dense-grouped-graph.json', 'r'))#json是用来存储简单的数据结构和对象的文件
    wnids = graph['wnids']
    n = len(wnids)

    edges_set = graph['edges_set']
    print('edges_set', [len(l) for l in edges_set])

    lim = 4#每个阶段k值的数量设置为4
    for i in range(lim + 1, len(edges_set)):
        edges_set[lim].extend(edges_set[i])
    edges_set = edges_set[:lim + 1]
    print('edges_set', [len(l) for l in edges_set])#这里将14层减少到了4层+上自环，将后面的层全部加到第4层。后面计算权重的时候将他们算作一层

    word_vectors = torch.tensor(graph['vectors']).cuda()
    word_vectors = F.normalize(word_vectors)#对输入数据进行标准化使得输入数据满足正态分布，使得网络更容易收敛
    #fc为全连接层
    fcfile = json.load(open('materials/fc-weights.json', 'r'))
    train_wnids = [x[0] for x in fcfile]
    fc_vectors = [x[1] for x in fcfile]
    assert train_wnids == wnids[:len(train_wnids)]
    fc_vectors = torch.tensor(fc_vectors).cuda()#cnn训练的权重
    fc_vectors = F.normalize(fc_vectors)#F.normalize将某一个维度除以那个维度对应的范数(默认是2范数)。

    hidden_layers = 'd2048,d'
    gcn = GCN_Dense_Att(n, edges_set,
                        word_vectors.shape[1], fc_vectors.shape[1], hidden_layers).cuda()#训练dgp来预测最后一层的cnn权值，最终输出维数对应于resnrt-50架构的最后一层
    #对几个邻接矩阵进行处理

    print('word vectors:', word_vectors.shape)
    print('fc vectors:', fc_vectors.shape)
    print('hidden layers:', hidden_layers)

    optimizer = torch.optim.Adam(gcn.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    v_train, v_val = map(float, args.trainval.split(','))
    n_trainval = len(fc_vectors)
    n_train = round(n_trainval * (v_train / (v_train + v_val)))#round四舍五入函数
    print('num train: {}, num val: {}'.format(n_train, n_trainval - n_train))
    tlist = list(range(len(fc_vectors)))
    random.shuffle(tlist)#将原列表打乱顺序,这里是将1000打乱顺序
    min_loss = 1e18

    trlog = {}
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['min_loss'] = 0

    for epoch in range(1, args.max_epoch + 1):
        gcn.train()
        output_vectors = gcn(word_vectors)
        # z=output_vectors[tlist[0]]
        # o=fc_vectors[tlist[0]]
        loss = mask_l2_loss(output_vectors, fc_vectors, tlist[:n_train])#对输出进行l2正则化
        #训练DGP来训练每个节点的权值，训练类的权重从一个预先训练好的resnet网络优化提取，训练模型，通过损失来预测所见类的分类器权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #上面的loss是为了进行反向传播进行参数的更新，而下面的train_loss是为了观察训练时损失的下降情况

        gcn.eval()
        output_vectors = gcn(word_vectors)#这里进行了测试是为了记录训练时候的每一步的损失么？？
        train_loss = mask_l2_loss(output_vectors, fc_vectors, tlist[:n_train]).item()
        if v_val > 0:
            val_loss = mask_l2_loss(output_vectors, fc_vectors, tlist[n_train:]).item()#这里是为了干什么的，与上面的train_loss不是一样的么
            loss = val_loss
        else:
            val_loss = 0
            loss = train_loss
        print('epoch {}, train_loss={:.4f}, val_loss={:.4f}'
              .format(epoch, train_loss, val_loss))

        trlog['train_loss'].append(train_loss)
        trlog['val_loss'].append(val_loss)
        trlog['min_loss'] = min_loss
        torch.save(trlog, osp.join(save_path, 'trlog'))

        if (epoch % args.save_epoch == 0):
            if args.no_pred:
                pred_obj = None
            else:
                pred_obj = {
                    'wnids': wnids,
                    'pred': output_vectors#个人理解：通过gcn传出来的一个分类器
                }

        if epoch % args.save_epoch == 0:
            save_checkpoint('epoch-{}'.format(epoch))

        pred_obj = None
    #一共运行3000次，也就是进行3000次参数优化，每次进行一次loss,
    #这里有一个疑问，就是我每次输出的output_vectors跟fc_vector进行损失计算都是取前1000个,那么后面的就不会参与更新
    #那我最后取的pred就必须也取前1000个参数才可以

