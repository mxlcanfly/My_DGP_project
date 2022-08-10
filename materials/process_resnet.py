import json
import torch
import torch.nn.functional as F
#使用resnst50进行处理得到一组权值和偏置值，而后给到要做训练的每一个语义上
p = torch.load('resnet50-raw.pth')#已经在Imagenet上与训练了
w = p['fc.weight'].data
b = p['fc.bias'].data
#这里就已经取到了w，b，下边的pop的作用是什么
p.pop('fc.weight')#通过p.pop后Tensor变成了Parameter类型了
p.pop('fc.bias')
torch.save(p, 'resnet50-base.pth')


#Dgp处理后的特征维数为2048，最终输出维度对应于resnet50的最后一层权重数2049
v = torch.cat([w, b.unsqueeze(1)], dim=1).tolist()#cat将两个张量拼接在一起，unsqueeze（1）对一维度扩展一维,从（1000，）变成（1000，1）
wnids = json.load(open('imagenet-split.json', 'r'))['train']
wnids = sorted(wnids)#对wnids进行排序
obj = []
for i in range(len(wnids)):
    obj.append((wnids[i], v[i]))
json.dump(obj, open('fc-weights.json', 'w'))#这里就是将权重和偏置赋给每一个语义向量

#后面可以学习这里的方式对权重和偏置进行拼接，结合make_resnet_base中的代码，这里其实只将权重和偏置进行了保存，而未对
#其他位置的参数进行保存，而后将wnids和每个v进行配对，由于使用的是ImageNet进行的构图，所以只需将wnids排序即可