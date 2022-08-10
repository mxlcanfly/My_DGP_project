import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='imagenet-induced-graph.json')
parser.add_argument('--output', default='imagenet-dense-grouped-graph.json')
args = parser.parse_args()

js = json.load(open(args.input, 'r'))
wnids = js['wnids']
vectors = js['vectors']
edges = js['edges']

n = len(wnids)
adjs = {}
for i in range(n):
    adjs[i] = []
for u, v in edges:
    adjs[u].append(v)

new_edges = [[] for i in range(99)]
#将所有的边分成了14组，目的暂时不明确
#关于为什么最后分了14组，因为最大联通的边就是14条，故分成了14组
for u, wnid in enumerate(wnids):
    q = [u]
    l = 0
    d = {}
    d[u] = 0
    while l < len(q):
        x = q[l]
        l += 1
        for y in adjs[x]:
            if d.get(y) is None:
                d[y] = d[x] + 1
                q.append(y)
    for x, dis in d.items():
        new_edges[dis].append((u, x))

while new_edges[-1] == []:
    new_edges.pop()

json.dump({'wnids': wnids, 'vectors': vectors, 'edges_set': new_edges},
          open(args.output, 'w'))
#分成了14组，可以更好地对进行分类处理

#归一化是为了消除不同数据之间的量纲，方便数据比较和共同处理，神经网络中，归一化可以加快训练网络的收敛性
#标准化是为了方便数据的下一步处理，而进行的数据所方等变换
#正则化是利用先验知识，在处理过程中一如正则化因子，增加引导约束的作用，逻辑回归中使用正则化，可有效减低过拟合
#知识图谱的构建：同样的ImageNet数据集提供原始图像数据，WordNet提供常识知识规则，构成知识图谱。

