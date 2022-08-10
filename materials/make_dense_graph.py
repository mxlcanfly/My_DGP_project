import argparse
import json

#根据构图找到所有子孙与祖先节点之间的联系
parser = argparse.ArgumentParser()
parser.add_argument('--input', default='imagenet-induced-graph.json')
parser.add_argument('--output', default='imagenet-dense-graph.json')
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

new_edges = []

for u, wnid in enumerate(wnids):
    q = [u]
    l = 0
    d = {}
    d[u] = 0
    while l < len(q):
        x = q[l]
        l += 1
        for y in adjs[x]:#首先取元素u=0，这里x=0，而后我们遍历adjs中所有值为0的元素所对应的键值，从而找到0号节点所有的子孙
            if d.get(y) is None:
                d[y] = d[x] + 1
                q.append(y)
    for x, dis in d.items():
        new_edges.append((u, x))#构建新的边，边为祖先与子孙节点的关系

json.dump({'wnids': wnids, 'vectors': vectors, 'edges': new_edges},
          open(args.output, 'w'))
#最后的结果只是在一个列表中存储的，不利于对该向量就行操作，因此我们可以将其分组
