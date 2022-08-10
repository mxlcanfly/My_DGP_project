import argparse
import json
import nltk
nltk.download('omw-1.4')

from nltk.corpus import wordnet as wn
import torch

from glove import GloVe

#找到所有的祖先节点对其进行构图，
def getnode(x):
    return wn.synset_from_pos_and_offset('n', int(x[1:]))


def getwnid(u):
    s = str(u.offset())
    return 'n' + (8 - len(s)) * '0' + s


def getedges(s):
    dic = {x: i for i, x in enumerate(s)}
    edges = []
    for i, u in enumerate(s):
        for v in u.hypernyms():
            j = dic.get(v)
            if j is not None:
                edges.append((i, j))
    return edges


def induce_parents(s, stop_set):
    q = s
    vis = set(s)
    l = 0
    while l < len(q):
        u = q[l]
        l += 1
        if u in stop_set:       #？
            continue            #满足条件不执行
        for p in u.hypernyms(): #？
            if p not in vis:
                vis.add(p)      #add往集合中添加元素
                q.append(p)     #往列表中添加元素


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='imagenet-split.json')
    parser.add_argument('--output', default='imagenet-induced-graph.json')
    args = parser.parse_args()

    print('making graph ...')#构造带有祖先借点的图

    xml_wnids = json.load(open('imagenet-xml-wnids.json', 'r'))
    xml_nodes = list(map(getnode, xml_wnids))
    xml_set = set(xml_nodes)

    js = json.load(open(args.input, 'r'))
    train_wnids = js['train']
    test_wnids = js['test']

    key_wnids = train_wnids + test_wnids

    s = list(map(getnode, key_wnids))
    induce_parents(s, xml_set)

    s_set = set(s)
    for u in xml_nodes:
        if u not in s_set:
            s.append(u)

    wnids = list(map(getwnid, s))
    edges = getedges(s)

    print('making glove embedding ...')#将语义信息嵌入到图中

    glove = GloVe('glove.6B.300d.txt')#作为图中的类别特征表示
    vectors = []
    for wnid in wnids:
        vectors.append(glove[getnode(wnid).lemma_names()])
    vectors = torch.stack(vectors)

    print('dumping ...')

    obj = {}
    obj['wnids'] = wnids
    obj['vectors'] = vectors.tolist()#个人理解，将一个类别所包含的语义信息拼接在一起，并变成列表格式
    obj['edges'] = edges
    json.dump(obj, open(args.output, 'w'))

