# Dense Graph Propagation

The code for the paper [Rethinking Knowledge Graph Propagation for Zero-Shot Learning](https://arxiv.org/abs/1805.11724).

### Citation
```
@inproceedings{kampffmeyer2019rethinking,
  title={Rethinking knowledge graph propagation for zero-shot learning},
  author={Kampffmeyer, Michael and Chen, Yinbo and Liang, Xiaodan and Wang, Hao and Zhang, Yujia and Xing, Eric P},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={11487--11496},
  year={2019}
}
```

## Requirements

* python 3
* pytorch 0.4.0
* nltk

## Instructions

### Materials Preparation

There is a folder `materials/`, which contains some meta data and programs already.

#### Glove Word Embedding
1. Download: http://nlp.stanford.edu/data/glove.6B.zip
2. Unzip it, find and put `glove.6B.300d.txt` to `materials/`.

#### Graphs
1. `cd materials/`
2. Run `python make_induced_graph.py`, get `imagenet-induced-graph.json`
3. Run `python make_dense_graph.py`, get `imagenet-dense-graph.json`
3. Run `python make_dense_grouped_graph.py`, get `imagenet-dense-grouped-graph.json`

#### Pretrained ResNet50
1. Download: https://download.pytorch.org/models/resnet50-19c8e357.pth
2. Rename and put it as `materials/resnet50-raw.pth`
3. `cd materials/`, run `python process_resnet.py`, get `fc-weights.json` and `resnet50-base.pth`

#### ImageNet and AwA2

Download ImageNet and AwA2, create the softlinks (command `ln -s`): `materials/datasets/imagenet` and `materials/datasets/awa2`, to the root directory of the dataset.

An ImageNet root directory should contain image folders, each folder with the wordnet id of the class.

An AwA2 root directory should contain the folder JPEGImages.

### Training

Make a directory `save/` for saving models.

In most programs, use `--gpu` to specify the devices to run the code (default: use gpu 0).

#### Train Graph Networks
* SGCN: Run `python train_gcn_basic.py`, get results in `save/gcn-basic`
* DGP: Run `python train_gcn_dense_att.py`, get results in `save/gcn-dense-att`

In the results folder:
* `*.pth` is the state dict of Graph Networks model
* `*.pred` is the prediction file, which can be loaded by `torch.load()`. It is a python dict, having two keys: `wnids` - the wordnet ids of the predicted classes, `pred` - the predicted fc weights

#### Finetune ResNet
Run `python train_resnet_fit.py` with the args:
* `--pred`: the `.pred` file for finetuning
* `--train-dir`: the directory contains 1K imagenet training classes, each class with a folder named by its wordnet id
* `--save-path`: the folder you want to save the result, e.g. `save/resnet-fit-xxx`

(In the paper's setting, --train-dir is the folder composed of 1K classes from fall2011.tar, with the missing class "teddy bear" from ILSVRC2012.)

### Testing

#### ImageNet
Run `python evaluate_imagenet.py` with the args:
* `--cnn`: path to resnet50 weights, e.g. `materials/resnet50-base.pth` or `save/resnet-fit-xxx/x.pth`
* `--pred`: the `.pred` file for testing
* `--test-set`: load test set in `materials/imagenet-testsets.json`, choices: `[2-hops, 3-hops, all]`
* (optional) `--keep-ratio` for the ratio of testing data, `--consider-trains` to include training classes' classifiers, `--test-train` for testing with train classes images only.

#### AwA2
Run `python evaluate_awa2.py` with the args:
* `--cnn`: path to resnet50 weights, e.g. `materials/resnet50-base.pth` or `save/resnet-fit-xxx/x.pth`
* `--pred`: the `.pred` file for testing
* (optional) `--consider-trains` to include training classes' classifiers

