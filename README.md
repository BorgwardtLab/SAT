# Structure-Aware Transformer

The repository implements the Structure-Aware Transformer (SAT) in Pytorch Geometric described in the following paper

>Dexiong Chen*, Leslie O'Bray*, and Karsten Borgwardt
[Structure-Aware Transformer for Graph Representation Learning][1]. ICML 2022.
<br/>*Equal contribution

**TL;DR**: A class of simple and flexible graph transformers built upon a new self-attention mechanism, which incorporates structural information into the original self-attention by extracting a subgraph representation rooted at each node before computing the attention. Our structure-aware framework can leverage any existing GNN to extract the subgraph representation and systematically improve the peroformance relative to the base GNN.

## Citation

TODO

## A short description about the SAT attention mechanism

TODO: Image + brief summary

## Installation

The dependencies are managed by [miniconda][2]

```
python=3.9
numpy
scipy
pytorch=1.9.1
pytorch-geometric=2.0.2
einops
ogb
```

Once you have activated the environment and installed all dependencies, run:

```bash
source s
```
TODO: use setup.py instead of source?

Datasets will be downloaded via Pytorch geometric and OGB package.

## Train SAT on graph and node prediction datasets

All our experimental scripts are in the folder `experiments`. So to start with, run `cd experiments`. The hyperparameters used below are selected as optimal

#### Graph regression on ZINC dataset

Train a k-subtree SAT with PNA:
```bash
python train_zinc.py --abs-pe rw --se gnn --gnn-type pna2 --dropout 0.3 --k-hop 3 --use-edge-attr
```

Train a k-subgraph SAT with PNA
```bash
python train_zinc.py --abs-pe rw --se khopgnn --gnn-type pna2 --dropout 0.2 --k-hop 3 --use-edge-attr
```

#### Node classification on PATTERN and CLUSTER datasets

Train a k-subtree SAT on PATTERN:
```bash
python train_SBMs.py --dataset PATTERN --weight-class --abs-pe rw --abs-pe-dim 7 --se gnn --gnn-type pna3 --dropout 0.2 --k-hop 3 --num-layers 6 --lr 0.0003
```

and on CLUSTER:
```bash
python train_SBMs.py --dataset CLUSTER --weight-class --abs-pe rw --abs-pe-dim 3 --se gnn --gnn-type pna2 --dropout 0.4 --k-hop 3 --num-layers 16 --dim-hidden 48 --lr 0.0005
```

#### Graph classification on OGB datasets

`--gnn-type` can be `gcn`, `gine` or `pna`, where `pna` obtains the best performance.

```bash
python train_ppa.py --gnn-type gcn --use-edge-attr
```

```bash
python train_code2.py --gnn-type gcn --use-edge-attr
```


[1]: https://arxiv.org/abs/2202.03036
[2]: https://conda.io/miniconda.html
