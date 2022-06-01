# Structure-Aware Transformer

The repository implements the Structure-Aware Transformer (SAT) in Pytorch Geometric described in the following paper

>Dexiong Chen*, Leslie O'Bray*, and Karsten Borgwardt
[Structure-Aware Transformer for Graph Representation Learning][1]. ICML 2022.
<br/>*Equal contribution

**TL;DR**: A class of simple and flexible graph transformers built upon a new self-attention mechanism, which incorporates structural information into the original self-attention by extracting a subgraph representation rooted at each node before computing the attention. Our structure-aware framework can leverage any existing GNN to extract the subgraph representation and systematically improve the peroformance relative to the base GNN.

## Citation


```bibtex
@InProceedings:{Chen22a,
	author = {Dexiong Chen and Leslie O'Bray and Karsten Borgwardt},
	title = {Structure-Aware Transformer for Graph Representation Learning},
	year = {2022},
	booktitle = {Proceedings of the 39th International Conference on Machine Learning~(ICML)},
	series = {Proceedings of Machine Learning Research}
}
```


TODO

## A short description about the SAT attention mechanism

TODO: Image + brief summary

#### A quick-start example

Below you can find a quick-start example on the ZINC dataset, see `./experiments/train_zinc.py` for more details.

<details><summary>click to see the example:</summary>

```python
import torch
from torch_geometric import datasets
from torch_geometric.loader import DataLoader
from sat.data import GraphDataset
from sat import GraphTransformer

# Load the ZINC dataset using our wrapper GraphDataset,
# which automatically creates the fully connected graph.
# For datasets with large graph, we recommend setting return_complete_index=False
# leading to faster computation
dset = datasets.ZINC('./datasets/ZINC', subset=True, split='train')
dset = GraphDataset(dset)

# Create a PyG data loader
train_loader = DataLoader(dset, batch_size=16, shuffle=True)

# Create a SAT model
dim_hidden = 16
gnn_type = 'gcn' # use GCN as the structure extractor
k_hop = 2 # use a 2-layer GCN

model = GraphTransformer(
    in_size=28, # number of node labels for ZINC
    num_class=1, # regression task
    d_model=dim_hidden,
    dim_feedforward=2 * dim_hidden,
    num_layers=2,
    batch_norm=True,
    gnn_type='gcn', # use GCN as the structure extractor
    use_edge_attr=True,
    num_edge_features=4, # number of edge labels
    edge_dim=dim_hidden,
    k_hop=k_hop,
    se='gnn', # we use the k-subtree structure extractor
    global_pool='add'
)

for data in train_loader:
    output = model(data) # batch_size x 1
    break
```
</details>

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

All our experimental scripts are in the folder `experiments`. So to start with, after having run `source s`, run `cd experiments`. The hyperparameters used below are selected as optimal

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
# Train SAT on OGBG-PPA
python train_ppa.py --gnn-type gcn --use-edge-attr
```

```bash
# Train SAT on OGBG-CODE2
python train_code2.py --gnn-type gcn --use-edge-attr
```

TODO: include pretrained models for CODE2


[1]: https://arxiv.org/abs/2202.03036
[2]: https://conda.io/miniconda.html
