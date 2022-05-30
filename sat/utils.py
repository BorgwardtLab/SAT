# -*- coding: utf-8 -*-
from typing import Optional
import torch
from torch import Tensor
from torch_scatter import scatter, segment_csr, gather_csr
from torch_geometric.utils.num_nodes import maybe_num_nodes
from scipy.sparse import csr_matrix,lil_matrix
import numpy as np


def count_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])

def dense_to_sparse_tensor(matrix):
    rows, columns = torch.where(matrix > 0)
    values = torch.ones(rows.shape)
    indices = torch.from_numpy(np.vstack((rows,
                                          columns))).long()
    shape = torch.Size(matrix.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


def extract_node_feature(data, reduce='add'):
    if reduce in ['mean', 'max', 'add']:
        data.x = scatter(data.edge_attr,
                         data.edge_index[0],
                         dim=0,
                         dim_size=data.num_nodes,
                         reduce=reduce)
    else:
        raise Exception('Unknown Aggregation Type')
    return data


def pad_batch(x, ptr, return_mask=False):
    bsz = len(ptr) - 1
    # num_nodes = torch.diff(ptr)
    max_num_nodes = torch.diff(ptr).max().item()

    all_num_nodes = ptr[-1].item()
    cls_tokens = False
    x_size = len(x[0]) if isinstance(x, (list, tuple)) else len(x)
    if x_size > all_num_nodes:
        cls_tokens = True
        max_num_nodes += 1
    if isinstance(x, (list, tuple)):
        new_x = [xi.new_zeros(bsz, max_num_nodes, xi.shape[-1]) for xi in x]
        if return_mask:
            padding_mask = x[0].new_zeros(bsz, max_num_nodes).bool()
    else:
        new_x = x.new_zeros(bsz, max_num_nodes, x.shape[-1])
        if return_mask:
            padding_mask = x.new_zeros(bsz, max_num_nodes).bool()

    for i in range(bsz):
        num_node = ptr[i + 1] - ptr[i]
        if isinstance(x, (list, tuple)):
            for j in range(len(x)):
                new_x[j][i][:num_node] = x[j][ptr[i]:ptr[i + 1]]
                if cls_tokens:
                    new_x[j][i][-1] = x[j][all_num_nodes + i]
        else:
            new_x[i][:num_node] = x[ptr[i]:ptr[i + 1]]
            if cls_tokens:
                new_x[i][-1] = x[all_num_nodes + i]
        if return_mask:
            padding_mask[i][num_node:] = True
            if cls_tokens:
                padding_mask[i][-1] = False
    if return_mask:
        return new_x, padding_mask
    return new_x

def unpad_batch(x, ptr):
    bsz, n, d = x.shape
    max_num_nodes = torch.diff(ptr).max().item()
    num_nodes = ptr[-1].item()
    all_num_nodes = num_nodes
    cls_tokens = False
    if n > max_num_nodes:
        cls_tokens = True
        all_num_nodes += bsz
    new_x = x.new_zeros(all_num_nodes, d)
    for i in range(bsz):
        new_x[ptr[i]:ptr[i + 1]] = x[i][:ptr[i + 1] - ptr[i]]
        if cls_tokens:
            new_x[num_nodes + i] = x[i][-1]
    return new_x
