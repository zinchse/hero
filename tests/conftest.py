import sys
import os
sys.path.insert(0, os.getcwd())

from typing import List, Tuple, cast
import pytest
from src.models.binary_tree_layers import (
    BinaryTreeActivation,
    BinaryTreeAdaptivePooling,
    BinaryTreeSequential,
    BinaryTreeConv,
    BinaryTreeLayerNorm,
    BinaryTreeInstanceNorm,
)
import torch
from torch import nn, Tensor


@pytest.fixture
def bias() -> "Tensor":
    return torch.tensor([100.0])


@pytest.fixture
def first_channel_weights() -> "Tensor":
    return torch.tensor([1.0, -1.0, 1.0])


@pytest.fixture
def second_channel_weights() -> "Tensor":
    return torch.tensor([-1.0, -1.0, 1.0])


@pytest.fixture
def conv1d(bias: "Tensor", first_channel_weights: "Tensor", second_channel_weights: "Tensor") -> "torch.nn.Conv1d":
    conv_layer = nn.Conv1d(in_channels=2, out_channels=1, stride=3, kernel_size=3)
    conv_layer.weight.data = torch.stack([first_channel_weights, second_channel_weights], dim=0).unsqueeze(0)
    conv_layer.bias = cast(torch.nn.Parameter, conv_layer.bias)
    conv_layer.bias.data = bias
    return conv_layer


@pytest.fixture
def vertices() -> "Tensor":
    root_node, l_child_node, ll_child_node, rl_child_node = (
        [1.0, 1.0],
        [1.0, -1.0],
        [-1.0, -1.0],
        [1.0, 1.0],
    )
    return torch.tensor([root_node, l_child_node, ll_child_node, rl_child_node])


@pytest.fixture
def edges() -> "Tensor":
    padding_idx, root, l_child, ll_child, rl_child = (0, 1, 2, 3, 4)
    return torch.tensor(
        [
            [root, l_child, padding_idx],
            [l_child, ll_child, rl_child],
            [ll_child, padding_idx, padding_idx],
            [rl_child, padding_idx, padding_idx],
        ],
        dtype=torch.long,
    )


@pytest.fixture
def convolved_vertices(
    vertices: "Tensor",
    edges: "Tensor",
    first_channel_weights: "Tensor",
    second_channel_weights: "Tensor",
    bias: "Tensor",
) -> "Tensor":
    res = []
    padded_vertices = torch.cat([torch.tensor([[0.0, 0.0]]), vertices])
    for edge in edges:
        ngbs_1 = padded_vertices[edge, 0]
        ngbs_2 = padded_vertices[edge, 1]
        res.append([torch.dot(ngbs_1, first_channel_weights) + torch.dot(ngbs_2, second_channel_weights) + bias])
    return torch.tensor(res, dtype=torch.float32)


@pytest.fixture
def activated_vertices(vertices: "Tensor") -> "Tensor":
    return torch.tensor([[max(0, v[c]) for c in range(2)] for v in vertices])


@pytest.fixture
def pooled_vertices(vertices: "Tensor") -> "Tensor":
    return torch.tensor([max([v[c] for v in vertices]) for c in range(2)])


@pytest.fixture
def layer_normalized_vertices(vertices: "Tensor") -> "Tensor":
    return (vertices - vertices.mean()) / vertices.std(unbiased=False)


@pytest.fixture
def instance_normalized_vertices(vertices: "Tensor") -> "Tensor":
    return (vertices - vertices.mean(dim=0, keepdim=True)) / vertices.std(dim=0, keepdim=True, unbiased=False)


@pytest.fixture
def full_block_processed_vertices(convolved_vertices: "Tensor") -> "Tensor":
    normalized_vertices = (convolved_vertices - convolved_vertices.mean()) / convolved_vertices.std(unbiased=False)
    activated_vertices = torch.tensor([[max(0, v[c]) for c in range(1)] for v in normalized_vertices])
    return torch.tensor([max([v[c] for v in activated_vertices]) for c in range(1)])
