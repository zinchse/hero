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


def test_example_convolution(convolved_vertices: "Tensor"):
    expected_ans = torch.tensor([[100 + 0.0], [100 + 6.0], [100 + 0.0], [100 + 0.0]])
    assert torch.allclose(convolved_vertices, expected_ans)


def test_example_activation(activated_vertices: "Tensor"):
    expected_ans = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    assert torch.allclose(activated_vertices, expected_ans)


def test_example_pooling(pooled_vertices: "Tensor"):
    expected_ans = torch.tensor([[1.0], [1.0]])
    assert torch.allclose(pooled_vertices, expected_ans)


def test_example_layer_normalization(vertices: "Tensor", layer_normalized_vertices: "Tensor"):
    expected_ans = (vertices - torch.tensor([1 / 4])) / torch.tensor([0.9682458365518543])
    assert torch.allclose(layer_normalized_vertices, expected_ans)


def test_example_instance_normalization(vertices: "Tensor", instance_normalized_vertices: "Tensor"):
    expected_ans = (vertices - torch.tensor([1 / 2, 0.0])) / torch.tensor([0.8660254037844386, 1.0])
    assert torch.allclose(instance_normalized_vertices, expected_ans)


def test_example_full_block(vertices: "Tensor", edges: "Tensor", full_block_processed_vertices: "Tensor"):
    expected_ans = torch.tensor([[4.5 / 2.598076211353316]])
    assert torch.allclose(full_block_processed_vertices, expected_ans)
