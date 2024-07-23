import pytest
from src.models.binary_tree_layers import (
    BinaryTreeActivation,
    BinaryTreeAdaptivePooling,
    BinaryTreeSequential,
    BinaryTreeConv,
    BinaryTreeLayerNorm,
    BinaryTreeInstanceNorm,
    InvalidShapeError,
    NEIGHBORHOOD_SIZE,
    _check_shapes,
)
from src.models.regressor import BinaryTreeRegressor
import torch
from torch import Tensor, nn


@pytest.fixture
def batch_vertices(vertices: "Tensor") -> "Tensor":
    return torch.stack([vertices]).transpose(1, 2)


@pytest.fixture
def batch_edges(edges: "Tensor") -> "Tensor":
    return torch.stack([edges]).unsqueeze(1)


def test_format():
    b, d, n = 1, 2, 10
    with pytest.raises(InvalidShapeError):
        _check_shapes(in_channels=d, vertices=torch.randn(b, n), edges=torch.randn(b, 1, n, NEIGHBORHOOD_SIZE))
    with pytest.raises(InvalidShapeError):
        _check_shapes(in_channels=d, vertices=torch.randn(b, d, n), edges=torch.randn(b, n, NEIGHBORHOOD_SIZE))
    with pytest.raises(InvalidShapeError):
        _check_shapes(in_channels=d, vertices=torch.randn(b, d + 1, n), edges=torch.randn(b, 1, n, NEIGHBORHOOD_SIZE))
    with pytest.raises(InvalidShapeError):
        _check_shapes(in_channels=d, vertices=torch.randn(b, d, n), edges=torch.randn(b, 1 + 1, n, NEIGHBORHOOD_SIZE))
    with pytest.raises(InvalidShapeError):
        _check_shapes(in_channels=d, vertices=torch.randn(b, d, n), edges=torch.randn(b, 1, n, NEIGHBORHOOD_SIZE + 1))


def test_convolution_implementation(
    batch_vertices: "Tensor", batch_edges: "Tensor", convolved_vertices: "Tensor", conv1d: "nn.Conv1d"
):
    binary_tree_convolution = BinaryTreeConv(in_channels=2, out_channels=1)
    binary_tree_convolution.conv1d = conv1d
    with torch.no_grad():
        res = binary_tree_convolution(batch_vertices, batch_edges).transpose(1, 2).squeeze(0)
        assert torch.allclose(convolved_vertices, res)


def test_activation_implementation(batch_vertices: "Tensor", batch_edges: "Tensor", activated_vertices: "Tensor"):
    binary_tree_activation = BinaryTreeActivation(activation=nn.functional.relu)
    with torch.no_grad():
        res = binary_tree_activation(batch_vertices, batch_edges).transpose(1, 2).squeeze(0)
        assert torch.allclose(activated_vertices, res)


def test_pooling_implementation(batch_vertices: "Tensor", batch_edges: "Tensor", pooled_vertices: "Tensor"):
    binary_tree_adaptive_pooling = BinaryTreeAdaptivePooling(adaptive_pool=nn.AdaptiveMaxPool1d(1))
    with torch.no_grad():
        res = binary_tree_adaptive_pooling(batch_vertices, batch_edges).squeeze(0)
        assert torch.allclose(pooled_vertices, res)


def test_layer_norm_implementation(
    batch_vertices: "Tensor", batch_edges: "Tensor", layer_normalized_vertices: "Tensor"
):
    binary_tree_layer_norm = BinaryTreeLayerNorm(in_channels=2, eps=0)
    with torch.no_grad():
        res = binary_tree_layer_norm(batch_vertices, batch_edges).transpose(1, 2).squeeze(0)
        assert torch.allclose(layer_normalized_vertices, res)


def test_instance_norm_implementation(
    batch_vertices: "Tensor", batch_edges: "Tensor", instance_normalized_vertices: "Tensor"
):
    binary_tree_instance_norm = BinaryTreeInstanceNorm(in_channels=2, eps=0)
    with torch.no_grad():
        res = binary_tree_instance_norm(batch_vertices, batch_edges).transpose(1, 2).squeeze(0)
        assert torch.allclose(instance_normalized_vertices, res)


def test_full_block_implementation(
    batch_vertices: "Tensor", batch_edges: "Tensor", full_block_processed_vertices: "Tensor", conv1d: "nn.Conv1d"
):
    block = BinaryTreeSequential(
        BinaryTreeConv(in_channels=2, out_channels=1),
        BinaryTreeLayerNorm(in_channels=1, eps=0),
        BinaryTreeActivation(activation=nn.functional.relu),
        BinaryTreeAdaptivePooling(adaptive_pool=nn.AdaptiveMaxPool1d(1)),
    )
    block.layers[0].conv1d = conv1d
    with torch.no_grad():
        res = block(vertices=batch_vertices, edges=batch_edges).squeeze(0)
        assert torch.allclose(full_block_processed_vertices, res)


def test_full_model(batch_vertices: "Tensor", batch_edges: "Tensor"):
    in_channels = 2
    small_btcnn = BinaryTreeSequential(
        BinaryTreeConv(in_channels, 64),
        BinaryTreeAdaptivePooling(nn.AdaptiveMaxPool1d(1)),
    )
    small_fcnn = nn.Sequential(
        nn.Linear(64, 32),
        nn.LeakyReLU(),
        nn.Linear(32, 1),
        nn.Softplus(),
    )
    model = BinaryTreeRegressor(small_btcnn, small_fcnn, "SmallTestArchitecture")
    assert model(batch_vertices, batch_edges).shape == (1, 1)
