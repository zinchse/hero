from typing import Callable
import torch
from torch import nn, Tensor


VERTICES_DIM = 2
NEIGHBORHOOD_SIZE = 3


class InvalidShapeError(Exception):
    pass


def _check_shapes(in_channels: "int", vertices: "Tensor", edges: "Tensor") -> "None":
    if len(vertices.shape) != 3:
        raise InvalidShapeError(
            f"Expected # axis for vertices is 3 (batch x channel x node)\
                but now it is {len(vertices.shape)}"
        )
    if len(edges.shape) != 4:
        raise InvalidShapeError(
            f"Expected # axis for edges is 4 (batch x channel x node x neighbor_index),\
                but now it is {len(edges.shape)}"
        )

    vertices_channels = vertices.shape[1]
    edges_channels = edges.shape[1]
    edges_last_dim = edges.shape[-1]

    if vertices_channels != in_channels:
        raise InvalidShapeError(
            f"Expected # of channels in vertices is {in_channels},\
                but now it is {vertices_channels}"
        )
    if edges_channels != 1:
        raise InvalidShapeError(
            f"Expected # of channels in edges is 1,\
            but now it is {edges_channels}"
        )
    if edges_last_dim != NEIGHBORHOOD_SIZE:
        raise InvalidShapeError(
            f"Expected # of edges for each node is {NEIGHBORHOOD_SIZE},\
                but now it is {edges_last_dim}"
        )


class BinaryTreeActivation(nn.Module):
    def __init__(self, activation: "Callable"):
        super().__init__()
        self._activation = activation

    def forward(self, vertices: "Tensor", edges: "Tensor") -> "Tensor":
        _ = edges
        return self._activation(vertices)


class BinaryTreeAdaptivePooling(nn.Module):
    def __init__(self, adaptive_pool: "Callable"):
        super().__init__()
        self._adaptive_pool = adaptive_pool

    def forward(self, vertices: "Tensor", edges: "Tensor") -> "Tensor":
        _ = edges
        return self._adaptive_pool(vertices).squeeze(-1)


class BinaryTreeConv(nn.Module):
    def __init__(self, in_channels: "int", out_channels: "int"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1d = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            stride=NEIGHBORHOOD_SIZE,
            kernel_size=NEIGHBORHOOD_SIZE,
        )

    def forward(self, vertices: "Tensor", edges: "Tensor") -> "Tensor":
        _check_shapes(in_channels=self.in_channels, vertices=vertices, edges=edges)
        padded_vertices = self._add_padding_vertex(vertices=vertices)
        flattened_neighborhoods = self._flatten_neighborhoods(vertices=padded_vertices, edges=edges)
        convoluted_vertices = self.conv1d(flattened_neighborhoods)
        return convoluted_vertices

    def _flatten_neighborhoods(self, vertices: "Tensor", edges: "Tensor") -> "Tensor":
        all_channels_edges = edges.expand(-1, self.in_channels, -1, -1)
        neighborhood_backbone = vertices.unsqueeze(-1).expand(-1, -1, -1, NEIGHBORHOOD_SIZE)
        neighborhoods = torch.gather(neighborhood_backbone, VERTICES_DIM, all_channels_edges)
        flattened_neighborhoods = neighborhoods.flatten(2)
        return flattened_neighborhoods

    def _add_padding_vertex(self, vertices: "Tensor") -> "Tensor":
        batch_size = vertices.shape[0]
        padding_vertex = torch.zeros((batch_size, self.in_channels, 1)).to(vertices.device)
        padded_vertices = torch.cat((padding_vertex, vertices), dim=2)
        return padded_vertices


class BinaryTreeSequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.layers = nn.ModuleList(args)

    def forward(self, vertices: "Tensor", edges: "Tensor") -> "Tensor":
        for layer in self.layers:
            vertices = layer(vertices, edges)
        return vertices


class BinaryTreeLayerNorm(nn.Module):
    def __init__(self, in_channels: "int", eps: "float" = 1e-5, frozen: "bool" = False):
        super().__init__()
        self.in_channels = in_channels
        self.gamma = nn.Parameter(torch.ones(in_channels))
        self.beta = nn.Parameter(torch.zeros(in_channels))
        self.eps = eps
        if frozen:
            self.gamma.requires_grad = False
            self.beta.requires_grad = False
        
    def forward(self, vertices: "Tensor", edges: "Tensor") -> "Tensor":
        _check_shapes(in_channels=self.in_channels, vertices=vertices, edges=edges)
        mean = vertices.mean(dim=(1, 2), keepdim=True)
        std = vertices.std(dim=(1, 2), keepdim=True, unbiased=False)
        normalized_vertices = (vertices - mean) / (std + self.eps)
        gamma = self.gamma.view(1, -1, 1)
        beta = self.beta.view(1, -1, 1)
        return gamma * normalized_vertices + beta


class BinaryTreeInstanceNorm(nn.Module):
    def __init__(self, in_channels: "int", eps: "float" = 1e-5, frozen: "bool" = False):
        super().__init__()
        self.in_channels = in_channels
        self.gamma = nn.Parameter(torch.ones(in_channels))
        self.beta = nn.Parameter(torch.zeros(in_channels))
        self.eps = eps        
        if frozen:
            self.gamma.requires_grad = False
            self.beta.requires_grad = False

    def forward(self, vertices: "Tensor", edges: "Tensor") -> "Tensor":
        _check_shapes(in_channels=self.in_channels, vertices=vertices, edges=edges)
        mean = vertices.mean(dim=2, keepdim=True)
        std = vertices.std(dim=2, keepdim=True, unbiased=False)
        normalized_vertices = (vertices - mean) / (std + self.eps)
        gamma = self.gamma.view(1, -1, 1)
        beta = self.beta.view(1, -1, 1)
        return gamma * normalized_vertices + beta
