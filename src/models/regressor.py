from torch import nn, Tensor
from src.models.binary_tree_layers import BinaryTreeSequential


class BinaryTreeRegressor(nn.Module):
    def __init__(self, btcnn: "BinaryTreeSequential", fcnn: "nn.Sequential", name: "str" = "unknown"):
        super().__init__()
        self.btcnn: "BinaryTreeSequential" = btcnn
        self.fcnn: "nn.Sequential" = fcnn
        self.name: "str" = name

    def forward(self, vertices: "Tensor", edges: "Tensor") -> "Tensor":
        return self.fcnn(self.btcnn(vertices=vertices, edges=edges))
