"""
There's a minimal version of a dataloader that takes into account the structure of the tree
 representation and the fact that they are very often repeated (as a result, a 
 weighted dataset is considered) 
"""

from typing import List, Tuple, Dict
import torch
from torch import Tensor
from torch.utils.data import Dataset


def paddify_sequences(sequences: "List[Tensor]", target_length: "int") -> "List[Tensor]":
    """
    Pads sequences to make them of equal length.
    """
    padded_sequences = []
    n_channels = sequences[0].shape[1]
    for seq in sequences:
        padding_tokens = torch.zeros((target_length - len(seq), n_channels), dtype=seq.dtype, device=seq.device)
        padded_seq = torch.cat((seq, padding_tokens), dim=0)
        padded_sequences.append(padded_seq)
    return padded_sequences


class WeightedBinaryTreeDataset(Dataset):
    def __init__(
        self,
        list_vertices: "List[Tensor]",
        list_edges: "List[Tensor]",
        list_time: "List[Tensor]",
        device: "torch.device",
    ):
        """
        An iterator over <tensor of vectorized tree nodes, tree structure, frequency execution time>
        with the ability to move data to the specified device.
        """
        self.data_dict: "Dict[Tuple, Dict]" = {}

        for vertices, edges, time in zip(list_vertices, list_edges, list_time):
            key = str(vertices.flatten().tolist()), str(edges.flatten().tolist())
            if key in self.data_dict:
                self.data_dict[key]["freq"] += 1
                self.data_dict[key]["time"].append(time)
            else:
                self.data_dict[key] = {"vertices": vertices, "edges": edges, "time": [time], "freq": 1}

        self.list_vertices = [v["vertices"] for v in self.data_dict.values()]
        self.list_edges = [v["edges"] for v in self.data_dict.values()]
        self.list_time = [torch.stack(v["time"]).mean() for v in self.data_dict.values()]
        self.list_frequencies = [torch.tensor(v["freq"]) for v in self.data_dict.values()]
        self.size = len(self.data_dict)
        self.device = device
        self.move_to_device()

    def move_to_device(self) -> "None":
        for idx in range(self.size):
            self.list_vertices[idx] = self.list_vertices[idx].to(device=self.device)
            self.list_edges[idx] = self.list_edges[idx].to(device=self.device)
            self.list_frequencies[idx] = self.list_frequencies[idx].to(device=self.device)
            self.list_time[idx] = self.list_time[idx].to(device=self.device)

    def __len__(self) -> "int":
        return self.size

    def __getitem__(self, idx) -> "Tuple[Tensor, Tensor, Tensor, Tensor]":
        return self.list_vertices[idx], self.list_edges[idx], self.list_frequencies[idx], self.list_time[idx]


def weighted_binary_tree_collate(
    batch: "List[Tuple[Tensor, Tensor, Tensor, Tensor]]", target_length: "int"
) -> "Tuple[Tuple[Tensor, Tensor, Tensor], Tensor]":
    """
    Adds padding to equalize lengths, changes the number of axes and
    their order to make neural network inference more suitable.
    """
    list_vertices, list_edges, list_freq, list_time = [], [], [], []
    for vertices, edges, freq, time in batch:
        list_vertices.append(vertices)
        list_edges.append(edges)
        list_freq.append(freq)
        list_time.append(time)

    batch_vertices = torch.stack(paddify_sequences(list_vertices, target_length)).transpose(1, 2)
    batch_edges = torch.stack(paddify_sequences(list_edges, target_length)).unsqueeze(1)
    batch_freq = torch.stack(list_freq)
    return (batch_vertices, batch_edges, batch_freq), torch.stack(list_time)
