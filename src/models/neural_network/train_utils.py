""" 
Here's a minimal set for training, saving and loading of neural network architecture over binary trees.

By default the most successful model is used - a big convolutional network with instance normalisation.
"""

import os
from typing import Optional
from tqdm import tqdm
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from src.datasets.binary_tree_dataset import WeightedBinaryTreeDataset
from src.models.neural_network.regressor import BinaryTreeRegressor, get_big_fcnn, get_big_btcnn_and_instance_norm

DEFAULT_LR = 3e-4
DEFAULT_BATCH_SIZE = 256


def load_model(device: "torch.device", path: "str", model: "Optional[torch.nn.Module]" = None) -> "BinaryTreeRegressor":
    if model is None:
        model = BinaryTreeRegressor(get_big_btcnn_and_instance_norm(), get_big_fcnn(), device=device)
    ckpt_path = path
    ckpt_state = torch.load(ckpt_path, map_location=device)
    model = BinaryTreeRegressor(get_big_btcnn_and_instance_norm(), get_big_fcnn())
    model.load_state_dict(ckpt_state["model_state_dict"])
    model = model.to(device)
    model.device = device
    return model


def save_ckpt(
    model: "BinaryTreeRegressor", optimizer: "Optimizer", scheduler: "ReduceLROnPlateau", epoch: "int", path: "str"
) -> "None":
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def set_seed(seed: "int") -> "None":
    torch.manual_seed(seed)


def calculate_loss(
    model: "BinaryTreeRegressor",
    optimizer: "Optimizer",
    criterion: "nn.Module",
    dataloader: "DataLoader[WeightedBinaryTreeDataset]",
    train_mode: "bool" = True,
) -> "float":
    _ = model.train() if train_mode else model.eval()
    running_loss, total_samples = 0.0, 0
    for (vertices, edges, freq), time in dataloader:
        if train_mode:
            optimizer.zero_grad()

        outputs = model(vertices, edges)
        weighted_loss = (freq.float().squeeze(-1) * criterion(outputs.squeeze(-1), time)).mean()

        if train_mode:
            weighted_loss.backward()
            optimizer.step()

        running_loss += weighted_loss.item() * vertices.size(0)
        total_samples += freq.sum()
    return running_loss / total_samples


def weighted_train_loop(
    model: "BinaryTreeRegressor",
    optimizer: "Optimizer",
    criterion: "nn.Module",
    scheduler: "ReduceLROnPlateau",
    train_dataloader: "DataLoader[WeightedBinaryTreeDataset]",
    num_epochs: "int",
    start_epoch: "int" = 0,
    ckpt_period: "int" = 10,
    path_to_save: "Optional[str]" = None,
) -> "None":
    tqdm_desc = "Initialization"
    progress_bar = tqdm(range(start_epoch + 1, start_epoch + num_epochs + 1), desc=tqdm_desc, leave=True, position=0)
    for epoch in progress_bar:
        train_loss = calculate_loss(model, optimizer, criterion, train_dataloader)
        scheduler.step(train_loss)
        progress_bar.set_description(f"[{epoch}/{start_epoch + num_epochs}] MSE: {train_loss:.4f}")
        if path_to_save and not epoch % ckpt_period:
            save_ckpt(model, optimizer, scheduler, epoch, path_to_save)
