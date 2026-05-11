import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from torch.optim import Adam
from torch_geometric.nn import GATConv, GCNConv

from utils import get_disk_mask


torch.use_deterministic_algorithms(True)


def build_grid_edge_index(height: int, width: int, k: int = 4, self_loop: bool = False, device=None):
    coordinates = np.array([[i, j] for i in range(height) for j in range(width)])
    distances = cdist(coordinates, coordinates)

    edges = []
    for src in range(height * width):
        nearest = np.argsort(distances[src])[1 : k + 1]
        for dst in nearest:
            edges.append((src, dst))
            edges.append((dst, src))

    if self_loop:
        for src in range(height * width):
            edges.append((src, src))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    if device is not None:
        edge_index = edge_index.to(device)
    return edge_index


def batch_edge_index(edge_index, batch_size, num_nodes):
    edge_list = [edge_index + batch_idx * num_nodes for batch_idx in range(batch_size)]
    return torch.cat(edge_list, dim=1)


class Linear(nn.Module):
    def __init__(self, num_hidden, num_genes, alpha=0.01, beta=0.01, bias=True):
        super().__init__()
        self.linear = nn.Linear(num_hidden, num_genes, bias=bias)
        self.act = nn.ELU(alpha)
        self.beta = beta

    def forward(self, x):
        return self.act(self.linear(x)) + self.beta


class GCNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False, residual=True):
        super().__init__()
        self.conv = GCNConv(in_dim, out_dim, bias=bias, add_self_loops=False)
        self.residual = residual and (in_dim == out_dim)
        if not self.residual and residual:
            self.res_proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, edge_index):
        residual = x
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.1)
        if self.residual:
            x = x + residual
        elif hasattr(self, "res_proj"):
            x = x + self.res_proj(residual)
        return x


class GATBlock(nn.Module):
    def __init__(self, in_dim, out_dim, heads=1, bias=False, residual=True):
        super().__init__()
        self.conv = GATConv(in_dim, out_dim, heads=heads, concat=False, bias=bias)
        self.residual = residual and (in_dim == out_dim)
        if not self.residual and residual:
            self.res_proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, edge_index):
        residual = x
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.1)
        if self.residual:
            x = x + residual
        elif hasattr(self, "res_proj"):
            x = x + self.res_proj(residual)
        return x


class scstGCN(pl.LightningModule):
    def __init__(self, lr, num_features, num_genes, ori_radius, bias=False, graph_model="gcn", gat_heads=1):
        super().__init__()

        self.lr = lr
        self.ori_radius = ori_radius
        self.graph_model = graph_model.lower()

        block_cls = {"gcn": GCNBlock, "gat": GATBlock}[self.graph_model]
        block_kwargs = {"bias": bias}
        if self.graph_model == "gat":
            block_kwargs["heads"] = gat_heads

        self.conv1 = block_cls(num_features, 512, **block_kwargs)
        self.conv2 = block_cls(512, 512, **block_kwargs)
        self.output = nn.ModuleList(
            [Linear(512, num, alpha=0.01, beta=0.01, bias=bias) for num in num_genes]
        )

        self.save_hyperparameters()

    def forward(self, x):
        x = x.float()
        batch_size, num_nodes, channels = x.shape
        side = int(num_nodes**0.5)

        edge_index_single = build_grid_edge_index(side, side)
        edge_index = batch_edge_index(edge_index_single, batch_size, num_nodes)

        x = x.reshape(batch_size * num_nodes, channels)
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)

        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = F.dropout(x, 0.5, training=self.training)

        outputs = [head(x) for head in self.output]
        out = torch.cat(outputs, dim=-1)
        return out.view(batch_size, num_nodes, -1)

    def shared_step(self, batch, batch_idx):
        x, y_mean = batch
        mask = get_disk_mask(self.ori_radius / 16)
        mask = torch.BoolTensor(mask).to(self.device)

        y_pred = self.forward(x)
        y_pred = y_pred.reshape(y_pred.shape[0], mask.shape[0], mask.shape[1], y_pred.shape[2])
        y_pred = torch.masked_select(y_pred, mask.unsqueeze(0).unsqueeze(-1)).view(
            y_pred.shape[0], -1, y_pred.shape[-1]
        )
        y_mean_pred = y_pred.mean(-2)
        return ((y_mean_pred - y_mean) ** 2).mean()

    def training_step(self, batch, batch_idx):
        mse = self.shared_step(batch, batch_idx)
        self.log("loss_train", mse**0.5, prog_bar=True)
        return mse

    def validation_step(self, batch, batch_idx):
        mse = self.shared_step(batch, batch_idx)
        self.log("loss_val", mse**0.5, prog_bar=True)
        return mse

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
