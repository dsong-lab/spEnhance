import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch.optim import Adam

from utils import get_disk_mask

import torch
import numpy as np
from scipy.spatial.distance import cdist

def build_grid_edge_index(H: int, W: int, k: int = 4, self_loop: bool = False, device=None):

    coordinates = np.array([[i, j] for i in range(H) for j in range(W)])
    
    distances = cdist(coordinates, coordinates)  # (H*W, H*W)
    
    edges = []
    for i in range(H * W):
        sorted_indices = np.argsort(distances[i])
        nearest_indices = sorted_indices[1:k+1]  
        
        for j in nearest_indices:
            edges.append((i, j))
            edges.append((j, i)) 
    
    if self_loop:
        for i in range(H * W):
            edges.append((i, i))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    if device is not None:
        edge_index = edge_index.to(device)
    
    return edge_index


def batch_edge_index(edge_index, batch_size, num_nodes):
        edge_list = []
        for b in range(batch_size):
            edge_list.append(edge_index + b * num_nodes)
        edge_index_batch = torch.cat(edge_list, dim=1)
        return edge_index_batch

class Linear(nn.Module):
    def __init__(self, num_hidden, num_genes, alpha=0.01, beta=0.01, bias=True):
        super().__init__()
        self.linear = nn.Linear(num_hidden, num_genes, bias=bias)
        self.act = nn.ELU(alpha)
        self.beta = beta

    def forward(self, x):
        return self.act(self.linear(x)) + self.beta

class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, residual=True):
        super().__init__()
        self.conv = GCNConv(in_dim, out_dim, bias=bias, add_self_loops=True)
        self.residual = residual and (in_dim == out_dim)
        if not self.residual and residual:
            self.res_proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, edge_index):
        res = x
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.1)
        if self.residual:
            x = x + res
        elif hasattr(self, 'res_proj'):
            x = x + self.res_proj(res)
        return x

class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, heads=1, bias=False, residual=True):
        super().__init__()
        self.conv = GATConv(in_dim, out_dim, heads=heads, concat=False, bias=bias)
        self.residual = residual and (in_dim == out_dim)
        if not self.residual and residual:
            self.res_proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, edge_index):
        res = x
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.1)
        if self.residual:
            x = x + res
        elif hasattr(self, 'res_proj'):
            x = x + self.res_proj(res)
        return x
        
class scstGCN(pl.LightningModule):
    def __init__(self, lr, num_features, num_genes, ori_radius, bias=False):
        super(scstGCN, self).__init__()

        self.lr = lr
        self.ori_radius = ori_radius

        # ===== Shared GraphConv layers =====
        self.conv1 = GCN(num_features, 512, bias=bias)
        self.conv2 = GCN(512, 512, bias=bias)

        # ===== Output heads =====
        self.output = nn.ModuleList([
            Linear(512, num, alpha=0.01, beta=0.01, bias=bias) for num in num_genes
        ])

        self.save_hyperparameters()

    def forward(self, x):
        """
        x: [B, L*L, C]
        """
        x = x.float()
        B, N, C = x.shape
        L = int(N ** 0.5)
        
        edge_index_single = build_grid_edge_index(L, L)
        edge_index = batch_edge_index(edge_index_single, B, N)

        x = x.reshape(B * N, C)

        x = x.to('cuda')
        edge_index = edge_index.to('cuda')
        # shared layers
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)

        x = F.dropout(x, 0.5, training=self.training)

        # multi-head
        outs = [head(x) for head in self.output]
        out = torch.cat(outs, dim=-1)
        out = out.view(B, N, -1)
        return out # [B, L*L, G]

    def shared_step(self, batch, batch_idx):
        x, y_mean = batch
        mask = get_disk_mask(self.ori_radius/16)
        mask = torch.BoolTensor(mask).to('cuda')
        
        y_pred = self.forward(x)
        y_pred = y_pred.reshape(y_pred.shape[0], mask.shape[0], mask.shape[1], y_pred.shape[2])
        y_pred = torch.masked_select(y_pred, mask.unsqueeze(0).unsqueeze(-1)).view(y_pred.shape[0], -1, y_pred.shape[-1])

        y_mean_pred = y_pred.mean(-2)

        mse = ((y_mean_pred - y_mean)**2).mean()
        # loss_fn = nn.HuberLoss(delta=1.0, reduction="mean")
        
        # loss = loss_fn(y_mean_pred, y_mean)
        loss = mse
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log('loss_train', loss**0.5, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log('loss_val', loss**0.5, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer
