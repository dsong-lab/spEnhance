import math
import torch
import numpy as np
import pytorch_lightning as pl
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
from utils import get_disk_mask
from scipy.spatial.distance import cdist

from model import get_odj, GraphConvLayer, Linear

class scstGCN(pl.LightningModule):
    def __init__(self, lr, num_features, num_genes, ori_radius, bias=False):
        super(scstGCN, self).__init__()

        self.lr = lr
        self.ori_radius = ori_radius

        self.shared = nn.Sequential(GraphConvLayer(num_features, 512))
        
        self.output = nn.ModuleList([
            nn.Sequential(GraphConvLayer(512, 512),
                          nn.Dropout(0.5),
                          Linear(512, num, alpha=0.01, beta=0.01, bias=bias)) for num in num_genes])

        self.save_hyperparameters()

    def forward(self, x, indices=None):
        x = self.shared(x)
        # x = F.dropout(x, 0.5, training=self.training)
        x = [head(x) for head in self.output]
        out = torch.cat(x, dim=-1) 
        
        return out

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
        loss = mse**0.5
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log('loss_train', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log('loss_val', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)

        return optimizer
