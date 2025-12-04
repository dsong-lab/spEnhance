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

        self.GCN_module = nn.Sequential(
            GraphConvLayer(num_features, 512),
            GraphConvLayer(512, 512))

        self.output_module = Linear(512, num_genes, alpha=0.01, beta=0.01, bias=bias)

        self.save_hyperparameters()

    def get_hidden(self, x):
        x = self.GCN_module.forward(x)
        x = F.dropout(x, 0.5, training=self.training)
        return x

    def get_gene(self, x, indices=None):
        x = self.output_module.forward(x, indices)
        return x

    def forward(self, x, indices=None):
        x = self.get_hidden(x)
        x = self.get_gene(x, indices)
        return x

    def shared_step(self, batch, batch_idx):
        x, y_mean = batch
        mask = get_disk_mask(self.ori_radius/16)
        mask = torch.BoolTensor(mask).to('cuda')
        y_pred = self.forward(x)
        y_pred = y_pred.reshape(y_pred.shape[0], mask.shape[0], mask.shape[1], y_pred.shape[2])
        y_pred = torch.masked_select(y_pred, mask.unsqueeze(0).unsqueeze(-1)).view(y_pred.shape[0], -1, y_pred.shape[-1])

        y_mean_pred = y_pred.mean(-2)

        mse = ((y_mean_pred - y_mean)**2).mean()
        return mse**0.5

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
