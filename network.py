import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.nn import SAGEConv, LayerNorm

from module import drug_feature_module, drug_graph_module, cell_feature_module, interaction_module


class ISGDRP(nn.Module):
    def __init__(self, args):
        super(ISGDRP, self).__init__()
        self.embed_dim =args.embed_dim
        self.droprate = args.droprate
        self.drug_feature_module = drug_feature_module(args)
        self.drug_graph_module = drug_graph_module(args)
        self.cell_feature_module = cell_feature_module(args)
        self.drug_graph_emb = nn.Sequential(
            nn.Linear(self.embed_dim*3, self.embed_dim),
            nn.ELU(),
            nn.Dropout(self.droprate),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        )
        self.inter = interaction_module(args)
        self.channel_size = 64
        self.total_layer = nn.Linear(1408, self.channel_size * 4)
        self.total_bn = nn.BatchNorm1d((self.channel_size * 4 + 2 * self.embed_dim), momentum=0.5)
        self.con_layer =nn.Sequential(
            nn.Linear(self.channel_size * 4, 512),
            nn.ELU(),
            nn.Dropout(self.droprate),
            nn.Linear(512, 1)
        )
    def forward(self, drug_feature, drug_graph, cell_feature):
        d_graph = self.drug_graph_emb(self.drug_graph_module(drug_graph))

        d_drugs, d_feature = self.drug_feature_module(drug_feature,d_graph)
        c_cells, c_feature = self.cell_feature_module(cell_feature)

        # 内积外积模块
        Inner, Outer = self.inter(d_drugs, c_cells)
        total = torch.cat((d_feature, c_feature, Inner,  Outer), dim=1)
        total = F.relu(self.total_layer(total), inplace=True)
        total = F.dropout(total, p=self.droprate)

        regression = self.con_layer(total)
        return  regression.squeeze()#, auto_map, auto_map_AE

