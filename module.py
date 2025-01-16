import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.nn import SAGEConv, LayerNorm, JumpingKnowledge

from en_decoder import Encoder_MultipleLayers
from torch_geometric.nn import GINConv, JumpingKnowledge

class interaction_module(nn.Module):
    def __init__(self, args):
        super(interaction_module, self).__init__()
        self.embed_dim = args.embed_dim
        self.number_map = 13*4
        self.dropout2 = args.droprate

        self.ii =MultiHeadAttention_inner(self.embed_dim, 8)
        self.hw = nn.Sequential(
                nn.Linear(6656, 1024),
                nn.ELU(),
                nn.Dropout(self.dropout2),
                nn.Linear(1024, 1024),
                nn.ELU(),
                nn.Dropout(self.dropout2),
                nn.Linear(1024,128)
        )


        self.channel_size = 64
        ###外积残差块
        self.Outer_product_rb_1 = nn.Sequential(
            nn.Conv2d(self.number_map, self.channel_size * 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.channel_size * 4),
            nn.ReLU(),
            nn.Conv2d(self.channel_size * 4, self.channel_size, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.channel_size)
        )
        self.Outer_product_downsample = nn.Sequential(
            nn.Conv2d(self.number_map, self.channel_size, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.channel_size)
        )

        self.Outer_product_conv = nn.Sequential(
            nn.Conv2d(self.channel_size, self.channel_size, kernel_size=1, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=4, padding=1),
        )
        self.Outer_product_rb_2 = nn.Sequential(
            nn.Conv2d(self.channel_size, self.channel_size * 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.channel_size * 4),
            nn.ReLU(),
            nn.Conv2d(self.channel_size * 4, self.channel_size, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.channel_size)
        )
        self.Outer_product_maxpool = nn.MaxPool2d(kernel_size=3, stride=4, padding=1)
        self.Outer_product_maxpool1 = nn.MaxPool2d(kernel_size=3, stride=4, padding=1)
        self.re = nn.ReLU()
    def forward(self, drugs,cells):
        # =============外积==========================================
        maps = []
        for i in range(len(drugs)):
            for j in range(len(cells)):
                maps.append(torch.bmm(drugs[i].unsqueeze(2), cells[j].unsqueeze(1)))
        Outer_product_map = maps[0].view((-1, 1, self.embed_dim, self.embed_dim))

        for i in range(1, len(maps)):
            interaction = maps[i].view((-1, 1, self.embed_dim, self.embed_dim))
            Outer_product_map = torch.cat([Outer_product_map, interaction], dim=1)

        # ===============内积============================================
        total = []

        for i in range(len(drugs)):
            for j in range(len(cells)):
                total.append(drugs[i].unsqueeze(1) * cells[j].unsqueeze(1))

        Inner_Product_map = total[0]

        for i in range(1, len(maps)):
            Inner_Product_map = torch.cat([Inner_Product_map, total[i]], dim=1)



        # ====================残差块=====================================================

        A, _, _ = Inner_Product_map.shape

        Inner_Product = self.ii(Inner_Product_map)
        Inner_Product = self.hw(Inner_Product.reshape(A, -1))



        #########外积残差

        x = self.Outer_product_downsample(Outer_product_map)
        Outer_product_feature_map = self.Outer_product_rb_1(Outer_product_map)
        Outer_product_feature_map = Outer_product_feature_map + x
        Outer_product_feature_map = self.re(Outer_product_feature_map)

        Outer_product_feature_map = self.Outer_product_conv(Outer_product_feature_map)

        x = Outer_product_feature_map
        Outer_product_feature_map = self.Outer_product_rb_2(Outer_product_feature_map)
        Outer_product_feature_map = Outer_product_feature_map + x
        Outer_product_feature_map = self.re(Outer_product_feature_map)

        Outer_product_feature_map = self.Outer_product_maxpool(Outer_product_feature_map)

        Outer_product = Outer_product_feature_map.view((drugs[0].shape[0], -1))

        return Inner_Product, Outer_product

class drug_feature_module(nn.Module):
    def __init__(self, args):
        super(drug_feature_module, self).__init__()
        self.embed_dim = args.embed_dim
        self.droprate = args.droprate
        self.drug_1 = nn.Sequential(
            # 580
            nn.Linear(5181, self.embed_dim),
            nn.ELU(),
            nn.Dropout(self.droprate),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        )
        self.drug_2 = nn.Sequential(
            # 1024
            nn.Linear(5181, self.embed_dim),
            nn.ELU(),
            nn.Dropout(self.droprate),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        )
        self.drug_3 = nn.Sequential(
            # 315
            nn.Linear(5181, self.embed_dim),
            nn.ELU(),
            nn.Dropout(self.droprate),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        )
        self.drug_4 = nn.Sequential(
            # 2586
            nn.Linear(5181, self.embed_dim),
            nn.ELU(),
            nn.Dropout(self.droprate),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        )
        self.drug_5 = nn.Sequential(
            # 881
            nn.Linear(5181, self.embed_dim),
            nn.ELU(),
            nn.Dropout(self.droprate),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        )
        self.drug_6 = nn.Sequential(
            # 200
            nn.Linear(5181, self.embed_dim),
            nn.ELU(),
            nn.Dropout(self.droprate),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        )
        self.drug_7 = nn.Sequential(
            nn.Linear(5181, self.embed_dim),
            nn.ELU(),
            nn.Dropout(self.droprate),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        )
        self.drug_8 = nn.Sequential(
            # 4693
            nn.Linear(5181, self.embed_dim),
            nn.ELU(),
            nn.Dropout(self.droprate),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        )
        self.drug_9 = nn.Sequential(
            # 822
            nn.Linear(5181, self.embed_dim),
            nn.ELU(),
            nn.Dropout(self.droprate),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        )
        self.drug_10 = nn.Sequential(
            # 636
            nn.Linear(5181, self.embed_dim),
            nn.ELU(),
            nn.Dropout(self.droprate),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        )
        self.drug_11 = nn.Sequential(
            # 170
            nn.Linear(5181, self.embed_dim),
            nn.ELU(),
            nn.Dropout(self.droprate),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        )
        self.drug_12 = nn.Sequential(
            # 170
            nn.Linear(5181, self.embed_dim),
            nn.ELU(),
            nn.Dropout(self.droprate),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        )
        self.transformer = Transformer(args)
        self.drug_emb = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ELU(),
            nn.Dropout(self.droprate),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Dropout(self.droprate),
            nn.Linear(256, 128)
        )
    def forward(self, drug_feature,d_graph):
        # # 580  1024   315  2586   881  200  5181  4693  822   636  170

        drug1,drug2,drug3,drug4,drug5,drug6,drug7,drug8,drug9,drug10,drug11,drug12 = drug_feature.chunk(12,1)
        drug_1 = self.drug_1(drug1.float())
        drug_2 = self.drug_2(drug2.float())
        drug_3 = self.drug_3(drug3.float())
        drug_4 = self.drug_4(drug4.float())
        drug_5 = self.drug_5(drug5.float())
        drug_6 = self.drug_6(drug6.float())
        drug_7 = self.drug_7(drug7.float())
        drug_8 = self.drug_8(drug8.float())
        drug_9 = self.drug_9(drug9.float())
        drug_10 = self.drug_10(drug10.float())
        drug_11 = self.drug_11(drug11.float())
        drug_12 = self.drug_12(drug12.float())

        drugs = [drug_1, drug_2, drug_3, drug_4, drug_5, drug_6, drug_7, drug_8, drug_9, drug_10, drug_11,drug_12]
   
        drugs_feature = [drug+d_graph for drug in drugs]

        drugs.append(d_graph)
        drugs_feature = torch.stack(drugs_feature,1)

        drugs_feature_new = self.transformer(drugs_feature)
        A, B, _ = drugs_feature_new.shape
        x_drug = self.drug_emb(drugs_feature_new.reshape(A, -1))

        return drugs, x_drug


class cell_feature_module(nn.Module):
    def __init__(self, args):
        super(cell_feature_module, self).__init__()
        self.embed_dim =args.embed_dim
        self.droprate = args.droprate
        self.cell_1 = nn.Sequential(
            nn.Linear(706, self.embed_dim),
            nn.ELU(),
            nn.Dropout(self.droprate),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        )
        self.cell_2 = nn.Sequential(
            nn.Linear(706, self.embed_dim),
            nn.ELU(),
            nn.Dropout(self.droprate),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        )
        self.cell_3 = nn.Sequential(
            nn.Linear(706, self.embed_dim),
            nn.ELU(),
            nn.Dropout(self.droprate),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        )
        self.cell_4 = nn.Sequential(
            nn.Linear(170, self.embed_dim),
            nn.ELU(),
            nn.Dropout(self.droprate),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        )
        self.cell_MHA = MultiHeadAttention(self.embed_dim, 8)

        self.cell_emb = nn.Sequential(
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Dropout(self.droprate),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(self.droprate),
            nn.Linear(128, 128)
        )

    def forward(self, cell_feature):
        cell1 = cell_feature[:, :706]
        cell2 = cell_feature[:, 706:1412]
        cell3 = cell_feature[:, 1412:2118]
        cell4 = cell_feature[:, 2118:]
        cell1 = F.relu(self.cell_1(cell1.float()))
        cell2 = F.relu(self.cell_2(cell2.float()))
        cell3 = F.relu(self.cell_3(cell3.float()))
        cell4 = F.relu(self.cell_4(cell4.float()))
        cells = [cell1, cell2, cell3, cell4]

        cells_feature = torch.stack(cells,1)

        cells_feature = self.cell_MHA(cells_feature)
        A, _, _ = cells_feature.shape
        x_cell = self.cell_emb(cells_feature.reshape(A, -1))
        cells = torch.unbind(cells_feature,1)
        return cells,x_cell


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):

        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.ouput_dim, self.ouput_dim, bias=False)

    def forward(self, X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(X.shape[0],-1, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        K = self.W_K(X).view(X.shape[0],-1, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        V = self.W_V(X).view(X.shape[0],-1, self.n_heads, self.d_v).permute(0, 2, 1, 3)

        scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # context: [len_q, n_heads * d_v]
        context =context.permute(0, 2, 1, 3).contiguous()
        context = context.view(X.shape[0], -1, self.n_heads * (self.ouput_dim // self.n_heads))
        output = self.fc(context)
        output = output + X
        return output


class MultiHeadAttention_inner(torch.nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):

        super(MultiHeadAttention_inner, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.ouput_dim, self.ouput_dim, bias=False)

    def forward(self, X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(X.shape[0],-1, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        K = self.W_K(X).view(X.shape[0],-1, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        V = self.W_V(X).view(X.shape[0],-1, self.n_heads, self.d_v).permute(0, 2, 1, 3)

        scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # context: [len_q, n_heads * d_v]
        context =context.permute(0, 2, 1, 3).contiguous()
        context = context.view(X.shape[0], -1, self.n_heads * (self.ouput_dim // self.n_heads))
        output = self.fc(context)
        output = output + X
        return output


class drug_graph_module(nn.Module):
    def __init__(self, args):
        super(drug_graph_module, self).__init__()
        self.layer_drug = args.layer_drug
        self.dim_drug = 128
        self.JK = JumpingKnowledge('cat')
        self.convs_drug = torch.nn.ModuleList()
        self.bns_drug = torch.nn.ModuleList()
        # self.emb_drug = torch.nn.ModuleList()
        for i in range(self.layer_drug):
            if i:
                block = nn.Sequential(nn.Linear(self.dim_drug, self.dim_drug), nn.ReLU(),
                                      nn.Linear(self.dim_drug, self.dim_drug))
            else:
                block = nn.Sequential(nn.Linear(5181, self.dim_drug), nn.ReLU(),
                                      nn.Linear(self.dim_drug, self.dim_drug))
            conv = GINConv(block)
            bn = torch.nn.BatchNorm1d(self.dim_drug)
            # self.emb_drug.append(block)
            self.convs_drug.append(conv)
            self.bns_drug.append(bn)

    def forward(self, drug_graph):
        x, edge_index, batch = drug_graph.x, drug_graph.edge_index, drug_graph.batch
        x_drug_list = []
        for i in range(self.layer_drug):
            x = F.relu(self.convs_drug[i](x, edge_index))
            x = self.bns_drug[i](x)
            x_drug_list.append(x)

        node_representation = self.JK(x_drug_list)
        # xxx = node_representation.detach().cpu().numpy()
        index = torch.arange(0, len(batch), 7)
        x_drug = node_representation[index]
        return x_drug


# transformer for drug feature and cross-attention module
class Transformer(nn.Sequential):
    def __init__(self, args):
        super(Transformer, self).__init__()
        input_dim_drug = 2586
        transformer_emb_size_drug = 128
        transformer_dropout_rate = 0.1
        transformer_n_layer_drug = 8
        transformer_intermediate_size_drug = 512
        transformer_num_attention_heads_drug = 8
        transformer_attention_probs_dropout = 0.1
        transformer_hidden_dropout_rate = 0.1

        self.device = args.device
        self.encoder = Encoder_MultipleLayers(transformer_n_layer_drug,
                                              transformer_emb_size_drug,
                                              transformer_intermediate_size_drug,
                                              transformer_num_attention_heads_drug,
                                              transformer_attention_probs_dropout,
                                              transformer_hidden_dropout_rate)


    def forward(self, v):

        e = v[0].long().to(self.device)
        e_mask = v[:, :, 1].long().to(self.device)
        ex_e_mask = e_mask.unsqueeze(1).unsqueeze(2)
        ex_e_mask = (1.0 - ex_e_mask) * -10000.0

        encoded_layers = self.encoder(v, ex_e_mask.float())
        return encoded_layers
