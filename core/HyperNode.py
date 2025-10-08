import torch

import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing, GCNConv, GATConv
from core.Layer import *

import math

from torch_scatter import scatter
from torch_geometric.utils import softmax

from core.HypergraphTransformer import HypergraphTransformer
class HyperKGEConfig:
    dropout = 0
    num_layers= 2
    embedding_dim=500
    MLP_hidden=1000
    MLP_num_layers=2
    heads=4
    aggregate='mean'
    normalization='ln'
    deepset_input_norm=True
    deepset_input_norm=True
    GPR=False
    PMA=True
    lr=0.001
    wd=0.0
    conv_args_num_attention_heads = 4
    conv_args_residual_beta = 1
    conv_args_learn_beta = 0.001
    conv_args_conv_dropout_rate = 0.1
    conv_args_trans_method = "add"
    conv_args_negative_slope = 0.2
    conv_args_head_fusion_mode = "add"
    conv_args_edge_fusion_mode = "add"
    gamma = 10

class EdgeEmbedding(torch.nn.Module):
    def __init__(self, embed_size, fusion_type, num_edge_type):
        super(EdgeEmbedding, self).__init__()
        self.embed_size = embed_size
        self.fusion_type = fusion_type
        self.edge_type_embedding = nn.Embedding(num_edge_type, self.embed_size)
        self.output_embed_size = self.embed_size

    def forward(self, data):
        embedding_list = [self.edge_type_embedding(data.long())]

        if self.fusion_type == 'concat':
            self.output_embed_size = len(embedding_list) * self.embed_size
            return torch.cat(embedding_list, -1)
        elif self.fusion_type == 'add':
            return sum(embedding_list)
        else:
            raise ValueError(f"Get wrong fusion type {self.fusion_type}")

# 需要理解这段代码，具体是如何完成图卷积的模型的
class HyperNode(nn.Module):
    def __init__(self, args, n_node=0,n_hyper_edge=0):
        super(HyperNode, self).__init__()
        """
        args should contain the following:
        V_in_dim, V_enc_hid_dim, V_dec_hid_dim, V_out_dim, V_enc_num_layers, V_dec_num_layers
        E_in_dim, E_enc_hid_dim, E_dec_hid_dim, E_out_dim, E_enc_num_layers, E_dec_num_layers
        All_num_layers,dropout
        !!! V_in_dim should be the dimension of node features
        !!! E_out_dim should be the number of classes (for classification)
        """

#         Now set all dropout the same, but can be different
        self.All_num_layers = args.num_layers
        self.dropout = args.dropout
        self.aggr = args.aggregate
        self.NormLayer = args.normalization
        self.InputNorm = args.deepset_input_norm
        self.GPR = args.GPR
        self.num_node = n_node
        self.embedding_dim = args.embedding_dim

        self.hyperkgeConfig = args

        self.edge_type_embedding_layer = EdgeEmbedding(
            embed_size=self.embedding_dim,
            fusion_type="add",
            num_edge_type=4
        )
        self.edge_attr_embedding_layer = EdgeEmbedding(
            embed_size=self.embedding_dim,
            fusion_type="add",
            num_edge_type=4
        )
        self.E2EConvs = nn.ModuleList()
        self.bnE2Es = nn.ModuleList()

        # 节点-> 超边之间的卷积
        self.v2e = HypergraphTransformer(
                        in_channels=self.embedding_dim,
                        out_channels=self.embedding_dim,
                        attn_heads=args.conv_args_num_attention_heads,
                        residual_beta=args.conv_args_residual_beta,
                        learn_beta=args.conv_args_learn_beta,
                        dropout=args.conv_args_conv_dropout_rate,
                        trans_method=args.conv_args_trans_method,
                        edge_dim=self.embedding_dim,
                        rel_embed_dim=self.embedding_dim,
                        negative_slope=args.conv_args_negative_slope,
                        have_query_feature=True,
                        head_fusion_mode=args.conv_args_head_fusion_mode,
                        edge_fusion_mode=args.conv_args_edge_fusion_mode,
                    )
        self.v2e_bn = nn.BatchNorm1d(self.embedding_dim)

        self.e2v = HypergraphTransformer(
                        in_channels=self.embedding_dim,
                        out_channels=self.embedding_dim,
                        attn_heads=args.conv_args_num_attention_heads,
                        residual_beta=args.conv_args_residual_beta,
                        learn_beta=args.conv_args_learn_beta,
                        dropout=args.conv_args_conv_dropout_rate,
                        trans_method=args.conv_args_trans_method,
                        edge_dim=self.embedding_dim,
                        rel_embed_dim=self.embedding_dim,
                        negative_slope=args.conv_args_negative_slope,
                        have_query_feature=True,
                        head_fusion_mode=args.conv_args_head_fusion_mode,
                        edge_fusion_mode=args.conv_args_edge_fusion_mode,
                    )
        self.e2v_bn = nn.BatchNorm1d(self.embedding_dim)

        self.e2rel = HypergraphTransformer(
                        in_channels=self.embedding_dim,
                        out_channels=self.embedding_dim,
                        attn_heads=args.conv_args_num_attention_heads,
                        residual_beta=args.conv_args_residual_beta,
                        learn_beta=args.conv_args_learn_beta,
                        dropout=args.conv_args_conv_dropout_rate,
                        trans_method=args.conv_args_trans_method,
                        edge_dim=self.embedding_dim,
                        rel_embed_dim=self.embedding_dim,
                        negative_slope=args.conv_args_negative_slope,
                        have_query_feature=True,
                        head_fusion_mode=args.conv_args_head_fusion_mode,
                        edge_fusion_mode=args.conv_args_edge_fusion_mode,
                    )
        for _ in range(self.All_num_layers-1):
            self.E2EConvs.append(
                    HypergraphTransformer(
                        in_channels=self.embedding_dim,
                        out_channels=self.embedding_dim,
                        attn_heads=args.conv_args_num_attention_heads,
                        residual_beta=args.conv_args_residual_beta,
                        learn_beta=args.conv_args_learn_beta,
                        dropout=args.conv_args_conv_dropout_rate,
                        trans_method=args.conv_args_trans_method,
                        edge_dim=self.embedding_dim,
                        rel_embed_dim=self.embedding_dim,
                        negative_slope=args.conv_args_negative_slope,
                        have_query_feature=True,
                        head_fusion_mode=args.conv_args_head_fusion_mode,
                        edge_fusion_mode=args.conv_args_edge_fusion_mode,
                    )
            )
            self.bnE2Es.append(nn.BatchNorm1d(self.embedding_dim))

    def reset_parameters(self):
        for layer in self.E2EConvs:
            layer.reset_parameters()
        for layer in self.bnE2Es:
            layer.reset_parameters()
        self.v2e.reset_parameters()
        self.v2e_bn.reset_parameters()

        self.e2v.reset_parameters()
        self.e2v_bn.reset_parameters()

        self.e2rel.reset_parameters()
      
    def forward(self, n_id, x , adjs , cuda, mode="rel"):
        
        if cuda:
            x = x.cuda()
        # hyper_edge_emb = torch.zeros(
        #     split_idx,
        #     self.hyperkgeConfig.embedding_dim,
        #     device=x.device
        # )
        # x = torch.cat([hyper_edge_emb,x], dim=0)

        for idx in range(len(adjs)):
            adj_t, edge_attr,  edge_type, e_id, size = adjs[idx]
            edge_attr_embed, edge_type_embed = None, None
            x_target = x[:adj_t.size(0)]
            if cuda:
                if edge_type != None:
                    edge_type = edge_type.cuda()
                adj_t = adj_t.cuda()

            if edge_type is not None:
                edge_type_embed = self.edge_type_embedding_layer(edge_type)
                edge_attr_embed = self.edge_attr_embedding_layer(edge_type)

            if idx == 0:   # 第一层一定是实体到超边，
                x = self.v2e(
                        (x, x_target),
                        edge_index=adj_t,
                        edge_attr_embed=edge_attr_embed,
                        edge_type_embed=edge_type_embed,
                        edge_time_embed=None,
                        edge_dist_embed=None,
                    )
                x = self.v2e_bn(x)
            elif idx == len(adjs) -1 : #最后一层是 超边到实体或者超边到关系
                if mode == "rel":
                    x = self.e2rel(
                            (x, x_target),
                            edge_index=adj_t,
                            edge_attr_embed=edge_attr_embed,
                            edge_type_embed=edge_type_embed,
                            edge_time_embed=None,
                            edge_dist_embed=None,
                    )
                else:
                    x = self.e2v(
                            (x, x_target),
                            edge_index=adj_t,
                            edge_attr_embed=edge_attr_embed,
                            edge_type_embed=edge_type_embed,
                            edge_time_embed=None,
                            edge_dist_embed=None,
                    )
            else:
                x = self.E2EConvs[idx-1](
                    (x, x_target),
                    edge_index=adj_t,
                    edge_attr_embed=edge_attr_embed,
                    edge_type_embed=edge_type_embed,
                    edge_time_embed=None,
                    edge_dist_embed=None,
                )
                x = self.bnE2Es[idx-1](x)
        return x
