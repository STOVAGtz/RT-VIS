import torch
from detectron2.config import configurable
from torch import nn
from torch.nn import functional as F
from fastinst.modeling.transformer_decoder.utils import CrossAttentionLayer, SelfAttentionLayer, FFNLayer, MLP
import fvcore.nn.weight_init as weight_init
from .noiser import Noiser
from .referringcrossattn import ReferringCrossAttentionLayer

class Tracker(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.num_heads = 8
        self.hidden_dim = 256
        self.num_layers = 3
        self.dim_feedforward = 1024
        self.pre_norm = False
        self.mask_dim = 256
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=self.hidden_dim,
                    nhead=self.num_heads,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
            self.transformer_cross_attention_layers.append(
                ReferringCrossAttentionLayer(
                    d_model=self.hidden_dim,
                    nhead=self.num_heads,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=self.hidden_dim,
                    dim_feedforward=self.dim_feedforward,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
        self.noiser = Noiser()

    def forward(self,query_features,last_query,last_embed):
        if last_query!=None:
            last_query = last_query.transpose(0,1)
        if last_embed!=None:
            last_embed = last_embed.transpose(0,1)
        # last_query [100,b,256]
        # last_embed [100,b,256]

        ref = last_embed # 使用last_embed作为引导
        ms_output = []
        for j in range(self.num_layers):
            if j == 0:
                indices, noised_init = self.noiser(
                    last_query if last_query!=None else query_features,
                    query_features,
                    cur_embeds_no_norm=None, # None
                    activate=self.training if ref!=None else False,
                    cur_classes=None, # None
                )
                output = self.transformer_cross_attention_layers[j]( # (id,q,k,v,...)
                    noised_init, ref if ref!=None else query_features, ##
                    query_features, query_features, # 
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None, query_pos=None
                )

                output = self.transformer_self_attention_layers[j](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=None
                )
                # FFN
                output = self.transformer_ffn_layers[j](
                    output
                )
                ms_output.append(output)
            else:
                output = self.transformer_cross_attention_layers[j]( # (id,q,k,v,...)
                    ms_output[-1], ref if ref!=None else query_features,
                    query_features, query_features,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None, query_pos=None
                )

                output = self.transformer_self_attention_layers[j](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=None
                )
                # FFN
                output = self.transformer_ffn_layers[j](
                    output
                )
                ms_output.append(output)
        return indices, output