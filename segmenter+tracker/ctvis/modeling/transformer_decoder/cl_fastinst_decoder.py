import math

import torch
from detectron2.config import configurable
from torch import nn
from torch.nn import functional as F

from fastinst.utils.misc import nested_tensor_from_tensor_list
from fastinst.modeling.transformer_decoder.utils import TRANSFORMER_DECODER_REGISTRY, QueryProposal, \
    CrossAttentionLayer, SelfAttentionLayer, FFNLayer, MLP
import fvcore.nn.weight_init as weight_init

#from .tracker import Tracker

@TRANSFORMER_DECODER_REGISTRY.register()
class CLFastInstDecoder(nn.Module):

    @configurable
    def __init__(
            self,
            in_channels,
            *,
            num_classes: int,
            hidden_dim: int,
            num_queries: int,
            num_aux_queries: int,
            nheads: int,
            dim_feedforward: int,
            dec_layers: int,
            pre_norm: bool,
            mask_dim: int,
            
            reid_hidden_dim,
            num_reid_head_layers,
    ):
        """
        Args:
            in_channels: channels of the input features
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            num_aux_queries: number of auxiliary queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
        """
        super().__init__()
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.num_queries = num_queries
        self.num_aux_queries = num_aux_queries
        self.criterion = None

        meta_pos_size = int(round(math.sqrt(self.num_queries)))
        self.meta_pos_embed = nn.Parameter(torch.empty(1, hidden_dim, meta_pos_size, meta_pos_size))
        if num_aux_queries > 0:
            self.empty_query_features = nn.Embedding(num_aux_queries, hidden_dim)
            self.empty_query_pos_embed = nn.Embedding(num_aux_queries, hidden_dim)

        self.query_proposal = QueryProposal(hidden_dim, num_queries, num_classes)

        self.transformer_query_cross_attention_layers = nn.ModuleList()
        self.transformer_query_self_attention_layers = nn.ModuleList()
        self.transformer_query_ffn_layers = nn.ModuleList()
        self.transformer_mask_cross_attention_layers = nn.ModuleList()
        self.transformer_mask_ffn_layers = nn.ModuleList()
        for idx in range(self.num_layers):
            self.transformer_query_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm
                )
            )
            self.transformer_query_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm
                )
            )
            self.transformer_query_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0, normalize_before=pre_norm
                )
            )
            self.transformer_mask_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm
                )
            )
            self.transformer_mask_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0, normalize_before=pre_norm
                )
            )

        self.decoder_query_norm_layers = nn.ModuleList()
        self.class_embed_layers = nn.ModuleList()
        self.mask_embed_layers = nn.ModuleList()
        self.mask_features_layers = nn.ModuleList()
        for idx in range(self.num_layers + 1):
            self.decoder_query_norm_layers.append(nn.LayerNorm(hidden_dim))
            self.class_embed_layers.append(MLP(hidden_dim, hidden_dim, num_classes + 1, 3))
            self.mask_embed_layers.append(MLP(hidden_dim, hidden_dim, mask_dim, 3))
            self.mask_features_layers.append(nn.Linear(hidden_dim, mask_dim))
            
        if num_reid_head_layers > 0:
            self.reid_embed = MLP(
                hidden_dim, reid_hidden_dim, hidden_dim, num_reid_head_layers)
            for layer in self.reid_embed.layers:
                weight_init.c2_xavier_fill(layer)
        else:
            self.reid_embed = nn.Identity()  # do nothing

        #self.last_query = None
        #self.last_embed = None
        #self.tracker = Tracker()

    @classmethod
    def from_config(cls, cfg, in_channels, input_shape):
        ret = {}
        ret["in_channels"] = in_channels

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.FASTINST.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.FASTINST.NUM_OBJECT_QUERIES
        ret["num_aux_queries"] = cfg.MODEL.FASTINST.NUM_AUX_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.FASTINST.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.FASTINST.DIM_FEEDFORWARD

        ret["dec_layers"] = cfg.MODEL.FASTINST.DEC_LAYERS
        ret["pre_norm"] = cfg.MODEL.FASTINST.PRE_NORM

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        
        ret['reid_hidden_dim']=cfg.MODEL.FASTINST.REID_HIDDEN_DIM
        ret['num_reid_head_layers']=cfg.MODEL.FASTINST.NUM_REID_HEAD_LAYERS
         
        return ret
    
    def reset_reference(self):
        #del self.last_query
        #del self.last_embed
        #self.last_query = None
        #self.last_embed = None
        return

    def forward(self, x, mask_features, targets=None,isfirst=False): # isfirst=True 视频第一帧
        if isfirst==True:
            self.reset_reference()

        bs = x[0].shape[0]
        proposal_size = x[1].shape[-2:]
        pixel_feature_size = x[2].shape[-2:]

        pixel_pos_embeds = F.interpolate(self.meta_pos_embed, size=pixel_feature_size,
                                         mode="bilinear", align_corners=False)
        proposal_pos_embeds = F.interpolate(self.meta_pos_embed, size=proposal_size,
                                            mode="bilinear", align_corners=False)

        pixel_features = x[2].flatten(2).permute(2, 0, 1)
        pixel_pos_embeds = pixel_pos_embeds.flatten(2).permute(2, 0, 1)

        query_features, query_pos_embeds, query_locations, proposal_cls_logits = self.query_proposal(
            x[1], proposal_pos_embeds
        )
        query_features = query_features.permute(2, 0, 1)
        query_pos_embeds = query_pos_embeds.permute(2, 0, 1)
        if self.num_aux_queries > 0:
            aux_query_features = self.empty_query_features.weight.unsqueeze(1).repeat(1, bs, 1)
            aux_query_pos_embed = self.empty_query_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            query_features = torch.cat([query_features, aux_query_features], dim=0)
            query_pos_embeds = torch.cat([query_pos_embeds, aux_query_pos_embed], dim=0)     

        outputs_class, outputs_mask, attn_mask, _, _ ,outputs_query, outputs_embed,_,_= self.forward_prediction_heads(
            query_features, pixel_features, pixel_feature_size, -1, return_attn_mask=True
        )
        predictions_class = [outputs_class]
        predictions_mask = [outputs_mask]
        predictions_matching_index = [None]
        query_feature_memory = [query_features]
        pixel_feature_memory = [pixel_features]
        
        predictions_query=[outputs_query]
        predictions_embed=[outputs_embed]

        for i in range(self.num_layers):
            query_features, pixel_features = self.forward_one_layer(
                query_features, pixel_features, query_pos_embeds, pixel_pos_embeds, attn_mask, i
            )
            if i < self.num_layers - 1:
                outputs_class, outputs_mask, attn_mask, _, _ ,outputs_query, outputs_embed,_,_= self.forward_prediction_heads(
                    query_features, pixel_features, pixel_feature_size, i, return_attn_mask=True,
                )
            else:
                # output query_without_norm
                outputs_class, outputs_mask, _, matching_indices, gt_attn_mask,outputs_query, outputs_embed,outputs_query_without_norm,seg_mask_features = self.forward_prediction_heads(
                    query_features, pixel_features, pixel_feature_size, i,
                    return_gt_attn_mask=self.training, targets=targets, query_locations=query_locations
                )
                
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_matching_index.append(None)
            query_feature_memory.append(query_features)
            pixel_feature_memory.append(pixel_features)
            
            predictions_query.append(outputs_query)
            predictions_embed.append(outputs_embed)

        # 之后为GT-mask guided

        guided_predictions_class = []
        guided_predictions_mask = []
        guided_predictions_matching_index = []
        
        guided_predictions_query=[]
        guided_predictions_embed=[]
        if self.training:
            for i in range(self.num_layers):
                query_features, pixel_features = self.forward_one_layer(
                    query_feature_memory[i + 1], pixel_feature_memory[i + 1], query_pos_embeds,
                    pixel_pos_embeds, gt_attn_mask, i
                )

                outputs_class, outputs_mask, _, _, _,outputs_query, outputs_embed,_,_ = self.forward_prediction_heads(
                    query_features, pixel_features, pixel_feature_size, idx_layer=i
                )

                guided_predictions_class.append(outputs_class)
                guided_predictions_mask.append(outputs_mask)
                guided_predictions_matching_index.append(matching_indices)
               
                guided_predictions_query.append(outputs_query)
                guided_predictions_embed.append(outputs_embed)

        predictions_class = guided_predictions_class + predictions_class
        predictions_mask = guided_predictions_mask + predictions_mask
        predictions_matching_index = guided_predictions_matching_index + predictions_matching_index
      
        predictions_query = guided_predictions_query + predictions_query
        predictions_embed = guided_predictions_embed + predictions_embed

        out = {
            'proposal_cls_logits': proposal_cls_logits,
            'query_locations': query_locations,
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'pred_matching_indices': predictions_matching_index[-1],
            'pred_queries': predictions_query[-1],#
            'pred_embeds': predictions_embed[-1],#
            'aux_outputs': self._set_aux_loss(
                predictions_class, predictions_mask, predictions_matching_index, query_locations,predictions_query,
                predictions_embed
            ),
            # for td_tracker
            'pred_queries_without_norm':outputs_query_without_norm,
            'mask_features':seg_mask_features # the tensor used to calc the mask (mask_features in M2F) is x[2]
        }
        return out

    def forward_one_layer(self, query_features, pixel_features, query_pos_embeds, pixel_pos_embeds, attn_mask, i):
        pixel_features = self.transformer_mask_cross_attention_layers[i](
            pixel_features, query_features, query_pos=pixel_pos_embeds, pos=query_pos_embeds
        )
        pixel_features = self.transformer_mask_ffn_layers[i](pixel_features)

        query_features = self.transformer_query_cross_attention_layers[i](
            query_features, pixel_features, memory_mask=attn_mask, query_pos=query_pos_embeds, pos=pixel_pos_embeds
        )
        query_features = self.transformer_query_self_attention_layers[i](
            query_features, query_pos=query_pos_embeds
        )
        query_features = self.transformer_query_ffn_layers[i](query_features)
        return query_features, pixel_features

    def forward_prediction_heads(self, query_features, pixel_features, pixel_feature_size, idx_layer,
                                 return_attn_mask=False, return_gt_attn_mask=False,
                                 targets=None, query_locations=None):
        decoder_query_features = self.decoder_query_norm_layers[idx_layer + 1](query_features[:self.num_queries])
        # to get queries without norm
        decoder_query_features_without_norm = query_features[:self.num_queries]
        decoder_query_features_without_norm = decoder_query_features_without_norm.transpose(0, 1) #
        #
        decoder_query_features = decoder_query_features.transpose(0, 1)
        if self.training or idx_layer + 1 == self.num_layers:
            outputs_class = self.class_embed_layers[idx_layer + 1](decoder_query_features)
        else:
            outputs_class = None
        outputs_mask_embed = self.mask_embed_layers[idx_layer + 1](decoder_query_features)
        outputs_mask_features = self.mask_features_layers[idx_layer + 1](pixel_features.transpose(0, 1))
        
        reid_embed=self.reid_embed(decoder_query_features)

        outputs_mask = torch.einsum("bqc,blc->bql", outputs_mask_embed, outputs_mask_features)
        outputs_mask = outputs_mask.reshape(-1, self.num_queries, *pixel_feature_size)

        seg_mask_features = outputs_mask_features.reshape(-1,*pixel_feature_size,256).permute(0,3,1,2) # (b,c=256,h,w)

        if return_attn_mask:
            # outputs_mask.shape: b, q, h, w
            attn_mask = F.pad(outputs_mask, (0, 0, 0, 0, 0, self.num_aux_queries), "constant", 1)
            attn_mask = (attn_mask < 0.).flatten(2)  # b, q, hw
            invalid_query = attn_mask.all(-1, keepdim=True)  # b, q, 1
            attn_mask = (~ invalid_query) & attn_mask  # b, q, hw
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            attn_mask = attn_mask.detach()
        else:
            attn_mask = None

        if return_gt_attn_mask:
            assert targets is not None and query_locations is not None
            matching_indices = self.criterion.matcher(
                {'pred_logits': outputs_class, 'pred_masks': outputs_mask,
                 'query_locations': query_locations}, targets)
            src_idx = self.criterion._get_src_permutation_idx(matching_indices)
            tgt_idx = self.criterion._get_tgt_permutation_idx(matching_indices)

            mask = [t["masks"] for t in targets]
            target_mask, valid = nested_tensor_from_tensor_list(mask).decompose()
            if target_mask.shape[1] > 0:
                target_mask = target_mask.to(outputs_mask)
                target_mask = F.interpolate(target_mask, size=pixel_feature_size, mode="nearest").bool()
            else:
                target_mask_size = [target_mask.shape[0], target_mask.shape[1], *pixel_feature_size]
                target_mask = torch.zeros(size=target_mask_size, device=outputs_mask.device).bool()

            gt_attn_mask_size = [
                outputs_mask.shape[0], self.num_queries + self.num_aux_queries, *pixel_feature_size
            ]
            gt_attn_mask = torch.zeros(size=gt_attn_mask_size, device=outputs_mask.device).bool()
            gt_attn_mask[src_idx] = ~ target_mask[tgt_idx]
            gt_attn_mask = gt_attn_mask.flatten(2)

            invalid_gt_query = gt_attn_mask.all(-1, keepdim=True)  # b, n, 1
            gt_attn_mask = (~invalid_gt_query) & gt_attn_mask
            gt_attn_mask = gt_attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            gt_attn_mask = gt_attn_mask.detach()
        else:
            matching_indices = None
            gt_attn_mask = None

        return outputs_class, outputs_mask, attn_mask, matching_indices, gt_attn_mask,decoder_query_features, reid_embed,decoder_query_features_without_norm,seg_mask_features

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, output_indices, output_query_locations,outputs_queries, outputs_embeds):
        return [
            {
                "query_locations": output_query_locations,
                "pred_logits": a,
                "pred_masks": b,
                "pred_matching_indices": c,
                "pred_queries": d, "pred_embeds": e}
            for a, b, c,d,e in zip(outputs_class[:-1], outputs_seg_masks[:-1], output_indices[:-1],outputs_queries[:-1], outputs_embeds[:-1])
        ]
