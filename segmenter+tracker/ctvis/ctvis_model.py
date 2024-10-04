import logging
import math

import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import ImageList, BitMasks

#from mask2former import MaskFormer
from fastinst import FastInst
from .utils import retry_if_cuda_oom
from .modeling.cl_plugin import build_cl_plugin
from .modeling.tracker import build_tracker

from ctvis.modeling.td_tracker.tracker import Tracker
from ctvis.modeling.criterion import VideoSetCriterion
from ctvis.modeling.matcher import VideoHungarianMatcher, VideoHungarianMatcher_Consistent
import einops
import torch.nn as nn

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
#class CTVISModel(MaskFormer):
class CTVISModel(FastInst):
    @configurable
    def __init__(
            self,
            num_frames,
            num_topk,
            num_clip_frames,
            to_cpu_frames,
            test_interpolate_chunk_size,
            test_instance_chunk_size,
            cl_plugin,
            tracker,
            criterion_td: torch.nn.Module,
            max_iter_num,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.cl_plugin = cl_plugin
        self.tracker = tracker

        self.num_frames = num_frames
        self.num_topk = num_topk
        self.num_clip_frames = num_clip_frames
        self.to_cpu_frames = to_cpu_frames
        self.test_interpolate_chunk_size = test_interpolate_chunk_size
        self.test_instance_chunk_size = test_instance_chunk_size

        self.td_tracker = Tracker()
        #criterion =  torch.nn.Module()
        self.criterion = criterion_td
        self.use_cl = True
        self.iter = 0
        #self.max_iter_num = 10000
        self.max_iter_num = max_iter_num #
        #self.max_num = self.num_topk
        self.max_num = 10 # the same as DVIS++

        # frozen the segmenter
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.sem_seg_head.parameters():
            p.requires_grad_(False)

    @classmethod
    def from_config(cls, cfg):
        #rets = MaskFormer.from_config(cfg)  # noqa
        rets = FastInst.from_config(cfg)  # noqa

        cl_plugin = build_cl_plugin(cfg)  # train
        tracker = build_tracker(cfg)  # inference

        num_frames = cfg.INPUT.SAMPLING_FRAME_NUM
        num_topk = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        num_clip_frames = cfg.TEST.NUM_CLIP_FRAMES
        to_cpu_frames = cfg.TEST.TO_CPU_FRAMES
        test_interpolate_chunk_size = cfg.TEST.TEST_INTERPOLATE_CHUNK_SIZE
        test_instance_chunk_size = cfg.TEST.TEST_INSTANCE_CHUNK_SIZE

        # Loss parameters:
        deep_supervision = cfg.MODEL.FASTINST.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.FASTINST.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.FASTINST.CLASS_WEIGHT
        dice_weight = cfg.MODEL.FASTINST.DICE_WEIGHT
        mask_weight = cfg.MODEL.FASTINST.MASK_WEIGHT

        # building criterion
        
        matcher = VideoHungarianMatcher_Consistent(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.FASTINST.TRAIN_NUM_POINTS,
            frames = cfg.INPUT.SAMPLING_FRAME_NUM
        )
        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            #dec_layers = cfg.MODEL.FASTINST.DEC_LAYERS
            dec_layers = 6
            aux_weight_dict = {}
            for i in range(dec_layers-1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        # use CL
        weight_dict.update({'loss_reid': 2})

        losses = ["labels", "masks"]

        criterion_td = VideoSetCriterion( # the same as fastinst.py
            cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.FASTINST.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.FASTINST.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.FASTINST.IMPORTANCE_SAMPLE_RATIO,
        )

        max_iter_num = cfg.SOLVER.MAX_ITER

        rets.update(
            num_frames=num_frames,
            num_topk=num_topk,
            num_clip_frames=num_clip_frames,
            to_cpu_frames=to_cpu_frames,
            test_interpolate_chunk_size=test_interpolate_chunk_size,
            test_instance_chunk_size=test_instance_chunk_size,
            cl_plugin=cl_plugin,
            tracker=tracker,
            criterion_td=criterion_td, #
            max_iter_num = max_iter_num
        )

        return rets

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        if self.training:
            return self.train_model(batched_inputs)
        else:
            return self.inference_model(batched_inputs)
            

    def pre_process(self, batched_inputs):
        images = []  # noqa
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        return images

    def train_model(self, batched_inputs):
        images = self.pre_process(batched_inputs)  # noqa

        # mask classification target
        if "instances" in batched_inputs[0]:
            gt_instances = []
            for video in batched_inputs:
                for frame in video["instances"]:
                    gt_instances.append(frame.to(self.device))
            targets = self.prepare_targets(gt_instances, images)
        else:
            targets = None

        batched_inputs_length = images.tensor.shape[0]
        init_list = [None]*batched_inputs_length
        pred_logits, pred_masks, pred_embeds, pred_queries = init_list.copy(), init_list.copy(), init_list.copy(), init_list.copy()
        proposal_cls_logits,query_locations,pred_matching_indices = init_list.copy(), init_list.copy(), init_list.copy()
        pred_queries_without_norm = init_list.copy()
        mask_features = init_list.copy()
        aux_outputs = [] 
        video_length = self.num_frames # cfg.INPUT.SAMPLING_FRAME_NUM
        #losses = {}

        self.backbone.eval()
        self.sem_seg_head.eval()

        for i in range(video_length):
            with torch.no_grad():
                clip_images_tensor = images.tensor[i::video_length, ...] 
                clip_targets = targets[i::video_length]
                
                clip_features = self.backbone(clip_images_tensor)

                if i==0: 
                    clip_outputs = self.sem_seg_head(clip_features,clip_targets,isfirst=True) # first frame of the video
                else:
                    clip_outputs = self.sem_seg_head(clip_features,clip_targets,isfirst=False)            

            for j in range(int(batched_inputs_length/video_length)): # video batchsize
                pred_logits[j*video_length+i] =  clip_outputs['pred_logits'][j].unsqueeze(0) 
                pred_masks[j*video_length+i] =  clip_outputs['pred_masks'][j].unsqueeze(0) 
                pred_embeds[j*video_length+i] =  clip_outputs['pred_embeds'][j].unsqueeze(0) 
                pred_queries[j*video_length+i] =  clip_outputs['pred_queries'][j].unsqueeze(0) 
                proposal_cls_logits[j*video_length+i] =  clip_outputs['proposal_cls_logits'][j].unsqueeze(0) 

                pred_queries_without_norm[j*video_length+i] =  clip_outputs['pred_queries_without_norm'][j].unsqueeze(0)
                mask_features[j*video_length+i] =  clip_outputs['mask_features'][j].unsqueeze(0)
            
            aux_outputs.append(clip_outputs['aux_outputs'])

        aux=[]
        
        det_outputs = {
            'pred_logits': torch.cat(pred_logits, dim=0),
            'pred_masks': torch.cat(pred_masks, dim=0),
            'pred_embeds': torch.cat(pred_embeds, dim=0),
            'pred_queries': torch.cat(pred_queries, dim=0),
            'proposal_cls_logits': torch.cat(proposal_cls_logits, dim=0),
            #'query_locations': torch.cat(query_locations, dim=0),
            'query_locations':None,
            #'pred_matching_indices': torch.cat(pred_matching_indices, dim=0),
            'pred_matching_indices': None,
            'aux_outputs': aux,
            # for td_tracker
            'pred_queries_without_norm':torch.cat(pred_queries_without_norm,dim=0),
            'mask_features':torch.cat(mask_features,dim=0)
        }
        
        del pred_logits, pred_masks, pred_embeds, pred_queries ###
        del proposal_cls_logits,query_locations,pred_matching_indices,aux_outputs ###
        del aux
        del pred_queries_without_norm,mask_features # td_trakcer
        
        det_outputs['pred_masks'] = det_outputs['pred_masks'].reshape(-1,video_length,det_outputs['pred_masks'].shape[-3],det_outputs['pred_masks'].shape[-2],det_outputs['pred_masks'].shape[-1]) #(B, q, h, w)->(b,t,q,h,w)
        det_outputs['pred_masks'] = det_outputs['pred_masks'].permute(0,2,1,3,4) # (b,t,q,h,w)->(b,q,t,h,w)
        det_outputs['pred_logits'] = det_outputs['pred_logits'].reshape(-1,video_length,det_outputs['pred_logits'].shape[-2],det_outputs['pred_logits'].shape[-1]) #(B, q, c)->(b,t,q,c)

        det_outputs['pred_queries'] = torch.cat([det_outputs['pred_queries'],det_outputs['pred_embeds']],dim=2)
        det_outputs['pred_queries_without_norm'] = torch.cat([det_outputs['pred_queries_without_norm'],det_outputs['pred_embeds']],dim=2)

        # reshape-> M2F format only for td_tracker
        frame_queries = det_outputs['pred_queries'].clone().detach()  # (B, q, c)
        frame_queries = frame_queries.reshape(-1,video_length,frame_queries.shape[-2],frame_queries.shape[-1]) # (b,t,q,c)
        frame_queries = frame_queries.permute(0,3,1,2) # (b,c,t,q)
        frame_queries_no_norm = det_outputs['pred_queries_without_norm'].clone().detach()  # (B, q, c) # change cl_decoder
        frame_queries_no_norm = frame_queries_no_norm.reshape(-1,video_length,frame_queries_no_norm.shape[-2],frame_queries_no_norm.shape[-1]) # (b,t,q,c)
        frame_queries_no_norm = frame_queries_no_norm.permute(0,3,1,2) # (b,c,t,q)
        mask_features = det_outputs['mask_features'].clone().detach().unsqueeze(0) # change cl_decoder (B,c,h,w)
        mask_features = mask_features.reshape(-1,video_length,mask_features.shape[-3],mask_features.shape[-2],mask_features.shape[-1]) # (b,t,c,h,w)
        torch.cuda.empty_cache()
        keep = False  # in DVIS++ self.keep: for running demo on very long videos
        outputs, indices = self.td_tracker(frame_queries, mask_features, return_indices=True,
                                    resume=keep, frame_classes=None, ## class not used?
                                    frame_embeds_no_norm=frame_queries_no_norm)
        det_outputs = self.reset_image_output_order(det_outputs, indices)

        targets = self.prepare_targets_td(batched_inputs, images)
        # use the segmenter prediction results to guide the matching process during early training phase
        det_outputs, outputs, targets = self.frame_decoder_loss_reshape(
            outputs, targets, image_outputs=det_outputs
        )
        if self.iter < self.max_iter_num // 2:
            losses, reference_match_result = self.criterion(outputs, targets,
                                                            matcher_outputs=det_outputs,
                                                            ret_match_result=True)
        else:
            losses, reference_match_result = self.criterion(outputs, targets,
                                                            matcher_outputs=None,
                                                            ret_match_result=True)
        
        if self.use_cl:  # set to False
            losses_cl = self.get_cl_loss_ref(outputs, reference_match_result)
            losses.update(losses_cl)

        self.iter += 1

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                losses.pop(k)

        return losses

    def reset_image_output_order(self, output, indices):
        # from DVIS++
        """
        in order to maintain consistency between the initial query and the guided results (segmenter prediction)
        :param output: segmenter prediction results (image-level segmentation results)
        :param indices: matched indicates
        :return: reordered outputs
        """
        # pred_keys, (b, c, t, q)
        indices = torch.Tensor(indices).to(torch.int64)  # (t, q)
        frame_indices = torch.range(0, indices.shape[0] - 1).to(indices).unsqueeze(1).repeat(1, indices.shape[1])
        # pred_masks, shape is (b, q, t, h, w)
        output['pred_masks'][0] = output['pred_masks'][0][indices, frame_indices].transpose(0, 1)
        # pred logits, shape is (b, t, q, c)
        output['pred_logits'][0] = output['pred_logits'][0][frame_indices, indices]
        return output

    def frame_decoder_loss_reshape(self, outputs, targets, image_outputs=None):
        # from DVIS++
        outputs['pred_masks'] = einops.rearrange(outputs['pred_masks'], 'b q t h w -> (b t) q () h w')
        outputs['pred_logits'] = einops.rearrange(outputs['pred_logits'], 'b t q c -> (b t) q c')
        outputs['pred_references'] = einops.rearrange(outputs['pred_references'], 'b c t q -> (b t) q c')

        if image_outputs is not None:
            image_outputs['pred_masks'] = einops.rearrange(image_outputs['pred_masks'], 'b q t h w -> (b t) q () h w')
            image_outputs['pred_logits'] = einops.rearrange(image_outputs['pred_logits'], 'b t q c -> (b t) q c')
        if 'aux_outputs' in outputs:
            for i in range(len(outputs['aux_outputs'])):
                outputs['aux_outputs'][i]['pred_masks'] = einops.rearrange(
                    outputs['aux_outputs'][i]['pred_masks'], 'b q t h w -> (b t) q () h w'
                )
                outputs['aux_outputs'][i]['pred_logits'] = einops.rearrange(
                    outputs['aux_outputs'][i]['pred_logits'], 'b t q c -> (b t) q c'
                )
        gt_instances = []
        for targets_per_video in targets:
            num_labeled_frames = targets_per_video['ids'].shape[1]
            for f in range(num_labeled_frames):
                labels = targets_per_video['labels']
                ids = targets_per_video['ids'][:, [f]]
                masks = targets_per_video['masks'][:, [f], :, :]
                gt_instances.append({"labels": labels, "ids": ids, "masks": masks})
        return image_outputs, outputs, gt_instances

    def inference_model(self, batched_inputs):
        images = self.pre_process(batched_inputs)

        # Avoid Out-of-Memory
        num_frames = len(images)
        to_store = self.device if num_frames <= self.to_cpu_frames else "cpu"

        if num_frames <= self.num_clip_frames:  # noqa
            with torch.no_grad():
                features = self.backbone(images.tensor)
                det_outputs = self.sem_seg_head(features)
        else: # actually
            pred_logits, pred_masks, pred_embeds, pred_queries = [], [], [], []
            pred_queries_without_norm, mask_features = [], []
            self.num_clip_frames = 1 # 单帧推理
            num_clips = math.ceil(
                num_frames / self.num_clip_frames)  
            self.backbone.eval()
            self.sem_seg_head.eval()
            for i in range(num_clips):
                start_idx = i * self.num_clip_frames
                end_idx = (i + 1) * self.num_clip_frames
                clip_images_tensor = images.tensor[start_idx:end_idx, ...]
                with torch.no_grad():
                    clip_features = self.backbone(clip_images_tensor)
                    if i==0:
                        clip_outputs = self.sem_seg_head(clip_features,None,True)
                    else:
                        clip_outputs = self.sem_seg_head(clip_features,None,False)

                pred_logits.append(clip_outputs['pred_logits'])
                pred_masks.append(clip_outputs['pred_masks'])
                pred_embeds.append(clip_outputs['pred_embeds'])
                pred_queries.append(clip_outputs['pred_queries'])

                pred_queries_without_norm.append(clip_outputs['pred_queries_without_norm'])
                mask_features.append(clip_outputs['mask_features'])

            det_outputs = {
                'pred_logits': torch.cat(pred_logits, dim=0),
                'pred_masks': torch.cat(pred_masks, dim=0),
                'pred_embeds': torch.cat(pred_embeds, dim=0),
                'pred_queries': torch.cat(pred_queries, dim=0),
                'pred_queries_without_norm': torch.cat(pred_queries_without_norm, dim=0),
                'mask_features': torch.cat(mask_features, dim=0)
            }

        #  the same in training
        video_length = num_frames
        det_outputs['pred_masks'] = det_outputs['pred_masks'].reshape(-1,video_length,det_outputs['pred_masks'].shape[-3],det_outputs['pred_masks'].shape[-2],det_outputs['pred_masks'].shape[-1]) #(B, q, h, w)->(b,t,q,h,w)
        det_outputs['pred_masks'] = det_outputs['pred_masks'].permute(0,2,1,3,4) # (b,t,q,h,w)->(b,q,t,h,w)
        det_outputs['pred_logits'] = det_outputs['pred_logits'].reshape(-1,video_length,det_outputs['pred_logits'].shape[-2],det_outputs['pred_logits'].shape[-1]) #(B, q, c)->(b,t,q,c)

        det_outputs['pred_queries'] = torch.cat([det_outputs['pred_queries'],det_outputs['pred_embeds']],dim=2)
        det_outputs['pred_queries_without_norm'] = torch.cat([det_outputs['pred_queries_without_norm'],det_outputs['pred_embeds']],dim=2)

        # reshape-> M2F format only for td_tracker
        frame_queries = det_outputs['pred_queries'].clone().detach()  # (B, q, c)
        frame_queries = frame_queries.reshape(-1,video_length,frame_queries.shape[-2],frame_queries.shape[-1]) # (b,t,q,c)
        frame_queries = frame_queries.permute(0,3,1,2) # (b,c,t,q)
        frame_queries_no_norm = det_outputs['pred_queries_without_norm'].clone().detach()  # (B, q, c) # change cl_decoder
        frame_queries_no_norm = frame_queries_no_norm.reshape(-1,video_length,frame_queries_no_norm.shape[-2],frame_queries_no_norm.shape[-1]) # (b,t,q,c)
        frame_queries_no_norm = frame_queries_no_norm.permute(0,3,1,2) # (b,c,t,q)
        mask_features = det_outputs['mask_features'].clone().detach().unsqueeze(0) # change cl_decoder (B,c,h,w)
        mask_features = mask_features.reshape(-1,video_length,mask_features.shape[-3],mask_features.shape[-2],mask_features.shape[-1]) # (b,t,c,h,w)
        torch.cuda.empty_cache()
        keep = False  # in DVIS++ self.keep: for running demo on very long videos
        outputs, indices = self.td_tracker(frame_queries, mask_features, return_indices=True,
                                    resume=keep, frame_classes=None, ## class not used?
                                    frame_embeds_no_norm=frame_queries_no_norm)
        det_outputs = self.reset_image_output_order(det_outputs, indices)

        outputs = self.post_processing(outputs)
        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]
        pred_ids = outputs["ids"]

        mask_cls_result = mask_cls_results[0]
        mask_pred_result = mask_pred_results[0]
        pred_id = pred_ids[0]
        first_resize_size = (images.tensor.shape[-2], images.tensor.shape[-1])

        input_per_image = batched_inputs[0]
        image_size = images.image_sizes[0]  # image size without padding after data augmentation

        height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
        width = input_per_image.get("width", image_size[1])

        return retry_if_cuda_oom(self.inference_video_vis)(
            mask_cls_result, mask_pred_result, image_size, height, width, first_resize_size, pred_id
        )

    def inference_video_vis(
        self, pred_cls, pred_masks, img_size, output_height, output_width,
        first_resize_size, pred_id, aux_pred_cls=None,
    ):
        if len(pred_cls) > 0:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]
            if aux_pred_cls is not None:
                aux_pred_cls = F.softmax(aux_pred_cls, dim=-1)[:, :-1]
                scores = torch.maximum(scores, aux_pred_cls.to(scores))
            labels = torch.arange(
                self.sem_seg_head.num_classes, device=self.device
            ).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
            # keep top-K predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.max_num, sorted=False)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.sem_seg_head.num_classes
            pred_masks = pred_masks[topk_indices]
            #pred_ids = pred_id[topk_indices]
            pred_ids = pred_id[topk_indices.cpu()]

            # interpolation to original image size
            pred_masks = F.interpolate(
                pred_masks, size=first_resize_size, mode="bilinear", align_corners=False
            )
            pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]
            pred_masks = F.interpolate(
                pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
            )
            masks = pred_masks > 0.
            del pred_masks

            out_scores = scores_per_image.tolist()
            out_labels = labels_per_image.tolist()
            out_ids = pred_ids.tolist()
            out_masks = [m for m in masks.cpu()]
        else:
            out_scores = []
            out_labels = []
            out_masks = []
            out_ids = []

        video_output = {
            "image_size": (output_height, output_width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
            "pred_ids": out_ids,
            "task": "vis",
        }

        return video_output

    def inference_video(self, mask_cls_results, mask_pred_results, image_tensor_size, image_size, height, width,
                        to_store):
        mask_cls_result = mask_cls_results[0]
        mask_pred_result = mask_pred_results[0]

        if len(mask_cls_result) > 0:
            scores = F.softmax(mask_cls_result, dim=-1)[:, :-1]
            labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(
                len(mask_cls_result), 1).flatten(0, 1)  # noqa
            scores_per_video, topk_indices = scores.flatten(
                0, 1).topk(self.num_topk, sorted=True)

            labels_per_video = labels[topk_indices]
            topk_indices = topk_indices // self.sem_seg_head.num_classes

            mask_pred_result = mask_pred_result[topk_indices]

            num_instance, num_frame = mask_pred_result.shape[:2]

            out_scores = []
            out_labels = []
            out_masks = []

            for k in range(math.ceil(num_instance / self.test_instance_chunk_size)):
                _mask_pred_result = mask_pred_result[
                    k * self.test_instance_chunk_size:(k + 1) * self.test_instance_chunk_size, ...]
                _scores_per_video = scores_per_video[
                    k * self.test_instance_chunk_size:(k + 1) * self.test_instance_chunk_size, ...]
                _labels_per_video = labels_per_video[
                    k * self.test_instance_chunk_size:(k + 1) * self.test_instance_chunk_size, ...]

                masks_list = []  # noqa
                numerator = torch.zeros(
                    _mask_pred_result.shape[0], dtype=torch.float, device=self.device)
                denominator = torch.zeros(
                    _mask_pred_result.shape[0], dtype=torch.float, device=self.device)
                for i in range(math.ceil(num_frame / self.test_interpolate_chunk_size)):
                    temp_pred_mask = _mask_pred_result[:,
                                     i * self.test_interpolate_chunk_size:(i + 1) * self.test_interpolate_chunk_size,
                                     ...]  # noqa
                    temp_pred_mask = retry_if_cuda_oom(F.interpolate)(
                        temp_pred_mask,
                        size=(image_tensor_size[-2], image_tensor_size[-1]),
                        mode="bilinear",
                        align_corners=False)
                    temp_pred_mask = temp_pred_mask[:, :,
                                                    : image_size[0], : image_size[1]]

                    temp_pred_mask = retry_if_cuda_oom(F.interpolate)(temp_pred_mask, size=(height, width),
                                                                      mode="bilinear", align_corners=False)  # noqa
                    masks = (temp_pred_mask > 0.).float()
                    numerator += (temp_pred_mask.sigmoid()
                                  * masks).flatten(1).sum(1)
                    denominator += masks.flatten(1).sum(1)

                    masks_list.append(masks.bool().to(to_store))
                _scores_per_video *= (numerator / (denominator + 1e-6))
                masks = torch.cat(masks_list, dim=1)

                out_scores.extend(_scores_per_video.tolist())
                out_labels.extend(_labels_per_video.tolist())
                out_masks.extend([m for m in masks.cpu()])
        else:
            out_scores = []
            out_labels = []
            out_masks = []

        video_output = {
            "image_size": (height, width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }
        return video_output

    def post_processing(self, outputs, aux_logits=None):
        # from DVIS++
        """
        average the class logits and append query ids
        """
        pred_logits = outputs['pred_logits']
        pred_logits = pred_logits[0]  # (t, q, c)
        out_logits = torch.mean(pred_logits, dim=0).unsqueeze(0)
        if aux_logits is not None:
            aux_logits = aux_logits[0]
            aux_logits = torch.mean(aux_logits, dim=0)  # (q, c)
        outputs['pred_logits'] = out_logits
        outputs['ids'] = [torch.arange(0, outputs['pred_masks'].size(1))]
        if aux_logits is not None:
            return outputs, aux_logits
        return outputs

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            if isinstance(targets_per_image.gt_masks, BitMasks):
                gt_masks = targets_per_image.gt_masks.tensor
            else:
                gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1],
                         : gt_masks.shape[2]] = gt_masks

            # filter empty instances
            gt_instance_ids = targets_per_image.gt_ids
            valid_index = gt_instance_ids != -1

            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes[valid_index],
                    "masks": padded_masks[valid_index],
                    "ids": gt_instance_ids[valid_index],
                }
            )
        return new_targets

    def prepare_targets_td(self, targets, images):
        # from DVIS++
        h_pad, w_pad = images.tensor.shape[-2:]
        gt_instances = []
        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

            gt_ids_per_video = []
            gt_classes_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])
                gt_classes_per_video.append(targets_per_frame.gt_classes[:, None])
                if isinstance(targets_per_frame.gt_masks, BitMasks):
                    gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor
                else:  # polygon
                    gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks

            gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)
            gt_classes_per_video = torch.cat(gt_classes_per_video, dim=1).max(dim=1)[0]
            valid_idx = (gt_ids_per_video != -1).any(dim=-1)

            gt_classes_per_video = gt_classes_per_video[valid_idx]  # N,
            gt_ids_per_video = gt_ids_per_video[valid_idx]  # N, num_frames

            gt_instances.append({"labels": gt_classes_per_video, "ids": gt_ids_per_video})
            gt_masks_per_video = gt_masks_per_video[valid_idx].float()  # N, num_frames, H, W
            gt_instances[-1].update({"masks": gt_masks_per_video})

        return gt_instances

    def get_cl_loss_ref(self, outputs, referecne_match_result):
        references = outputs['pred_references']  # t q c

        # per frame
        contrastive_items = []
        for i in range(references.size(0)):
            if i == 0:
                continue
            frame_reference = references[i]  # (q, c)
            frame_reference_ = references[i - 1]  # (q, c)

            if i != references.size(0) - 1:
                frame_reference_next = references[i + 1]
            else:
                frame_reference_next = None

            frame_ref_gt_indices = referecne_match_result[i]

            gt2ref = {}
            for i_ref, i_gt in zip(frame_ref_gt_indices[0], frame_ref_gt_indices[1]):
                gt2ref[i_gt.item()] = i_ref.item()
            # per instance
            for i_gt in gt2ref.keys():
                i_ref = gt2ref[i_gt]

                anchor_embeds = frame_reference[[i_ref]]
                pos_embeds = frame_reference_[[i_ref]]
                neg_range = list(range(0, i_ref)) + list(range(i_ref + 1, frame_reference.size(0)))
                neg_embeds = frame_reference_[neg_range]

                num_positive = pos_embeds.shape[0]
                # concate pos and neg to get whole constractive samples
                pos_neg_embedding = torch.cat(
                    [pos_embeds, neg_embeds], dim=0)
                # generate label, pos is 1, neg is 0
                pos_neg_label = pos_neg_embedding.new_zeros((pos_neg_embedding.shape[0],),
                                                            dtype=torch.int64)  # noqa
                pos_neg_label[:num_positive] = 1.

                # dot product
                dot_product = torch.einsum(
                    'ac,kc->ak', [pos_neg_embedding, anchor_embeds])
                aux_normalize_pos_neg_embedding = nn.functional.normalize(
                    pos_neg_embedding, dim=1)
                aux_normalize_anchor_embedding = nn.functional.normalize(
                    anchor_embeds, dim=1)

                aux_cosine_similarity = torch.einsum('ac,kc->ak', [aux_normalize_pos_neg_embedding,
                                                                   aux_normalize_anchor_embedding])
                contrastive_items.append({
                    'dot_product': dot_product,
                    'cosine_similarity': aux_cosine_similarity,
                    'label': pos_neg_label})

                if frame_reference_next is not None:
                    pos_embeds = frame_reference_next[[i_ref]]
                    neg_range = list(range(0, i_ref)) + list(range(i_ref + 1, frame_reference.size(0)))
                    neg_embeds = frame_reference_next[neg_range]

                    num_positive = pos_embeds.shape[0]
                    # concate pos and neg to get whole constractive samples
                    pos_neg_embedding = torch.cat(
                        [pos_embeds, neg_embeds], dim=0)
                    # generate label, pos is 1, neg is 0
                    pos_neg_label = pos_neg_embedding.new_zeros((pos_neg_embedding.shape[0],),
                                                                dtype=torch.int64)  # noqa
                    pos_neg_label[:num_positive] = 1.

                    # dot product
                    dot_product = torch.einsum(
                        'ac,kc->ak', [pos_neg_embedding, anchor_embeds])
                    aux_normalize_pos_neg_embedding = nn.functional.normalize(
                        pos_neg_embedding, dim=1)
                    aux_normalize_anchor_embedding = nn.functional.normalize(
                        anchor_embeds, dim=1)

                    aux_cosine_similarity = torch.einsum('ac,kc->ak', [aux_normalize_pos_neg_embedding,
                                                                       aux_normalize_anchor_embedding])
                    contrastive_items.append({
                        'dot_product': dot_product,
                        'cosine_similarity': aux_cosine_similarity,
                        'label': pos_neg_label})

        losses = self.loss_reid(contrastive_items, outputs)
        return losses

    def loss_reid(self,qd_items, outputs):
        # outputs only using when have not contrastive items
        # compute two loss, contrastive loss & similarity loss
        contras_loss = 0
        aux_loss = 0
        num_qd_items = len(qd_items) # n_instances * frames

        # if none items, return 0 loss
        if len(qd_items) == 0:
            if 'pred_references' in outputs.keys():
                losses = {'loss_reid': outputs['pred_references'].sum() * 0,
                        'loss_aux_reid': outputs['pred_references'].sum() * 0}
            else:
                losses = {'loss_reid': outputs['pred_embds'].sum() * 0,
                        'loss_aux_reid': outputs['pred_embds'].sum() * 0}
            return losses

        for qd_item in qd_items:
            # (n_pos, n_anchor) -> (n_anchor, n_pos)
            pred = qd_item['dot_product'].permute(1, 0)
            label = qd_item['label'].unsqueeze(0)
            # contrastive loss
            pos_inds = (label == 1)
            neg_inds = (label == 0)
            pred_pos = pred * pos_inds.float()
            pred_neg = pred * neg_inds.float()
            # use -inf to mask out unwanted elements.
            pred_pos[neg_inds] = pred_pos[neg_inds] + float('inf')
            pred_neg[pos_inds] = pred_neg[pos_inds] + float('-inf')

            _pos_expand = torch.repeat_interleave(pred_pos, pred.shape[1], dim=1)
            _neg_expand = pred_neg.repeat(1, pred.shape[1])
            # [bz,N], N is all pos and negative samples on reference frame, label indicate it's pos or negative
            x = torch.nn.functional.pad(
                (_neg_expand - _pos_expand), (0, 1), "constant", 0)
            contras_loss += torch.logsumexp(x, dim=1)

            aux_pred = qd_item['cosine_similarity'].permute(1, 0)
            aux_label = qd_item['label'].unsqueeze(0)
            aux_loss += (torch.abs(aux_pred - aux_label) ** 2).mean()

        losses = {'loss_reid': contras_loss.sum() / num_qd_items,
                'loss_aux_reid': aux_loss / num_qd_items}
        return losses