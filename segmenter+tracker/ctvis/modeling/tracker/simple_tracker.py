import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from detectron2.config import configurable
from detectron2.utils.registry import Registry

from ctvis.utils import mask_nms
from .memory_bank import MemoryBank

TRACKER_REGISTRY = Registry("Tracker")
TRACKER_REGISTRY.__doc__ = """Registry for Tracker for Online Video Instance Segmentation Models."""


def build_tracker(cfg):
    """
    Build a tracker for online video instance segmentation models
    """
    name = cfg.MODEL.TRACKER.TRACKER_NAME
    return TRACKER_REGISTRY.get(name)(cfg)


@TRACKER_REGISTRY.register()
class SimpleTracker(nn.Module):
    """
    Simple Tracker for Online Video Instance Segmentation.
    Follow IDOL.
    """

    @configurable
    def __init__(self,
                 *,
                 num_classes,
                 match_metric,
                 frame_weight,
                 match_score_thr,
                 temporal_score_type,
                 match_type,
                 inference_select_thr,
                 init_score_thr,
                 mask_nms_thr,
                 num_dead_frames,
                 embed_type,
                 maximum_cache,
                 noise_frame_num,
                 noise_frame_ratio,
                 suppress_frame_num,
                 none_frame_num):
        super().__init__()
        self.num_classes = num_classes
        self.match_metric = match_metric
        self.match_score_thr = match_score_thr
        self.temporal_score_type = temporal_score_type
        self.temporal_score_type = temporal_score_type
        assert self.temporal_score_type in ['mean', 'max', 'hybrid']
        self.match_type = match_type  # greedy hungarian

        self.inference_select_thr = inference_select_thr
        self.init_score_thr = init_score_thr
        self.mask_nms_thr = mask_nms_thr
        self.frame_weight = frame_weight

        self.num_dead_frames = num_dead_frames
        self.embed_type = embed_type
        self.maximum_cache = maximum_cache

        self.noise_frame_num = noise_frame_num
        self.noise_frame_ratio = noise_frame_ratio
        self.suppress_frame_num = suppress_frame_num
        self.none_frame_num = none_frame_num

    @classmethod
    def from_config(cls, cfg):
        num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES  # noqa
        match_metric = cfg.MODEL.TRACKER.MATCH_METRIC
        frame_weight = cfg.MODEL.TRACKER.FRAME_WEIGHT
        match_score_thr = cfg.MODEL.TRACKER.MATCH_SCORE_THR
        temporal_score_type = cfg.MODEL.TRACKER.TEMPORAL_SCORE_TYPE
        match_type = cfg.MODEL.TRACKER.MATCH_TYPE
        inference_select_thr = cfg.MODEL.TRACKER.INFERENCE_SELECT_THR
        init_score_thr = cfg.MODEL.TRACKER.INIT_SCORE_THR
        mask_nms_thr = cfg.MODEL.TRACKER.MASK_NMS_THR

        num_dead_frames = cfg.MODEL.TRACKER.MEMORY_BANK.NUM_DEAD_FRAMES
        embed_type = cfg.MODEL.TRACKER.MEMORY_BANK.EMBED_TYPE
        maximum_cache = cfg.MODEL.TRACKER.MEMORY_BANK.maximum_cache

        noise_frame_num = cfg.MODEL.TRACKER.NOISE_FRAME_NUM
        noise_frame_ratio = cfg.MODEL.TRACKER.NOISE_FRAME_RATIO
        suppress_frame_num = cfg.MODEL.TRACKER.SUPPRESS_FRAME_NUM
        none_frame_num = cfg.MODEL.TRACKER.NONE_FRAME_NUM

        ret = {
            "num_classes": num_classes,
            "match_metric": match_metric,
            "frame_weight": frame_weight,
            "match_score_thr": match_score_thr,
            "temporal_score_type": temporal_score_type,
            "match_type": match_type,
            "inference_select_thr": inference_select_thr,
            "init_score_thr": init_score_thr,
            "mask_nms_thr": mask_nms_thr,
            # memory bank & tracklet
            "num_dead_frames": num_dead_frames,
            "embed_type": embed_type,
            "maximum_cache": maximum_cache,
            "noise_frame_num": noise_frame_num,
            "noise_frame_ratio": noise_frame_ratio,
            "suppress_frame_num": suppress_frame_num,
            "none_frame_num": none_frame_num
        }
        return ret

    @property
    def device(self):
        return torch.device('cuda')

    @property
    def empty(self):
        return self.num_tracklets == 0

    def reset(self):
        self.num_tracklets = 0  # noqa
        self.memory_bank = MemoryBank(self.embed_type,  # noqa
                                      self.num_dead_frames,
                                      self.maximum_cache)

    def update(self, ids, pred_scores, pred_logits, pred_masks, pred_reid_embeds, frame_id):
        assert ids.shape[0] == pred_logits.shape[0], 'Shape must match.'  # noqa

        num_instances = ids.shape[0] # 更新的个数

        for instance_index in range(num_instances):

            instance_id = int(ids[instance_index].item())
            instance_score = pred_scores[instance_index]
            instance_logit = pred_logits[instance_index]
            instance_mask = pred_masks[instance_index]
            instance_reid_embed = pred_reid_embeds[instance_index] # pred_reid_embeds == pred_embed

            if instance_id in self.memory_bank.exist_ids: # for reid
                self.memory_bank.update(instance_id, instance_score, instance_logit,
                                        instance_mask, instance_reid_embed, frame_id)
            else:
                self.memory_bank.add(instance_id)
                self.memory_bank.update(instance_id, instance_score, instance_logit,
                                        instance_mask, instance_reid_embed, frame_id)

    def inference(self, det_outputs, hybrid_embed):
        num_frames, num_queries = det_outputs['pred_logits'].shape[:2] # shape[img_batch,num_queries,class_num]

        video_dict = dict()
        for frame_id in range(num_frames):
            if frame_id == 0:
                self.reset()

            # 抽取当前帧序号为frame_id的信息
            pred_logits = det_outputs['pred_logits'][frame_id] # shape[num_queries,class_num] [256,41]
            pred_masks = det_outputs['pred_masks'][frame_id] # shape[num_queries,h,w]
            pred_embeds = det_outputs['pred_embeds'][frame_id] # shape[num_queries,query_vector_len]
            pred_queries = det_outputs['pred_queries'][frame_id] # shape[num_queries,query_vector_len]

            scores = F.softmax(pred_logits, dim=-1)[:, :-1] # 在class上softmax，并丢弃最后一类（可能为空类）
            max_scores, max_indices = torch.max(scores, dim=1) # 将dim=1合并，返回max类的置信度和类标号，shape[256],[256] 
            _, sorted_indices = torch.sort(max_scores, dim=0, descending=True) # 将query根据分类置信度降序排序

            pred_scores = max_scores[sorted_indices] # 
            pred_logits = pred_logits[sorted_indices] # 重新排序，置信度降序
            pred_masks = pred_masks[sorted_indices]
            pred_embeds = pred_embeds[sorted_indices]
            pred_queries = pred_queries[sorted_indices]

            valid_indices = pred_scores > self.inference_select_thr # query的类别置信度大于阈值
            if valid_indices.sum() == 0: # 所有query的类别置信度都小于阈值，则取第一个query（当前帧仅有一个ins）
                valid_indices[0] = 1
            pred_scores = pred_scores[valid_indices] # tensor长度变化
            pred_logits = pred_logits[valid_indices]
            pred_masks = pred_masks[valid_indices]
            pred_embeds = pred_embeds[valid_indices]
            pred_queries = pred_queries[valid_indices]

            # NMS: can bring a slight improvement
            valid_nms_indices = mask_nms(pred_masks[:, None, ...], pred_scores, nms_thr=self.mask_nms_thr) # None为增加一个维度[query_num,1,h,w]
            # mask_nms计算masks两两之间的iou，大于阈值只保留第一个（置信度最大），返回值 布尔列表
            pred_scores = pred_scores[valid_nms_indices]
            pred_logits = pred_logits[valid_nms_indices]
            pred_masks = pred_masks[valid_nms_indices]
            pred_embeds = pred_embeds[valid_nms_indices]
            pred_queries = pred_queries[valid_nms_indices]

            ids, pred_logits, pred_masks, pred_queries = \
                self.track(pred_scores, pred_logits, pred_masks, pred_embeds, pred_queries, frame_id)

            for index in range(ids.shape[0]): # 对于self.track返回的获得匹配的query依次处理
                instance_id = ids[index]
                pred_logit = pred_logits[index]
                pred_mask = pred_masks[index]
                pred_query = pred_queries[index]

                if instance_id.item() in video_dict.keys(): # instance_id.item() -> tensor.item() tracklet序号
                    video_dict[instance_id.item()]['masks'].append(pred_mask)
                    video_dict[instance_id.item()]['scores'].append(pred_logit)
                    video_dict[instance_id.item()]['queries'].append(pred_query)
                else: # 构建ins-frameID表，每个ins拥有masks、scores、queries三个属性
                    video_dict[instance_id.item()] = {
                        'masks': [None for _ in range(frame_id)],
                        'scores': [None for _ in range(frame_id)],
                        'queries': [None for _ in range(frame_id)]}
                    video_dict[instance_id.item()]['masks'].append(pred_mask)
                    video_dict[instance_id.item()]['scores'].append(pred_logit)
                    video_dict[instance_id.item()]['queries'].append(pred_query)

            for k, v in video_dict.items():
                if len(v['masks']) < frame_id + 1: # list长度对齐
                    v['masks'].append(None)
                    v['scores'].append(None)
                    v['queries'].append(None)

            # filter sequences that are too short in video_dict (noise)，
            # the rule is: if the first two frames are None and valid is less than 3
            # stolen from IDOL
            # noise_frame_num = math.floor(num_frames * self.noise_frame_ratio)
            if frame_id > self.noise_frame_num:
                del_list = []
                for k, v in video_dict.items():
                    valid = sum([1 if _ is not None else 0 for _ in v['masks']])
                    none_frame = 0
                    for m in v['masks'][::-1]:
                        if m is None:
                            none_frame += 1
                        else:
                            break
                    if none_frame >= self.none_frame_num and valid < self.suppress_frame_num:
                        del_list.append(k)
                for del_k in del_list:
                    video_dict.pop(del_k)
                    # self.memory_bank.delete_tracklet(del_k)  uncomment will drop 0.24 AP

        logits_list = []
        masks_list = []
        mask_h, mask_w = det_outputs['pred_masks'].shape[-2:]
        for inst_id, m in enumerate(video_dict.keys()): # 对ins进行遍历
            score_list_ori = video_dict[m]['scores']
            query_list_ori = video_dict[m]['queries']
            scores_temporal = []
            queries_temporal = []
            for t, k in zip(query_list_ori, score_list_ori):
                if k is not None:
                    scores_temporal.append(k)
                    queries_temporal.append(t)
            logits_i = torch.stack(scores_temporal)
            if self.temporal_score_type == 'mean':
                logits_i = logits_i.mean(0)
            elif self.temporal_score_type == 'max':
                logits_i = logits_i.max(0)[0]
            elif self.temporal_score_type == 'hybrid':
                raise NotImplementedError
            logits_list.append(logits_i)

            masks_list_i = []
            for n in range(num_frames):
                mask_i = video_dict[m]['masks'][n]
                if mask_i is None:
                    zero_mask = det_outputs['pred_masks'].new_zeros(mask_h, mask_w)
                    masks_list_i.append(zero_mask)
                else:
                    masks_list_i.append(mask_i)
            masks_list_i = torch.stack(masks_list_i, dim=0) # 单个ins在所有出现的帧上的mask的集合（list）
            masks_list.append(masks_list_i)
        if len(logits_list) > 0:
            pred_cls = torch.stack(logits_list, dim=0)[None, ...]
            pred_masks = torch.stack(masks_list, dim=0)[None, ...] # None为增加一个维度
        else:
            pred_cls = []

        return {
            'pred_logits': pred_cls,
            'pred_masks': pred_masks
        }

    def track(self, pred_scores, pred_logits, pred_masks, pred_embeds, pred_queries, frame_id):
        ids = pred_logits.new_full((pred_logits.shape[0],), -1, dtype=torch.long) # ids=当前query的个数（经过类别置信度筛选和nms）

        if self.empty: # self.num_tracklets == 0, 视频第一个帧，直接添加，不需要匹配 / 无需reid
            valid_init_indices = pred_scores > self.init_score_thr # 筛选
            num_new_tracklets = valid_init_indices.sum()
            ids[valid_init_indices] = torch.arange(self.num_tracklets, self.num_tracklets + num_new_tracklets,
                                                   dtype=torch.long).to(self.device) # 通过筛选的instance在ids获得tracklets序号，其余instance保持-1
            self.num_tracklets += num_new_tracklets # 累计tracklets

        else: # tracklet非空，考虑reid
            num_instances = pred_logits.shape[0] # 当前帧的query
            # 返回memorybank中的tracklet对象的instance_id,instance_reid_embed,exist_frame_list
            exist_tracklet_ids, exist_reid_embeds, exist_frames = self.memory_bank.exist_reid_embeds(frame_id)

            if self.match_metric == 'bisoftmax':
                # d2t: 45.6
                # t2d: 45.7
                # bio: (t2d + d2t) / 2 : 48.3  good
                similarity = torch.mm(pred_embeds, exist_reid_embeds.t()) # 余弦相似度table shape[pred_embeds,exist_reid_embeds]
                d2t_scores = similarity.softmax(dim=1)
                t2d_scores = similarity.softmax(dim=0)
                match_scores = (d2t_scores + t2d_scores) / 2
            elif self.match_metric == 'cosine':
                key = F.normalize(pred_embeds, p=2, dim=1)
                query = F.normalize(exist_reid_embeds, p=2, dim=1)
                match_scores = torch.mm(key, query.t())
            else:
                raise NotImplementedError

            if self.match_type == 'greedy':
                for idx in range(num_instances):
                    if self.frame_weight: # 出现次数多的tracklet获得匹配的概率提高
                        valid_indices = match_scores[idx, :] > self.match_score_thr #一个q与所有reid_embeds的相似度 与 阈值 相比
                        if (match_scores[idx, valid_indices] > self.match_score_thr).sum() > 1: #有q的相似度大于阈值
                            wighted_scores = match_scores.clone()
                            frame_weight = exist_frames[valid_indices].to(wighted_scores)
                            wighted_scores[idx, valid_indices] = wighted_scores[idx, valid_indices] * frame_weight
                            wighted_scores[idx, ~valid_indices] = wighted_scores[
                                                                      idx, ~valid_indices] * frame_weight.mean()
                            match_score, max_indices = torch.max(wighted_scores[idx, :], dim=0)
                        else:
                            match_score, max_indices = torch.max(match_scores[idx, :], dim=0) # 直接根据match_scores得分匹配
                    else:
                        match_score, max_indices = torch.max(match_scores[idx, :], dim=0)

                    match_tracklet_id = exist_tracklet_ids[max_indices]
                    assert match_tracklet_id > -1
                    if match_score > self.match_score_thr:
                        ids[idx] = match_tracklet_id # 记录匹配
                        match_scores[:idx, max_indices] = 0 # max_indices列除了idx行，其余全部置0（不可匹配）
                        match_scores[idx + 1:, max_indices] = 0
            elif self.match_type == 'hungarian':
                # drop 3 AP
                match_cost = - match_scores.cpu()
                indices = linear_sum_assignment(match_cost)

                for i, (instance_id, _id) in enumerate(zip(*indices)):
                    if match_scores[instance_id, _id] < self.match_score_thr:
                        indices[1][i] = -1

                ids[indices[0]] = ids.new_tensor(exist_tracklet_ids[indices[1]])
                ids[indices[0][indices[1] == -1]] = -1
            else:
                raise NotImplementedError

            new_instance_indices = (ids == -1) & (pred_scores > self.init_score_thr)
            num_new_tracklets = new_instance_indices.sum().item()
            ids[new_instance_indices] = torch.arange(self.num_tracklets,
                                                     self.num_tracklets + num_new_tracklets,
                                                     dtype=torch.long).to(self.device)
            self.num_tracklets += num_new_tracklets

        valid_inds = ids > -1 
        ids = ids[valid_inds] # 筛选获得tracklet序号的query
        pred_scores = pred_scores[valid_inds]
        pred_logits = pred_logits[valid_inds]
        pred_masks = pred_masks[valid_inds]
        pred_embeds = pred_embeds[valid_inds]
        pred_queries = pred_queries[valid_inds]

        self.update(
            ids=ids,
            pred_scores=pred_scores,
            pred_logits=pred_logits,
            pred_masks=pred_masks,
            pred_reid_embeds=pred_embeds,
            frame_id=frame_id)
        self.memory_bank.clean_dead_tracklets(frame_id)

        return ids, pred_logits, pred_masks, pred_queries
