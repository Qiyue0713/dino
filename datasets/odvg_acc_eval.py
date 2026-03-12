"""
import torch
import util.misc as utils


def box_iou_xyxy(boxes1, boxes2):

    # boxes1: (N,4) xyxy
    # boxes2: (M,4) xyxy
    # return: (N,M) IoU

    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N,M,2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N,M,2)
    wh = (rb - lt).clamp(min=0)                         # (N,M,2)
    inter = wh[..., 0] * wh[..., 1]                     # (N,M)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


class ODVGAccEvaluator:

    # Phrase/label-aligned Acc@K (instance-level):
    #   For each GT instance (gt_box, gt_label):
    #     select predictions with pred_label == gt_label
    #     sort by score desc
    #     hit@K = 1 if any of top-K boxes has IoU >= thr with gt_box
    #   Acc@K = hits / #GT
    #
    # 同时返回 Top1/Top5（等价于 Acc@1 / Acc@5 的写法）。

    def __init__(self, iou_thr=0.5, ks=(1, 5)):
        self.iou_thr = float(iou_thr)
        self.ks = tuple(int(k) for k in ks)
        self.reset()

    def reset(self):
        self.total_gt = 0
        self.hits = {k: 0 for k in self.ks}

    @torch.no_grad()
    def update(self, results, targets):

        # results: list[dict] length=B
        #   each dict has:
        #     'boxes': Tensor(P,4) xyxy abs (on original image size, from postprocessor)
        #     'scores': Tensor(P,)
        #     'labels': Tensor(P,)  (for VG, should align with cap_list indices)
        # targets: list[dict] length=B
        #   must contain:
        #     'boxes_orig': Tensor(G,4) xyxy abs (original image)
        #     'labels_orig': Tensor(G,)

        for res, tgt in zip(results, targets):
            gt_boxes = tgt.get("boxes_orig", None)
            gt_labels = tgt.get("labels_orig", None)
            if gt_boxes is None or gt_labels is None:
                raise KeyError("targets must contain 'boxes_orig' and 'labels_orig' for ODVGAccEvaluator")

            if gt_boxes.numel() == 0:
                continue

            pred_boxes = res["boxes"]
            pred_scores = res["scores"]
            pred_labels = res.get("labels", None)

            # 没预测也要计入 GT 总数
            self.total_gt += int(gt_boxes.shape[0])

            if pred_boxes.numel() == 0 or pred_labels is None:
                continue

            # 先按总 score 排序，后面按 label 过滤时仍保持相对顺序
            order = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[order]
            pred_scores = pred_scores[order]
            pred_labels = pred_labels[order]

            # 对每个 GT 实例统计 hit@K
            for j in range(gt_boxes.shape[0]):
                glb = gt_labels[j]
                gbox = gt_boxes[j].unsqueeze(0)  # (1,4)

                mask = (pred_labels == glb)
                if not mask.any():
                    continue

                pb = pred_boxes[mask]  # (Pj,4)
                # pb 已按 score 降序（继承 order）
                ious = box_iou_xyxy(pb, gbox).squeeze(1)  # (Pj,)

                for k in self.ks:
                    kk = min(k, ious.shape[0])
                    if kk <= 0:
                        continue
                    if (ious[:kk].max() >= self.iou_thr).item():
                        self.hits[k] += 1

    def synchronize_between_processes(self):
        if not utils.is_dist_avail_and_initialized():
            return

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        total = torch.tensor([self.total_gt], dtype=torch.float64, device=device)
        hits = torch.tensor([self.hits[k] for k in self.ks], dtype=torch.float64, device=device)

        torch.distributed.all_reduce(total)
        torch.distributed.all_reduce(hits)

        self.total_gt = int(total.item())
        for i, k in enumerate(self.ks):
            self.hits[k] = int(hits[i].item())

    def summarize(self):
        denom = max(self.total_gt, 1)
        out = {}
        for k in self.ks:
            out[f"Acc@{k}"] = self.hits[k] / denom
            # 若你也想直接叫 TopK：
            out[f"Top{k}"] = out[f"Acc@{k}"]
        return out
"""

import torch
import util.misc as utils

# 转换坐标
def convert_xyxy(box):
    new_box = torch.zeros_like(box)
    new_box[:, :, 0] = box[:, :, 0] - box[:, :, 2] / 2
    new_box[:, :, 1] = box[:, :, 1] - box[:, :, 3] / 2
    new_box[:, :, 2] = box[:, :, 0] + box[:, :, 2] / 2
    new_box[:, :, 3] = box[:, :, 1] + box[:, :, 3] / 2
    return new_box

# 计算交并比
def compute_iou(gt_box, candidate_boxes):
    """
    计算 IoU。

    :param gt_box: Ground truth boxes, shape [bs, 4] (相对坐标)
    :param candidate_boxes: 候选 boxes, shape [bs, nq, 4] (相对坐标)
    :return: (first_iou, max_iou) - 第一候选框的 IoU 和最大 IoU
    """
    # 将 gt_box 和 candidate_boxes 进行广播，计算交集和并集
    gt_box = gt_box.unsqueeze(1)  # 形状变为 [bs, 1, 4]
    gt_box = convert_xyxy(gt_box)
    candidate_boxes = convert_xyxy(candidate_boxes)
    # 计算交集
    inter_x1 = torch.max(gt_box[:, :, 0], candidate_boxes[:, :, 0])  # x_min
    inter_y1 = torch.max(gt_box[:, :, 1], candidate_boxes[:, :, 1])  # y_min
    inter_x2 = torch.min(gt_box[:, :, 2], candidate_boxes[:, :, 2])  # x_max
    inter_y2 = torch.min(gt_box[:, :, 3], candidate_boxes[:, :, 3])  # y_max

    # 计算交集的宽和高
    inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_height = torch.clamp(inter_y2 - inter_y1, min=0)

    # 计算交集面积
    inter_area = inter_width * inter_height

    # 计算 gt_box 和 candidate_boxes 的面积
    gt_area = (gt_box[:, :, 2] - gt_box[:, :, 0]) * (gt_box[:, :, 3] - gt_box[:, :, 1])
    candidate_area = (candidate_boxes[:, :, 2] - candidate_boxes[:, :, 0]) * (
                candidate_boxes[:, :, 3] - candidate_boxes[:, :, 1])

    # 计算并集面积
    union_area = gt_area + candidate_area - inter_area

    # 计算 IoU
    iou = inter_area / (union_area + 1e-6)  # 防止除以零

    # 获取第一个候选框的 IoU 和最大 IoU
    first_iou = iou[:, 0]  # 第一个候选框的 IoU
    max_iou = iou.max(dim=1).values  # 每个样本的最大 IoU

    return first_iou, max_iou


class ODVGAccEvaluator:
    """
    与 eval_vg 完全同逻辑：
      - caption 在外面传入 model（这里不管）
      - logits = sigmoid(pred_logits).max(dim=2)
      - topk 选前K并按 score 降序排序
      - gt_box = targets[b]["boxes"][0]
      - first_iou / max_iou 用 compute_iou 计算
      - 输出 Top_1 / Top_k (阈值命中比例)
    """

    def __init__(self, iou_thr=0.5, topk=5):
        self.iou_thr = float(iou_thr)
        self.topk = int(topk)
        if self.topk <= 0:
            raise ValueError("topk must be positive")
        self.reset()

    def reset(self):
        self.total = 0
        self.top1_hit = 0
        self.topk_hit = 0

    @torch.no_grad()
    def update(self, outputs, targets):
        if "pred_logits" not in outputs or "pred_boxes" not in outputs:
            raise KeyError("outputs must contain 'pred_logits' and 'pred_boxes'")

        pred_logits = outputs["pred_logits"]  # (B,nq,T)
        pred_boxes = outputs["pred_boxes"]    # (B,nq,4) 这里假设是 cxcywh 相对坐标

        B, nq = pred_boxes.shape[:2]
        if B != len(targets):
            raise ValueError(f"Batch mismatch: outputs B={B}, targets len={len(targets)}")

        # (B,nq)
        logits = pred_logits.sigmoid().max(dim=2)[0]

        k = min(self.topk, nq)
        if k <= 0:
            return

        # topk
        top_values, top_indices = logits.topk(k, dim=1)  # (B,k)

        # 和 eval_vg 一致：再按 top_values 降序排一次
        sorted_indices = top_values.argsort(dim=1, descending=True)
        top_indices = top_indices.gather(1, sorted_indices)  # (B,k)

        # 取出 topk boxes: (B,k,4)
        batch_ids = torch.arange(B, device=pred_boxes.device).unsqueeze(1)
        boxes_selected_sorted = pred_boxes[batch_ids, top_indices]

        # 取 GT: (B,4) 只用每个样本第一个 GT
        gt_list = []
        valid = []
        for t in targets:
            tb = t.get("boxes", None)
            if tb is None:
                raise KeyError("targets must contain key 'boxes'")
            if tb.numel() == 0:
                valid.append(False)
                gt_list.append(pred_boxes.new_zeros((4,)))
            else:
                valid.append(True)
                gt_list.append(tb[0])

        valid = torch.tensor(valid, device=pred_boxes.device, dtype=torch.bool)
        if not valid.any():
            return

        gt_box = torch.stack(gt_list, dim=0)  # (B,4)

        # 用你现成的 compute_iou（内部会 convert_xyxy）
        first_iou, max_iou = compute_iou(gt_box, boxes_selected_sorted)

        # 只统计有 GT 的样本
        first_iou = first_iou[valid]
        max_iou = max_iou[valid]

        self.total += int(valid.sum().item())
        self.top1_hit += int((first_iou > self.iou_thr).sum().item())
        self.topk_hit += int((max_iou > self.iou_thr).sum().item())

    def synchronize_between_processes(self):
        if not utils.is_dist_avail_and_initialized():
            return

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        t = torch.tensor([self.total, self.top1_hit, self.topk_hit], dtype=torch.float64, device=device)
        torch.distributed.all_reduce(t)

        self.total = int(t[0].item())
        self.top1_hit = int(t[1].item())
        self.topk_hit = int(t[2].item())

    def summarize(self):
        denom = max(self.total, 1)
        return {
            "Top_1": self.top1_hit / denom,
            "Top_k": self.topk_hit / denom,
        }

