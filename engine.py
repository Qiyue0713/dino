# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable

from util.utils import to_device
import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.cocogrounding_eval import CocoGroundingEvaluator
from datasets.odvg_acc_eval import ODVGAccEvaluator


from datasets.panoptic_eval import PanopticEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)


    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0


    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):

        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        cap_list = [t["cap_list"] for t in targets]
        targets = [{k: v.to(device) for k, v in t.items() if torch.is_tensor(v)} for t in targets]
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(samples, captions=captions)
            loss_dict = criterion(outputs, targets, cap_list, captions)

            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()


        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat

"""
@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir,
             wo_class_error=False, args=None, logger=None):

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # TopK/Acc@K evaluator
    odvg_acc_evaluator = ODVGAccEvaluator(
        iou_thr=getattr(args, "acc_iou_thr", 0.5),
        ks=getattr(args, "acc_ks", (1, 5)),
    )

    output_state_dict = {}
    _cnt = 0

    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        # >>> 关键改动：直接对每个 target dict 调用 to_device（递归处理）
        targets = [to_device(t, device) for t in targets]

        # ODVG/VG：每张图自己的 caption
        input_captions = [t["caption"] for t in targets]

        with torch.cuda.amp.autocast(enabled=getattr(args, "amp", False)):
            outputs = model(samples, captions=input_captions)

        if "bbox" not in postprocessors:
            raise KeyError("postprocessors missing key 'bbox' (required for bbox grounding eval).")

        # 用 orig_size 把预测框还原到原图尺度
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors["bbox"](outputs, orig_target_sizes)
        # results: list[dict] each has 'boxes','scores','labels', ...

        # 更新 TopK/Acc@K（内部应使用 boxes_orig / labels_orig）
        odvg_acc_evaluator.update(results, targets)

        # 可选：保存 debug 结果
        if getattr(args, "save_results", False):
            for tgt, res_i in zip(targets, results):
                gt_bbox = tgt["boxes_orig"]
                gt_label = tgt["labels_orig"]
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), dim=1)

                _res_bbox = res_i["boxes"]
                _res_prob = res_i["scores"]
                _res_label = res_i["labels"]
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), dim=1)

                output_state_dict.setdefault("gt_info", []).append(gt_info.detach().cpu())
                output_state_dict.setdefault("res_info", []).append(res_info.detach().cpu())

        _cnt += 1
        if getattr(args, "debug", False) and (_cnt % 15 == 0):
            print("BREAK!" * 5)
            break

    # 分布式同步
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    odvg_acc_evaluator.synchronize_between_processes()

    # 汇总指标
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    stats.update(odvg_acc_evaluator.summarize())

    # 保存 debug 文件（每个 rank 一个）
    if getattr(args, "save_results", False):
        import os.path as osp
        savepath = osp.join(args.output_dir, f"results-{utils.get_rank()}.pkl")
        print(f"Saving res to {savepath}")
        torch.save(output_state_dict, savepath)

    return stats, None
"""

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir,
             wo_class_error=False, args=None, logger=None):

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # 与 eval_vg 对齐：只关心一个 topk
    odvg_acc_evaluator = ODVGAccEvaluator(
        iou_thr=getattr(args, "acc_iou_thr", 0.5),
        topk=getattr(args, "top_k", 5),   # 对齐你参考脚本里的 --top_k
    )

    output_state_dict = {}
    _cnt = 0

    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)
        targets = [to_device(t, device) for t in targets]

        # 关键：与 eval_vg 一致，用 anno 作为 captions
        input_captions = [t["anno"] for t in targets]

        with torch.cuda.amp.autocast(enabled=getattr(args, "amp", False)):
            outputs = model(samples, captions=input_captions)

        # 与 eval_vg 一致：直接用 outputs/targets 算 Top1/TopK
        odvg_acc_evaluator.update(outputs, targets)

        # Optional: 保存 debug 信息（也按 eval_vg 的 topk 逻辑）
        if getattr(args, "save_results", False):
            topk = getattr(args, "top_k", 5)
            pred_logits = outputs["pred_logits"]
            pred_boxes = outputs["pred_boxes"]
            pred_scores = pred_logits.sigmoid().max(dim=2)[0]  # (B, nq)

            B, nq = pred_scores.shape
            k_sel = min(topk, nq)

            if k_sel > 0:
                top_vals, top_idx = pred_scores.topk(k_sel, dim=1)
                sort_idx = top_vals.argsort(dim=1, descending=True)
                top_idx = top_idx.gather(1, sort_idx)

                batch_ids = torch.arange(B, device=pred_boxes.device).unsqueeze(1)
                sel_boxes = pred_boxes[batch_ids, top_idx]    # (B,k,4) cxcywh
                sel_scores = pred_scores[batch_ids, top_idx]  # (B,k)
            else:
                sel_boxes = pred_boxes.new_zeros((B, 0, 4))
                sel_scores = pred_scores.new_zeros((B, 0))

            for bi, tgt in enumerate(targets):
                tgt_boxes = tgt.get("boxes", None)
                if tgt_boxes is None or tgt_boxes.numel() == 0:
                    continue

                out_item = {
                    "anno": tgt.get("anno", ""),                 # 与 eval_vg 一致
                    "gt_box": tgt_boxes[0].detach().cpu(),       # (4,) cxcywh
                    "pred_topk_boxes": sel_boxes[bi].detach().cpu(),
                    "pred_topk_scores": sel_scores[bi].detach().cpu(),
                }
                output_state_dict.setdefault("items", []).append(out_item)

        _cnt += 1
        if getattr(args, "debug", False) and (_cnt % 15 == 0):
            break

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    odvg_acc_evaluator.synchronize_between_processes()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    stats.update(odvg_acc_evaluator.summarize())

    if getattr(args, "save_results", False):
        import os.path as osp
        savepath = osp.join(args.output_dir, f"results-{utils.get_rank()}.pkl")
        print(f"Saving res to {savepath}")
        torch.save(output_state_dict, savepath)

    return stats, None



