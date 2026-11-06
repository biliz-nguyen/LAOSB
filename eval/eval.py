import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from shapely.geometry import Polygon

from core.models.model import CustomModel
from core.datasets.dataset import Dataset
from utils.postprocess import filter_pred, non_max_suppression


def box_to_polygon_2d(x, y, l, w, yaw):
    """Compute 4 BEV corners and return a Shapely Polygon."""
    cos_t, sin_t = np.cos(yaw), np.sin(yaw)
    dx1, dy1 = l / 2.0 * cos_t, l / 2.0 * sin_t
    dx2, dy2 = -w / 2.0 * sin_t, w / 2.0 * cos_t

    c1 = (x + dx1 + dx2, y + dy1 + dy2)
    c2 = (x - dx1 + dx2, y - dy1 + dy2)
    c3 = (x - dx1 - dx2, y - dy1 - dy2)
    c4 = (x + dx1 - dx2, y + dy1 - dy2)
    return Polygon([c1, c2, c3, c4])


def compute_bev_iou(box1, box2):
    """Compute 2D BEV IoU between two oriented bounding boxes [x, y, l, w, yaw]."""
    try:
        p1 = box_to_polygon_2d(box1[0], box1[1], box1[2], box1[3], box1[4])
        p2 = box_to_polygon_2d(box2[0], box2[1], box2[2], box2[3], box2[4])
        if not p1.is_valid or not p2.is_valid:
            return 0.0
        inter = p1.intersection(p2).area
        union = p1.union(p2).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def evaluate_detector(model, data_loader, dataset, device, score_thresh=0.2, nms_thresh=0.1, iou_thresh=0.5):
    """Run evaluation and compute True Positives, False Positives, False Negatives, Precision, Recall, and AP."""
    model.eval()

    all_matches = []  # list of (score, is_tp)
    total_gt_boxes = 0

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating"):
            voxel = data["voxel"].to(device)
            pred = model(voxel)

            if dataset.cls_encoding == "binary":
                pred["cls"] = F.softmax(pred["cls"], dim=1)
            else:
                pred["cls"] = pred["cls"].sigmoid()

            dtype = data["dtype"][0] if "dtype" in data else list(dataset.config.keys())[0]
            boxes_pred = filter_pred(
                pred,
                dataset.config[dtype],
                dataset.out_size_factor,
                thres=score_thresh,
                nms_thres=nms_thresh,
                cls_encoding=dataset.cls_encoding,
            )

            # Ground truth boxes in metric coords [class, h, w, l, x, y, z, yaw]
            gt_3d = data.get("boxes", [])
            # Convert GT to [x, y, l, w, yaw]
            gt_list = []
            for b in gt_3d:
                # b: [front_left_top, ...] or [class, h, w, l, x, y, z, yaw]
                if isinstance(b, torch.Tensor):
                    b = b.cpu().numpy()
                if len(b) == 8:
                    gt_list.append([b[4], b[5], b[3], b[2], b[7]])

            total_gt_boxes += len(gt_list)
            gt_matched = [False] * len(gt_list)

            # Sort predictions by score descending
            if len(boxes_pred) > 0:
                sort_idx = np.argsort(-boxes_pred[:, 1])
                boxes_pred = boxes_pred[sort_idx]

                for p_box in boxes_pred:
                    score = p_box[1]
                    pred_bev = [p_box[2], p_box[3], p_box[4], p_box[5], p_box[6]]

                    best_iou = 0.0
                    best_gt_idx = -1
                    for g_idx, g_box in enumerate(gt_list):
                        if not gt_matched[g_idx]:
                            iou = compute_bev_iou(pred_bev, g_box)
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = g_idx

                    if best_iou >= iou_thresh and best_gt_idx >= 0:
                        all_matches.append((score, 1))
                        gt_matched[best_gt_idx] = True
                    else:
                        all_matches.append((score, 0))

    if not all_matches or total_gt_boxes == 0:
        print("No detections or ground truth found.")
        return {"precision": 0.0, "recall": 0.0, "ap": 0.0, "total_gt": total_gt_boxes}

    # Sort all predictions globally by score descending
    all_matches.sort(key=lambda x: -x[0])
    scores = np.array([m[0] for m in all_matches])
    is_tp = np.array([m[1] for m in all_matches])

    tp_cum = np.cumsum(is_tp)
    fp_cum = np.cumsum(1 - is_tp)

    recalls = tp_cum / float(total_gt_boxes)
    precisions = tp_cum / (tp_cum + fp_cum)

    # All-point interpolated AP
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    final_precision = float(precisions[-1]) if len(precisions) > 0 else 0.0
    final_recall = float(recalls[-1]) if len(recalls) > 0 else 0.0

    print(f"\n================ Evaluation Results (IoU@{iou_thresh}) ================")
    print(f"Total Ground Truth: {total_gt_boxes}")
    print(f"Total Predictions:  {len(all_matches)}")
    print(f"True Positives:     {int(tp_cum[-1])}")
    print(f"Precision:          {final_precision:.4f}")
    print(f"Recall:             {final_recall:.4f}")
    print(f"Average Precision:  {ap:.4f}")
    print("=================================================================\n")

    return {
        "precision": final_precision,
        "recall": final_recall,
        "ap": float(ap),
        "total_gt": total_gt_boxes,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate 3D LiDAR Object Detection Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--data", type=str, default=None, help="Evaluation dataset list manifest file (.txt)")
    parser.add_argument("--score_thresh", type=float, default=0.2, help="Confidence threshold")
    parser.add_argument("--nms_thresh", type=float, default=0.1, help="NMS IoU threshold")
    parser.add_argument("--iou_thresh", type=float, default=0.5, help="Matching IoU threshold")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    device = args.device or config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    data_file = args.data or config.get("val", {}).get("data")
    if not data_file:
        raise ValueError("Please provide --data manifest file path.")

    dataset = Dataset(data_file, config["data"], aug_config={}, cls_encoding=config["model"]["cls_encoding"], task="val")
    data_loader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=2)

    model = CustomModel(config["model"], config["data"]["num_classes"])
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    evaluate_detector(
        model,
        data_loader,
        dataset,
        device,
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh,
        iou_thresh=args.iou_thresh,
    )


if __name__ == "__main__":
    main()
