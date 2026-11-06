import numpy as np
import torch
import torch.nn.functional as F
from shapely.geometry import Polygon


def convert_format(boxes_array):
    """Convert an array of shape (N, 4, 2) to an array of shapely Polygon objects."""
    polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in boxes_array]
    return np.array(polygons)


def compute_iou(box, boxes):
    """Calculates IoU of a given polygon with an array of polygons."""
    iou = [box.intersection(b).area / (box.union(b).area + 1e-8) for b in boxes]
    return np.array(iou, dtype=np.float32)


def non_max_suppression(boxes, scores, threshold):
    """Performs oriented rotated non-maximum suppression (NMS) on bounding boxes.

    Args:
        boxes: (N, 4, 2) corner coordinates.
        scores: 1-D array of detection scores.
        threshold: IoU threshold for suppression.

    Returns:
        np.ndarray: Indices of kept boxes.
    """
    if boxes.shape[0] == 0:
        return np.empty((0,), dtype=np.int32)

    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    polygons = convert_format(boxes)
    ixs = np.argsort(-scores, kind="stable")
    pick = []

    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        if len(ixs) == 1:
            break
        iou = compute_iou(polygons[i], polygons[ixs[1:]])
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)

    return np.array(pick, dtype=np.int32)


def filter_pred(pred, config, out_size_factor, thres, nms_thres=None, cls_encoding="gaussian"):
    """Decode raw model predictions for a single frame into oriented 3D bounding boxes.

    Args:
        pred: Dict with 'cls', 'offset', 'size', 'yaw' tensors.
        config: Data config containing 'geometry'.
        out_size_factor: Downsampling factor of the backbone (e.g. 4).
        thres: Confidence score threshold.
        nms_thres: IoU threshold for rotated NMS (if None, use local peak pooling).
        cls_encoding: 'gaussian' or 'binary'.

    Returns:
        np.ndarray of shape (N, 7): [class_id, score, x, y, length, width, yaw].
    """
    geom = config["geometry"]

    def extract_frame(key):
        value = pred[key].detach().float()
        if value.ndim == 4:
            if value.shape[0] != 1:
                raise ValueError("Decode one frame at a time (batch size must be 1).")
            value = value[0]
        return value

    cls = extract_frame("cls")
    offset = extract_frame("offset")
    size = extract_frame("size")
    heading = extract_frame("yaw")

    scores, ids = cls.max(dim=0)
    height, width = scores.shape

    yy, xx = torch.meshgrid(
        torch.arange(height, device=cls.device),
        torch.arange(width, device=cls.device),
        indexing="ij",
    )

    x = offset[0] + xx * geom["x_res"] * out_size_factor + geom["x_min"]
    y = offset[1] + yy * geom["y_res"] * out_size_factor + geom["y_min"]
    box_w = size[0].exp()
    box_l = size[1].exp()
    yaw = torch.atan2(heading[1], heading[0]) / 2.0

    selected = scores > thres
    if cls_encoding == "binary":
        selected &= ids > 0
        ids = ids - 1  # Map 1-based class index back to 0-based

    # Center-in-ROI filter
    selected &= (x > geom["x_min"]) & (x < geom["x_max"])
    selected &= (y > geom["y_min"]) & (y < geom["y_max"])

    if nms_thres is None:
        pooled = F.max_pool2d(scores[None, None], 3, 1, 1)[0, 0]
        selected &= (scores == pooled)

    if not selected.any():
        return np.empty((0, 7), dtype=np.float32)

    boxes = torch.stack((ids.float(), scores, x, y, box_l, box_w, yaw), dim=-1)[selected]
    boxes = boxes.cpu().numpy().astype(np.float32, copy=False)

    if nms_thres is None:
        return boxes

    # Rotated NMS
    cx, cy, length, width, angle = boxes[:, 2:].T
    cosine, sine = np.cos(angle), np.sin(angle)
    corners = np.empty((len(boxes), 4, 2), dtype=np.float32)
    for i, (sx, sy) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):
        corners[:, i, 0] = cx + sx * length / 2 * cosine - sy * width / 2 * sine
        corners[:, i, 1] = cy + sx * length / 2 * sine + sy * width / 2 * cosine

    keep = non_max_suppression(corners, boxes[:, 1], nms_thres)
    return boxes[keep]
