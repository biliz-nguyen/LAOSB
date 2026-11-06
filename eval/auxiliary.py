import numpy as np
from pyquaternion import Quaternion


def quaternion_yaw(q: Quaternion) -> float:
    """Calculate yaw angle from quaternion in LiDAR coordinate frame."""
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))
    return np.arctan2(v[1], v[0])


def angle_diff(x: float, y: float, period: float = 2 * np.pi) -> float:
    """Smallest signed angle difference from y to x within (-pi, pi)."""
    diff = (x - y + period / 2) % period - period / 2
    if diff > np.pi:
        diff = diff - (2 * np.pi)
    return diff


def center_distance(gt_box, pred_box) -> float:
    """L2 distance between box centers in XY plane."""
    return float(np.linalg.norm(np.array(pred_box[:2]) - np.array(gt_box[:2])))


def yaw_diff(gt_box, pred_box, period: float = 2 * np.pi) -> float:
    """Yaw angle difference in radians between two boxes."""
    return float(abs(angle_diff(gt_box[-1], pred_box[-1], period)))


def scale_iou(gt_size: np.ndarray, pred_size: np.ndarray) -> float:
    """Intersection over Union (IoU) of 3D box dimensions assuming aligned centers."""
    gt_size = np.array(gt_size)
    pred_size = np.array(pred_size)
    min_dim = np.minimum(gt_size, pred_size)
    intersection = np.prod(min_dim)
    union = np.prod(gt_size) + np.prod(pred_size) - intersection
    return float(intersection / (union + 1e-8))
