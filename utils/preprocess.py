import math
import numpy as np
import torch
import torch.nn.functional as F


def voxelize(points: np.ndarray, geometry: dict) -> np.ndarray:
    """Convert raw 3D point cloud into bird's-eye-view (BEV) voxel occupancy grid.

    Args:
        points: (N, 3+) array of LiDAR points [x, y, z, ...].
        geometry: dict specifying x_min, x_max, y_min, y_max, z_min, z_max,
                  and resolutions x_res, y_res, z_res.

    Returns:
        np.ndarray: Voxel occupancy grid of shape (C, H, W).
    """
    x_min, x_max = geometry["x_min"], geometry["x_max"]
    y_min, y_max = geometry["y_min"], geometry["y_max"]
    z_min, z_max = geometry["z_min"], geometry["z_max"]
    x_res, y_res, z_res = geometry["x_res"], geometry["y_res"], geometry["z_res"]

    x_size = int((x_max - x_min) / x_res)
    y_size = int((y_max - y_min) / y_res)
    z_size = int((z_max - z_min) / z_res)

    eps = 0.001
    x_indexes = np.logical_and(points[:, 0] > x_min + eps, points[:, 0] < x_max - eps)
    y_indexes = np.logical_and(points[:, 1] > y_min + eps, points[:, 1] < y_max - eps)
    z_indexes = np.logical_and(points[:, 2] > z_min + eps, points[:, 2] < z_max - eps)
    pts = points[np.logical_and(np.logical_and(x_indexes, y_indexes), z_indexes)]

    occupancy_mask = np.zeros((pts.shape[0], 3), dtype=np.int32)
    voxels = np.zeros((x_size, y_size, z_size), dtype=np.float32)
    occupancy_mask[:, 0] = (pts[:, 0] - x_min) // x_res
    occupancy_mask[:, 1] = (pts[:, 1] - y_min) // y_res
    occupancy_mask[:, 2] = (pts[:, 2] - z_min) // z_res

    idxs = np.array([
        occupancy_mask[:, 0].reshape(-1),
        occupancy_mask[:, 1].reshape(-1),
        occupancy_mask[:, 2].reshape(-1)
    ])

    voxels[idxs[0], idxs[1], idxs[2]] = 1
    return np.swapaxes(voxels, 0, 1)


def voxel_to_points(voxel: np.ndarray, geometry: dict) -> np.ndarray:
    """Reconstruct 3D points from active voxel centers."""
    x_min = geometry["x_min"]
    y_min = geometry["y_min"]
    z_min = geometry["z_min"]
    x_res = geometry["x_res"]
    y_res = geometry["y_res"]
    z_res = geometry["z_res"]

    xs, ys, zs = np.where(voxel.astype(int) == 1)
    points_x = xs + x_res / 2
    points_y = ys + y_res / 2
    points_z = zs + z_res / 2

    points_x = points_x * x_res + x_min
    points_y = points_y * y_res + y_min
    points_z = points_z * z_res + z_min
    return np.transpose(np.array([points_x, points_y, points_z]))


def trasform_label2metric(label: np.ndarray, geometry: dict, ratio: int = 4) -> np.ndarray:
    """Map label coordinates from grid indices to metric space (meters)."""
    metric = np.copy(label).astype(np.float32)
    metric[..., 0] = metric[..., 0] * ratio * geometry["x_res"] + geometry["x_min"]
    metric[..., 1] = metric[..., 1] * ratio * geometry["y_res"] + geometry["y_min"]
    return metric


def transform_metric2label(metric: np.ndarray, ratio: int = 4, grid_size: float = 0.1, base_height: float = 100) -> np.ndarray:
    """Map coordinates from metric space (meters) to label grid space."""
    label = (metric / ratio) / grid_size
    label[..., 1] += base_height
    return label


def get_points_in_a_rotated_box(corners: np.ndarray, label_shape: list = [200, 175]):
    """Rasterize rotated 2D bounding box to find all grid pixel coordinates inside it."""
    def minY(x0, y0, x1, y1, x):
        if x0 == x1:
            return int(math.floor(y0))
        m = (y1 - y0) / (x1 - x0)
        if m >= 0.0:
            return int(math.floor(y0 + m * (x - x0)))
        else:
            return int(math.floor(y0 + m * ((x + 1.0) - x0)))

    def maxY(x0, y0, x1, y1, x):
        if x0 == x1:
            return int(math.ceil(y1))
        m = (y1 - y0) / (x1 - x0)
        if m >= 0.0:
            return int(math.ceil(y0 + m * ((x + 1.0) - x0)))
        else:
            return int(math.ceil(y0 + m * (x - x0)))

    view = [(corners[i, 0], corners[i, 1]) for i in range(4)]
    pixels = []

    l, m1, m2, r = sorted(view, key=lambda p: (p[0], p[1]))
    b, t = sorted([m1, m2], key=lambda p: (p[1], p[0]))

    lx, ly = l
    rx, ry = r
    bx, by = b
    tx, ty = t
    m1x, m1y = m1
    m2x, m2y = m2

    xmin, ymin = 0, 0
    xmax, ymax = label_shape[1], label_shape[0]

    lxi = max(int(math.ceil(lx)), xmin)
    rxi = min(int(math.floor(rx)), xmax)
    byi = max(int(math.ceil(by)), ymin)
    tyi = min(int(math.floor(ty)), ymax)

    for x in range(lxi, rxi):
        xf = float(x)
        if xf < m1x:
            y1 = minY(lx, ly, bx, by, xf)
            y2 = maxY(lx, ly, tx, ty, xf)
        elif xf < m2x:
            if m1y < m2y:
                y1 = minY(bx, by, rx, ry, xf)
                y2 = maxY(lx, ly, tx, ty, xf)
            else:
                y1 = minY(lx, ly, bx, by, xf)
                y2 = maxY(tx, ty, rx, ry, xf)
        else:
            y1 = minY(bx, by, rx, ry, xf)
            y2 = maxY(tx, ty, rx, ry, xf)

        y1 = max(y1, byi)
        y2 = min(y2, tyi)

        for y in range(y1, y2):
            pixels.append((x, y))

    return pixels


def voxel_to_img(voxel: torch.Tensor) -> np.ndarray:
    """Convert voxel grid to 2D top-down visualization image."""
    voxel_np = voxel.permute(1, 2, 0).cpu().numpy()
    max_inds = np.argmax(voxel_np, axis=2)
    img = np.zeros((voxel_np.shape[0], voxel_np.shape[1]), dtype=np.uint8)
    for i in range(voxel_np.shape[0]):
        for j in range(voxel_np.shape[1]):
            idx = max_inds[i][j]
            img[i][j] = int(idx / voxel_np.shape[2] * 255)
    return img
