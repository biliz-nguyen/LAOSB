import math
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.preprocess import voxelize, trasform_label2metric, get_points_in_a_rotated_box
from utils.transform import Random_Rotation, Random_Scaling, OneOf, Random_Translation
from utils.gaussian import gaussian_radius, draw_heatmap_gaussian


class Dataset(Dataset):
    """Dataset loader for 3D LiDAR point cloud object detection."""

    def __init__(self, data_file: str, config: dict, aug_config: dict = None, cls_encoding: str = "gaussian", task: str = "train") -> None:
        self.data_file = data_file
        self.config = config
        self.task = task
        self.cls_encoding = cls_encoding
        self.num_classes = self.config["num_classes"]

        if cls_encoding == "binary":
            self.num_classes += 1

        if aug_config:
            self.transforms = self.get_transforms(aug_config)
            self.augment = OneOf(self.transforms, aug_config.get("p", 0.5))
        else:
            self.augment = None

        self.out_size_factor = config["out_size_factor"]
        self.create_data_list()

        geom = self.config[self.data_type_list[0]]["geometry"]
        self.output_shape = [
            int((geom["x_max"] - geom["x_min"]) / geom["x_res"] / self.out_size_factor),
            int((geom["y_max"] - geom["y_min"]) / geom["y_res"] / self.out_size_factor),
        ]

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> dict:
        file = f"{self.data_list[idx]}.bin"
        data_type = self.data_type_list[idx]

        pointcloud_folder = os.path.join(self.config[data_type]["location"], "pointcloud")
        lidar_path = os.path.join(pointcloud_folder, file)
        points = self.read_points(lidar_path)

        geom = self.config[data_type]["geometry"]

        if self.task == "test":
            scan = voxelize(points, geom)
            scan = torch.from_numpy(scan).permute(2, 0, 1)
            return {
                "voxel": scan,
                "points": points,
                "dtype": data_type,
            }

        boxes = self.get_boxes(idx)

        if self.task == "train" and boxes.shape[0] != 0 and self.augment is not None:
            points, boxes[:, 1:] = self.augment(points, boxes[:, 1:8])

        boxes = self.filter_boxes(boxes, data_type)

        scan = voxelize(points, geom)
        scan = torch.from_numpy(scan).permute(2, 0, 1)

        labels = self.get_label(boxes, geom)
        labels["voxel"] = scan

        if self.task == "val":
            class_list, boxes_3d = self.read_bbox(boxes)
            labels["cls_list"] = class_list
            labels["points"] = points
            labels["boxes"] = boxes_3d
            labels["dtype"] = data_type
            return labels

        return labels

    def read_points(self, lidar_path: str) -> np.ndarray:
        raw = np.fromfile(lidar_path, dtype=np.float32)
        if raw.size % 4 == 0:
            return raw.reshape(-1, 4)
        return raw.reshape(-1, 3)

    def get_boxes(self, idx: int) -> np.ndarray:
        f_name = f"{self.data_list[idx]}.txt"
        data_type = self.data_type_list[idx]
        label_folder = os.path.join(self.config[data_type]["location"], "label")
        label_path = os.path.join(label_folder, f_name)
        object_list = self.config[data_type]["objects"]
        boxes = []

        if not os.path.exists(label_path):
            return np.empty((0, 8), dtype=np.float32)

        with open(label_path, "r") as f:
            for line in f:
                entry = line.split()
                if not entry:
                    continue
                name = entry[0]
                if name in object_list:
                    bbox = [object_list[name]]
                    bbox.extend([float(e) for e in entry[1:]])
                    boxes.append(bbox)

        result = np.asarray(boxes, dtype=np.float32).reshape(-1, 8)
        return result

    def get_label(self, boxes: torch.Tensor, geometry: dict) -> dict:
        offset_map = torch.zeros((self.output_shape[0], self.output_shape[1], 2))
        size_map = torch.zeros((self.output_shape[0], self.output_shape[1], 2))
        yaw_map = torch.zeros((self.output_shape[0], self.output_shape[1], 2))
        reg_mask = torch.zeros(self.output_shape)

        if self.cls_encoding == "binary":
            cls_map = torch.zeros((self.output_shape[0], self.output_shape[1]), dtype=torch.int64)
        else:
            cls_map = torch.zeros((self.num_classes, self.output_shape[0], self.output_shape[1]))

        for i in range(boxes.shape[0]):
            box = boxes[i]
            radius = self.update_cls_map(cls_map, box, geometry)
            self.update_reg_map(offset_map, size_map, yaw_map, reg_mask, radius, box, geometry)

        if self.cls_encoding == "binary":
            cls_map = cls_map.permute(1, 0)
        else:
            cls_map = cls_map.permute(0, 2, 1)

        return {
            "cls": cls_map,
            "offset": offset_map.permute(2, 1, 0),
            "yaw": yaw_map.permute(2, 1, 0),
            "size": size_map.permute(2, 1, 0),
            "reg_mask": reg_mask.permute(1, 0),
        }

    def update_cls_map(self, cls_map: torch.Tensor, box: torch.Tensor, geometry: dict) -> float:
        width = box[2] / geometry["x_res"] / self.out_size_factor
        length = box[3] / geometry["y_res"] / self.out_size_factor
        radius = 0.0

        if width > 0 and length > 0:
            radius = gaussian_radius((length, width), min_overlap=self.config.get("gaussian_overlap", 0.1))
            radius = max(self.config.get("min_radius", 4), int(radius))
            x, y = box[4], box[5]

            coor_x = (x - geometry["x_min"]) / geometry["x_res"] / self.out_size_factor
            coor_y = (y - geometry["y_min"]) / geometry["y_res"] / self.out_size_factor

            center = torch.tensor([coor_y, coor_x], dtype=torch.float32)
            center_int = center.to(torch.int32)

            if self.cls_encoding == "gaussian":
                draw_heatmap_gaussian(cls_map[int(box[0])], center_int, radius)
                radius = radius / 2.0
            elif self.cls_encoding == "inverse_distance":
                x_min = max(0, int(coor_x - radius))
                x_max = min(self.output_shape[0], int(coor_x + radius))
                y_min = max(0, int(coor_y - radius))
                y_max = min(self.output_shape[1], int(coor_y + radius))

                X, Y = np.ogrid[x_min:x_max, y_min:y_max]
                dist = np.sqrt((X - int(coor_x)) ** 2 + (Y - int(coor_y)) ** 2)
                inv_dist = np.copy(dist)
                inv_dist[inv_dist == 0] = 1.0
                inv_dist = 1.0 / inv_dist

                dist_t = torch.from_numpy(dist)
                inv_t = torch.from_numpy(inv_dist)
                cls_map[int(box[0])][x_min:x_max, y_min:y_max] = inv_t
                cls_map[int(box[0])][x_min:x_max, y_min:y_max][dist_t > radius] = 0.0
                radius = radius / 2.0
            else:
                x_min = max(0, int(coor_x - radius))
                x_max = min(self.output_shape[0], int(coor_x + radius))
                y_min = max(0, int(coor_y - radius))
                y_max = min(self.output_shape[1], int(coor_y + radius))
                for p_x in range(x_min, x_max):
                    for p_y in range(y_min, y_max):
                        if math.sqrt((p_x - coor_x) ** 2 + (p_y - coor_y) ** 2) < radius:
                            cls_map[p_x][p_y] = int(box[0]) + 1

        return radius

    def update_reg_map(self, offset_map, size_map, yaw_map, reg_mask, radius, box, geometry):
        cls, h, w, l, x, y, z, yaw = box
        yaw2 = math.fmod(2.0 * yaw, 2.0 * math.pi)

        coor_x = (x - geometry["x_min"]) / geometry["x_res"] / self.out_size_factor
        coor_y = (y - geometry["y_min"]) / geometry["y_res"] / self.out_size_factor

        x_min = max(0, int(coor_x - radius))
        x_max = min(self.output_shape[0], int(coor_x + radius))
        y_min = max(0, int(coor_y - radius))
        y_max = min(self.output_shape[1], int(coor_y + radius))

        for p_x in range(x_min, x_max):
            for p_y in range(y_min, y_max):
                if math.sqrt((p_x - coor_x) ** 2 + (p_y - coor_y) ** 2) < radius:
                    metric_x = p_x * self.out_size_factor * geometry["x_res"] + geometry["x_min"]
                    metric_y = p_y * self.out_size_factor * geometry["y_res"] + geometry["y_min"]

                    offset_map[p_x][p_y][0] = x - metric_x
                    offset_map[p_x][p_y][1] = y - metric_y
                    size_map[p_x][p_y][0] = math.log(w)
                    size_map[p_x][p_y][1] = math.log(l)
                    yaw_map[p_x][p_y][0] = math.cos(yaw2)
                    yaw_map[p_x][p_y][1] = math.sin(yaw2)
                    reg_mask[p_x][p_y] = 1

    def read_bbox(self, boxes):
        corner_list = []
        class_list = []
        for i in range(boxes.shape[0]):
            box = boxes[i]
            class_list.append(box[0])
            corner_list.append(self.get3D_corners(box))
        return class_list, corner_list

    def filter_boxes(self, boxes: np.ndarray, data_type: str) -> torch.Tensor:
        filtered = []
        geom = self.config[data_type]["geometry"]
        for i in range(boxes.shape[0]):
            box = boxes[i]
            x, y = box[4:6]
            if (x > geom["x_min"]) and (x < geom["x_max"]) and (y > geom["y_min"]) and (y < geom["y_max"]):
                filtered.append(box)
        return torch.from_numpy(np.asarray(filtered, dtype=np.float32).reshape(-1, 8))

    def get3D_corners(self, bbox):
        h, w, l, x, y, z, yaw = bbox[1:]
        front, back = l / 2.0, -l / 2.0
        left, right = w / 2.0, -w / 2.0
        top, bottom = h, 0.0

        corners = [
            [front, left, top], [front, left, bottom],
            [front, right, bottom], [front, right, top],
            [back, left, top], [back, left, bottom],
            [back, right, bottom], [back, right, top],
        ]

        rot_corners = []
        for corner in corners:
            rot = self.rotate_pointZ(corner, yaw)[0] + np.array([x, y, z])
            rot_corners.append(rot)
        return rot_corners

    def rotate_pointZ(self, point, yaw):
        rot_mat = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw),  np.cos(yaw), 0.0],
            [0.0,          0.0,         1.0],
        ])
        return np.reshape(np.matmul(rot_mat, np.reshape(point, (3, 1))), (1, 3))

    def get_transforms(self, config: dict) -> list:
        transforms = []
        if config.get("rotation", {}).get("use", False):
            transforms.append(Random_Rotation(config["rotation"]["limit_angle"], config["rotation"].get("p", 1.0)))
        if config.get("scaling", {}).get("use", False):
            transforms.append(Random_Scaling(config["scaling"]["range"], config["scaling"].get("p", 1.0)))
        if config.get("translation", {}).get("use", False):
            transforms.append(Random_Translation(config["translation"]["scale"], config["translation"].get("p", 1.0)))
        return transforms

    def create_data_list(self):
        data_list, data_type_list = [], []
        if not os.path.exists(self.data_file):
            print(f"Warning: Data manifest file not found: {self.data_file}")
            self.data_list, self.data_type_list = [], []
            return

        with open(self.data_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(";")
                    data_list.append(parts[0])
                    data_type_list.append(parts[1] if len(parts) > 1 else list(self.config.keys())[0])

        self.data_list = data_list
        self.data_type_list = data_type_list
