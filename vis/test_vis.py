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
import vispy
from vispy.scene import visuals, SceneCanvas
from vispy.scene.cameras import TurntableCamera

from core.models.model import CustomModel
from core.datasets.dataset import Dataset
from utils.postprocess import filter_pred


class PredictionVisualizer:
    """Interactive 3D LiDAR Object Detection Prediction Visualizer."""

    def __init__(self, data_loader, model, device, config, score_thresh=0.3, nms_thresh=0.1):
        self.data_loader = data_loader
        self.dataset = data_loader.dataset
        self.model = model
        self.model.eval()
        self.device = device
        self.config = config
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh

        self.index = 0
        self.draw_gt = True
        self.data_iter = iter(self.data_loader)

        # 3D LiDAR Canvas
        self.canvas = SceneCanvas(keys='interactive', show=True, size=(1280, 720), title="3D LiDAR Detections")
        self.canvas.events.key_press.connect(self._key_press)
        self.canvas.events.draw.connect(self._draw)

        self.grid = self.canvas.central_widget.add_grid()
        self.scan_view = vispy.scene.widgets.ViewBox(parent=self.canvas.scene, camera=TurntableCamera(distance=30.0))
        self.grid.add_widget(self.scan_view)

        self.scan_vis = visuals.Markers()
        self.scan_view.add(self.scan_vis)
        visuals.XYZAxis(parent=self.scan_view.scene)

        # Lines for predicted and GT boxes
        self.pred_lines = vispy.scene.visuals.Line(parent=self.scan_view.scene)
        self.gt_lines = vispy.scene.visuals.Line(parent=self.scan_view.scene)

        # BEV Prediction Heatmap Canvas
        self.canvas_bev = SceneCanvas(keys='interactive', show=True, size=(400, 400), title="BEV Confidence Heatmap")
        self.grid_bev = self.canvas_bev.central_widget.add_grid()
        self.bev_view = vispy.scene.widgets.ViewBox(parent=self.canvas_bev.scene)
        self.grid_bev.add_widget(self.bev_view)
        self.bev_image = visuals.Image(parent=self.bev_view.scene, cmap='hot')

        self.samples = list(self.data_loader)
        self.update_frame()

    def update_frame(self):
        if self.index >= len(self.samples):
            return

        data = self.samples[self.index]
        voxel = data["voxel"].to(self.device)
        points = data["points"][0].numpy() if isinstance(data["points"], torch.Tensor) else data["points"]
        gt_boxes = data.get("boxes", [])

        # Model inference
        with torch.no_grad():
            pred = self.model(voxel)
            if self.dataset.cls_encoding == "binary":
                pred["cls"] = F.softmax(pred["cls"], dim=1)
            else:
                pred["cls"] = pred["cls"].sigmoid()

            dtype = data["dtype"][0] if "dtype" in data else list(self.dataset.config.keys())[0]
            boxes_pred = filter_pred(
                pred,
                self.dataset.config[dtype],
                self.dataset.out_size_factor,
                thres=self.score_thresh,
                nms_thres=self.nms_thresh,
                cls_encoding=self.dataset.cls_encoding,
            )

        # Plot point cloud
        self.scan_vis.set_data(points[:, :3], face_color=(0.2, 0.7, 0.9, 0.8), size=2)

        # Plot predictions (Green)
        pred_line_pts = []
        for b in boxes_pred:
            corners = self.get_corners_from_pred(b)
            pred_line_pts.extend(self.build_box_lines(corners))

        if pred_line_pts:
            self.pred_lines.set_data(pos=np.array(pred_line_pts), color=(0.1, 1.0, 0.2, 1.0), width=2, connect='segments')
        else:
            self.pred_lines.set_data(pos=np.zeros((0, 3)), connect='segments')

        # Plot ground truth (Red)
        gt_line_pts = []
        if self.draw_gt:
            for b in gt_boxes:
                if len(b) == 8:
                    corners = b
                else:
                    corners = self.dataset.get3D_corners(b)
                gt_line_pts.extend(self.build_box_lines(corners))

        if gt_line_pts:
            self.gt_lines.set_data(pos=np.array(gt_line_pts), color=(1.0, 0.2, 0.2, 1.0), width=2, connect='segments')
        else:
            self.gt_lines.set_data(pos=np.zeros((0, 3)), connect='segments')

        # Plot BEV heatmap
        cls_pred = pred["cls"][0].detach().cpu().numpy()
        bev_max = np.max(cls_pred, axis=0)
        self.bev_image.set_data(bev_max)

        print(f"Viewing frame [{self.index + 1}/{len(self.samples)}] | Detections: {len(boxes_pred)} | GT: {len(gt_boxes)}")

    def get_corners_from_pred(self, box):
        """box format: [class_id, score, x, y, length, width, yaw]."""
        cls_id, score, x, y, l, w, yaw = box
        h = 1.8  # Default vertical extent for visualization
        z = 0.0

        front, back = l / 2.0, -l / 2.0
        left, right = w / 2.0, -w / 2.0
        top, bottom = h / 2.0, -h / 2.0

        corners = [
            [front, left, top], [front, left, bottom],
            [front, right, bottom], [front, right, top],
            [back, left, top], [back, left, bottom],
            [back, right, bottom], [back, right, top],
        ]

        rot_mat = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw),  np.cos(yaw), 0.0],
            [0.0,          0.0,         1.0],
        ])

        rot_corners = []
        for corner in corners:
            rot = np.dot(rot_mat, np.array(corner)) + np.array([x, y, z])
            rot_corners.append(rot)
        return rot_corners

    def build_box_lines(self, corners):
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        line_pts = []
        for p1, p2 in edges:
            line_pts.append(corners[p1])
            line_pts.append(corners[p2])
        return line_pts

    def _key_press(self, event):
        if event.key in ('N', 'Right', 'Space'):
            if self.index < len(self.samples) - 1:
                self.index += 1
                self.update_frame()
        elif event.key in ('B', 'Left'):
            if self.index > 0:
                self.index -= 1
                self.update_frame()
        elif event.key == 'G':
            self.draw_gt = not self.draw_gt
            print(f"Ground Truth display: {'ON' if self.draw_gt else 'OFF'}")
            self.update_frame()
        elif event.key in ('Q', 'Escape'):
            self.destroy()

    def destroy(self):
        self.canvas.close()
        self.canvas_bev.close()
        vispy.app.quit()

    def _draw(self, event):
        if self.canvas.events.key_press.blocked():
            self.canvas.events.key_press.unblock()

    def run(self):
        print("\n=== Controls ===")
        print(" [N] / [Right Arrow] / [Space] : Next scan")
        print(" [B] / [Left Arrow]            : Previous scan")
        print(" [G]                           : Toggle Ground Truth (Red boxes)")
        print(" [Q] / [Escape]                : Quit")
        print("================\n")
        self.canvas.app.run()


def main():
    parser = argparse.ArgumentParser(description="LiDAR 3D Object Detection Visualizer")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--data", type=str, required=True, help="Path to data split manifest (.txt)")
    parser.add_argument("--score_thresh", type=float, default=0.25, help="Detection score threshold")
    parser.add_argument("--nms_thresh", type=float, default=0.1, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    device = args.device or config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    dataset = Dataset(args.data, config["data"], aug_config={}, cls_encoding=config["model"]["cls_encoding"], task="val")
    data_loader = DataLoader(dataset, shuffle=False, batch_size=1)

    model = CustomModel(config["model"], config["data"]["num_classes"])
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    vis = PredictionVisualizer(
        data_loader,
        model,
        device,
        config,
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh,
    )
    vis.run()


if __name__ == "__main__":
    main()
