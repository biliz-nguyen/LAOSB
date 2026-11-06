import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import vispy
from vispy.scene import visuals, SceneCanvas
from vispy.scene.cameras import TurntableCamera

from core.datasets.dataset import Dataset
from utils.preprocess import voxelize, voxel_to_points
from utils.postprocess import filter_pred


class DatasetVisualizer:
    """Interactive 3D LiDAR and BEV Dataset Visualizer."""

    def __init__(self, dataset, img_size):
        self.dataset = dataset
        self.index = 0

        # 3D LiDAR point cloud canvas
        self.canvas = SceneCanvas(keys='interactive', show=True, size=(1280, 720), title="3D LiDAR Point Cloud")
        self.canvas.events.key_press.connect(self._key_press)
        self.canvas.events.draw.connect(self._draw)

        self.grid = self.canvas.central_widget.add_grid()
        self.scan_view = vispy.scene.widgets.ViewBox(parent=self.canvas.scene, camera=TurntableCamera(distance=30.0))
        self.grid.add_widget(self.scan_view)

        self.scan_vis = visuals.Markers()
        self.scan_view.add(self.scan_vis)
        visuals.XYZAxis(parent=self.scan_view.scene)
        self.bbox_lines = vispy.scene.visuals.Line(parent=self.scan_view.scene)

        # 2D BEV Map canvas
        self.canvas1 = SceneCanvas(keys='interactive', show=True, size=(img_size[0], img_size[1]), title="BEV Map")
        self.canvas1.events.key_press.connect(self._key_press)
        self.grid1 = self.canvas1.central_widget.add_grid()
        self.bev_view = vispy.scene.widgets.ViewBox(parent=self.canvas1.scene)
        self.grid1.add_widget(self.bev_view)
        self.image = visuals.Image(parent=self.bev_view.scene, cmap='viridis')

        self.update_scan()

    def update_scan(self):
        if self.index >= len(self.dataset):
            return

        data = self.dataset[self.index]
        points = data["points"]
        boxes = data.get("boxes", [])

        # Display points
        if points.shape[1] >= 4:
            colors = self.get_intensity_colors(points)
            self.scan_vis.set_data(points[:, :3], face_color=colors, size=2)
        else:
            self.scan_vis.set_data(points[:, :3], face_color=(0.0, 1.0, 0.5, 0.8), size=2)

        # Display 3D Bounding Boxes
        lines = []
        for box in boxes:
            box_lines = self.build_box_lines(box)
            lines.extend(box_lines)

        if lines:
            self.bbox_lines.set_data(pos=np.array(lines), color=(1.0, 0.2, 0.2, 1.0), width=2, connect='segments')
        else:
            self.bbox_lines.set_data(pos=np.zeros((0, 3)), connect='segments')

        # Display BEV feature
        cls_map = data["cls"].numpy()
        if cls_map.ndim == 3:
            bev_img = np.max(cls_map, axis=0)
        else:
            bev_img = cls_map

        self.image.set_data(bev_img.astype(np.float32))
        print(f"Viewing frame [{self.index + 1}/{len(self.dataset)}]: {self.dataset.data_list[self.index]}")

    def build_box_lines(self, corners):
        """Convert 8 3D box corners to 12 segment coordinate pairs."""
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # front face
            (4, 5), (5, 6), (6, 7), (7, 4),  # back face
            (0, 4), (1, 5), (2, 6), (3, 7),  # connecting edges
        ]
        line_pts = []
        for p1, p2 in edges:
            line_pts.append(corners[p1])
            line_pts.append(corners[p2])
        return line_pts

    def get_intensity_colors(self, points):
        intensity = np.clip(points[:, 3], 0.0, 1.0)
        cmap = plt.get_cmap("viridis")
        return cmap(intensity)

    def _key_press(self, event):
        if event.key in ('N', 'Right', 'Space'):
            if self.index < len(self.dataset) - 1:
                self.index += 1
                self.update_scan()
        elif event.key in ('B', 'Left'):
            if self.index > 0:
                self.index -= 1
                self.update_scan()
        elif event.key in ('Q', 'Escape'):
            self.destroy()

    def destroy(self):
        self.canvas.close()
        self.canvas1.close()
        vispy.app.quit()

    def _draw(self, event):
        if self.canvas.events.key_press.blocked():
            self.canvas.events.key_press.unblock()

    def run(self):
        print("\n=== Controls ===")
        print(" [N] or [Right Arrow] / [Space] : Next scan")
        print(" [B] or [Left Arrow]            : Previous scan")
        print(" [Q] or [Escape]                : Quit")
        print("================\n")
        self.canvas.app.run()


def main():
    parser = argparse.ArgumentParser(description="LiDAR Dataset Visualizer")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--data", type=str, required=True, help="Path to data split manifest (.txt)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    dataset = Dataset(args.data, config["data"], aug_config={}, cls_encoding=config["model"]["cls_encoding"], task="val")
    vis = DatasetVisualizer(dataset, dataset.output_shape)
    vis.run()


if __name__ == "__main__":
    main()
