import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from core.models.torchscript.torchscript_model import TorchscriptModel


def export_torchscript(config_path: str, checkpoint_path: str, output_path: str, half: bool = False):
    """Convert a trained PyTorch model checkpoint to an optimized TorchScript (.pt) model."""
    with open(config_path, "r") as f:
        config = json.load(f)

    model_cfg = config["model"]
    data_cfg = config["data"]
    out_size_factor = data_cfg.get("out_size_factor", 4)
    num_classes = data_cfg.get("num_classes", 4)

    model = TorchscriptModel(model_cfg, out_size_factor=out_size_factor, num_classes=num_classes)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    if half:
        model.half()

    print("Tracing / scripting model with TorchScript...")
    scripted_model = torch.jit.script(model)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    scripted_model.save(output_path)
    print(f"Successfully exported TorchScript model to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export trained LiDAR detection model to TorchScript for deployment")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON configuration file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained PyTorch weights (.pt/.pth)")
    parser.add_argument("--output", type=str, default="deployed_model.pt", help="Path to save TorchScript model (.pt)")
    parser.add_argument("--half", action="store_true", help="Export in FP16 precision")
    args = parser.parse_args()

    export_torchscript(args.config, args.checkpoint, args.output, half=args.half)


if __name__ == "__main__":
    main()
