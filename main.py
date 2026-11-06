#!/usr/bin/env python3
"""
Main training entrypoint for 3D LiDAR Bird's-Eye-View Object Detection.
Supports PIXOR, MobilePIXOR, RPN, and PointPillars backbones.
"""

import argparse
import json
import os
import sys

from core.train_agent import TrainAgent


def main():
    parser = argparse.ArgumentParser(description="LiDAR BEV 3D Object Detection (PIXOR / MobilePIXOR)")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON configuration file")
    parser.add_argument("--device", type=str, default=None, help="Device override ('cuda', 'cpu', 'mps')")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch count override")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
    parser.add_argument("--note", type=str, default=None, help="Experiment run description tag")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, "r") as f:
        config = json.load(f)

    if args.device:
        config["device"] = args.device
    if args.epochs:
        config["train"]["epochs"] = args.epochs
    if args.batch_size:
        config["train"]["batch_size"] = args.batch_size
    if args.note:
        config["note"] = args.note

    print(f"Loaded config: {args.config}")
    print(f"Backbone:      {config['model']['backbone']}")
    print(f"Target epochs: {config['train']['epochs']}")
    print(f"Batch size:    {config['train']['batch_size']}")

    agent = TrainAgent(config)
    agent.train()


if __name__ == "__main__":
    main()
