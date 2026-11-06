import argparse
import os
import random


def split_dataset(pointcloud_dir: str, save_dir: str, train_ratio: float = 0.8, val_ratio: float = 0.1, dataset_name: str = "kitti"):
    """Split pointcloud files into train and val text manifests."""
    if not os.path.isdir(pointcloud_dir):
        raise FileNotFoundError(f"Point cloud directory does not exist: {pointcloud_dir}")

    os.makedirs(save_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(pointcloud_dir) if f.endswith(".bin") or f.endswith(".npy") or f.endswith(".pcd")])
    if not files:
        raise ValueError(f"No point cloud files found in {pointcloud_dir}")

    random.seed(42)
    random.shuffle(files)

    n_total = len(files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]

    def write_manifest(path, file_list):
        with open(path, "w") as f:
            for item in file_list:
                name = os.path.splitext(item)[0]
                f.write(f"{name};{dataset_name}\n")

    train_txt = os.path.join(save_dir, "train.txt")
    val_txt = os.path.join(save_dir, "val.txt")

    write_manifest(train_txt, train_files)
    write_manifest(val_txt, val_files)

    print(f"Dataset split complete:")
    print(f"  Total scans: {n_total}")
    print(f"  Train: {len(train_files)} -> {train_txt}")
    print(f"  Val:   {len(val_files)} -> {val_txt}")


def main():
    parser = argparse.ArgumentParser(description="Split point cloud dataset into train/val manifests")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing point cloud files (.bin)")
    parser.add_argument("--save_dir", type=str, default="data/list", help="Directory to save train.txt and val.txt")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio (default: 0.8)")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Val split ratio (default: 0.1)")
    parser.add_argument("--dataset_name", type=str, default="kitti", help="Dataset identifier string")
    args = parser.parse_args()

    split_dataset(args.data_dir, args.save_dir, args.train_ratio, args.val_ratio, args.dataset_name)


if __name__ == "__main__":
    main()
