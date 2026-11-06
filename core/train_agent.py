import os
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:
        """No-op fallback when TensorBoard is not installed."""
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def close(self): pass

from core.datasets.dataset import Dataset
from core.models.model import CustomModel
from core.losses.loss_fn import LossFunction


class TrainAgent:
    """Handles end-to-end training and validation of the LiDAR 3D detector."""

    def __init__(self, config: dict):
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available on this machine. Falling back to CPU.")
            self.device = "cpu"

        self.multi_gpu = config.get("multi_gpu", False)
        self.prev_val_loss = float("inf")

        self.make_experiments_dirs()
        self.prepare_loaders()
        self.build_model()
        self.writer = SummaryWriter(self.runs_dir)

    def prepare_loaders(self):
        data_cfg = self.config["data"]
        aug_cfg = self.config.get("augmentation", {})
        model_cfg = self.config["model"]
        train_cfg = self.config["train"]
        val_cfg = self.config.get("val", {})

        num_workers = self.config.get("num_workers", 4)
        prefetch_factor = self.config.get("prefetch_factor", 2)
        pin_memory = self.config.get("pin_memory", True) if self.device == "cuda" else False

        train_dataset = Dataset(train_cfg["data"], data_cfg, aug_cfg, model_cfg["cls_encoding"], "train")
        loader_kwargs = {"shuffle": True, "batch_size": train_cfg["batch_size"], "num_workers": num_workers}
        if num_workers > 0:
            loader_kwargs.update({"prefetch_factor": prefetch_factor, "pin_memory": pin_memory})
        self.train_loader = DataLoader(train_dataset, **loader_kwargs)

        val_dataset = Dataset(val_cfg.get("data", train_cfg["data"]), data_cfg, aug_cfg, model_cfg["cls_encoding"], "val")
        val_loader_kwargs = {"shuffle": False, "batch_size": val_cfg.get("batch_size", train_cfg["batch_size"]), "num_workers": num_workers}
        if num_workers > 0:
            val_loader_kwargs.update({"prefetch_factor": prefetch_factor, "pin_memory": pin_memory})
        self.val_loader = DataLoader(val_dataset, **val_loader_kwargs)

    def build_model(self):
        train_cfg = self.config["train"]
        learning_rate = train_cfg["learning_rate"]
        weight_decay = train_cfg.get("weight_decay", 0.0005)

        self.model = CustomModel(self.config["model"], self.config["data"]["num_classes"])
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model backbone: {self.config['model']['backbone']}")
        print(f"Trainable parameters: {num_params:,}")

        if self.multi_gpu and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)

        self.loss = LossFunction(self.config["model"]["cls_encoding"], self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        decay_milestones = train_cfg.get("lr_decay_at", [15, 25])
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=decay_milestones,
            gamma=0.1,
        )

    def train_one_epoch(self, epoch: int):
        total_loss = {}
        start_time = time.time()
        self.model.train()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:02d} [Train]")
        for data in pbar:
            for key in data:
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(data["voxel"])
            loss_dict = self.loss(pred, data)

            loss = loss_dict["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Loss became non-finite at epoch {epoch}: {loss.item()}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()

            for key in loss_dict:
                val = loss_dict[key].item() if isinstance(loss_dict[key], torch.Tensor) else loss_dict[key]
                total_loss[key] = total_loss.get(key, 0.0) + val

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_time = time.time() - start_time
        num_batches = max(len(self.train_loader), 1)
        avg_loss = total_loss["loss"] / num_batches
        for key in total_loss:
            self.writer.add_scalar(f"train/{key}", total_loss[key] / num_batches, epoch)

        print(f"Epoch {epoch:02d} | Train Time: {epoch_time:.1f}s | Train Loss: {avg_loss:.5f}")

    def validate(self, epoch: int):
        total_loss = {}
        total_samples = 0
        start_time = time.time()
        self.model.eval()

        with torch.no_grad():
            for data in tqdm(self.val_loader, desc=f"Epoch {epoch:02d} [Val]"):
                for key in data:
                    if isinstance(data[key], torch.Tensor):
                        data[key] = data[key].to(self.device)

                pred = self.model(data["voxel"])
                loss_dict = self.loss(pred, data)

                batch_size = data["voxel"].shape[0]
                total_samples += batch_size

                for key in loss_dict:
                    val = loss_dict[key].item() if isinstance(loss_dict[key], torch.Tensor) else loss_dict[key]
                    total_loss[key] = total_loss.get(key, 0.0) + val * batch_size

        self.model.train()
        avg_loss = total_loss["loss"] / max(total_samples, 1)
        for key in total_loss:
            self.writer.add_scalar(f"val/{key}", total_loss[key] / max(total_samples, 1), epoch)

        val_time = time.time() - start_time
        print(f"Epoch {epoch:02d} | Val Time: {val_time:.1f}s | Val Loss: {avg_loss:.5f}")

        if avg_loss < self.prev_val_loss:
            self.prev_val_loss = avg_loss
            best_model_path = os.path.join(self.best_checkpoints_dir, "best_model.pt")
            state = self.model.module.state_dict() if self.multi_gpu and hasattr(self.model, "module") else self.model.state_dict()
            torch.save(state, best_model_path)
            print(f"[*] New best validation checkpoint saved: {best_model_path}")

    def train(self):
        epochs = self.config["train"]["epochs"]
        print(f"\n================ Starting Training ({epochs} epochs) ================")
        for epoch in range(1, epochs + 1):
            self.train_one_epoch(epoch)
            if epoch % self.config.get("val", {}).get("val_every", 1) == 0:
                self.validate(epoch)
            self.scheduler.step()

            if epoch % self.config["train"].get("save_every", 5) == 0:
                ckpt_path = os.path.join(self.checkpoints_dir, f"{epoch}epoch.pt")
                state = self.model.module.state_dict() if self.multi_gpu and hasattr(self.model, "module") else self.model.state_dict()
                torch.save(state, ckpt_path)

        self.writer.close()
        print("\n================ Training Finished ================\n")

    def make_experiments_dirs(self):
        exp_root = self.config.get("experiments", "experiments")
        exp_name = f"{self.config['model']['backbone']}_{self.config['model']['cls_encoding']}"
        run_tag = self.config.get("note", "baseline")
        path = os.path.join(exp_root, f"{exp_name}_{run_tag}")

        self.checkpoints_dir = os.path.join(path, "checkpoints")
        self.best_checkpoints_dir = os.path.join(path, "best_checkpoints")
        self.runs_dir = os.path.join(path, "runs")

        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.best_checkpoints_dir, exist_ok=True)
        os.makedirs(self.runs_dir, exist_ok=True)
        print(f"Experiment directory: {path}")
