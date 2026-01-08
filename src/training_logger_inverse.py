"""
Training logger for Inverse PINN.

Extends the base TrainingLogger with correct component names for inverse problem:
    - pde: PDE residual
    - flux_bc: Surface flux BC
    - wt_head: Water table head BC
    - data: Data matching
    - ic_h: Initial condition h
    - ic_zb: Initial condition z_b
    - zb_smooth: Smoothness regularization
    - zb_curv: Curvature regularization
"""

import csv
import json
import os
from datetime import datetime
import time


class InverseTrainingLogger:
    """Logger for inverse PINN training."""

    def __init__(self, log_dir=None, experiment_name=None, hyperparams=None):
        self.losses = []
        self.comps = {
            "pde": [],
            "flux_bc": [],
            "wt_head": [],
            "data": [],
            "ic_h": [],
            "ic_zb": [],
            "zb_smooth": [],
            "zb_curv": [],
        }
        self.grads = {
            "pde": [],
            "flux_bc": [],
            "wt_head": [],
            "data": [],
            "ic_h": [],
            "ic_zb": [],
            "zb_smooth": [],
            "zb_curv": [],
            "total": [],
        }

        self.sample_losses = []
        self.sample_comps = {
            "pde": [],
            "flux_bc": [],
            "wt_head": [],
            "data": [],
            "ic_h": [],
            "ic_zb": [],
            "zb_smooth": [],
            "zb_curv": [],
        }
        self.sample_epochs = []

        self.log_dir = log_dir
        self.experiment_name = experiment_name or f"inverse_pinn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.hyperparams = hyperparams or {}
        self.start_time = time.time()

        if self.log_dir:
            self._setup_file_logging()

        self.weights_history = {k: [] for k in self.comps.keys()}
        self.training_metrics = {
            "epoch_times": [],
            "learning_rates": [],
            "gradient_norms": [],
        }

        self.zb_history = {
            "epochs": [],
            "zb_values": [],  # List of arrays
        }

    def _to_numpy(self, data_list):
        """Convert list of mixed tensors/floats to numpy array."""
        import torch
        import numpy as np

        result = []
        for x in data_list:
            if isinstance(x, torch.Tensor):
                result.append(x.detach().cpu().item())
            else:
                result.append(x)
        return np.array(result)

    def get_losses_numpy(self):
        """Get losses as numpy array."""
        return self._to_numpy(self.losses)

    def get_comps_numpy(self):
        """Get loss components as dict of numpy arrays."""
        return {k: self._to_numpy(v) for k, v in self.comps.items()}

    def get_grads_numpy(self):
        """Get gradients as dict of numpy arrays."""
        return {k: self._to_numpy(v) for k, v in self.grads.items()}

    def _setup_file_logging(self):
        """Setup file logging directory and files."""
        self.exp_dir = os.path.join(self.log_dir, self.experiment_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        self.csv_file = os.path.join(self.exp_dir, "training_metrics.csv")
        self.csv_fieldnames = [
            "epoch", "timestamp", "elapsed_time",
            "total_loss",
            "pde_loss", "flux_bc_loss", "wt_head_loss", "data_loss",
            "ic_h_loss", "ic_zb_loss", "zb_smooth_loss", "zb_curv_loss",
            "pde_grad", "flux_bc_grad", "wt_head_grad", "data_grad",
            "ic_h_grad", "ic_zb_grad", "total_grad",
            "pde_weight", "flux_bc_weight", "wt_head_weight", "data_weight",
            "ic_h_weight", "ic_zb_weight", "zb_smooth_weight", "zb_curv_weight",
            "learning_rate", "epoch_time"
        ]

        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writeheader()

        self.metadata_file = os.path.join(self.exp_dir, "experiment_metadata.json")
        self._save_metadata()

        print(f"Logging initialized: {self.exp_dir}")

    def _save_metadata(self):
        """Save experiment metadata to JSON."""
        metadata = {
            "experiment_name": self.experiment_name,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "hyperparameters": self.hyperparams,
            "csv_file": "training_metrics.csv",
            "description": "Inverse PINN training - find water table from surface observations"
        }

        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def record_epoch_start(self):
        """Record start time of epoch."""
        self.epoch_start_time = time.time()

    def record_losses(self, total_loss, loss_dict):
        """Record loss values."""
        import torch

        if isinstance(total_loss, torch.Tensor):
            self.losses.append(total_loss.detach())
        else:
            self.losses.append(total_loss)

        for key in self.comps:
            if key in loss_dict:
                val = loss_dict[key]
                if isinstance(val, torch.Tensor):
                    self.comps[key].append(val.detach())
                else:
                    self.comps[key].append(val)

    def record_gradients(self, grad_dict):
        """Record gradient norms."""
        import torch

        for key in self.grads:
            if key in grad_dict:
                val = grad_dict[key]
                if isinstance(val, torch.Tensor):
                    self.grads[key].append(val.detach())
                else:
                    self.grads[key].append(val)

    def record_sample_losses(self, epoch, loss_dict, weights):
        """Record sample losses over full dataset."""
        self.sample_epochs.append(epoch)

        total_loss = sum(weights.get(key, 1.0) * loss_dict.get(key, 0.0) for key in loss_dict)
        self.sample_losses.append(total_loss)

        for key in self.sample_comps:
            if key in loss_dict and key in weights:
                weighted_loss = weights[key] * loss_dict[key]
                self.sample_comps[key].append(weighted_loss)

    def record_weights(self, weights):
        """Record current weight values."""
        for key in self.weights_history:
            if key in weights:
                self.weights_history[key].append(weights[key])

    def record_zb_evolution(self, epoch, t_values, zb_values):
        """Record water table depth evolution."""
        import numpy as np

        self.zb_history["epochs"].append(epoch)

        if hasattr(zb_values, 'cpu'):
            zb_np = zb_values.detach().cpu().numpy()
        else:
            zb_np = np.array(zb_values)

        self.zb_history["zb_values"].append(zb_np)

    def record_training_metrics(self, learning_rate=None):
        """Record additional training metrics."""
        if hasattr(self, 'epoch_start_time'):
            epoch_time = time.time() - self.epoch_start_time
            self.training_metrics["epoch_times"].append(epoch_time)
        else:
            self.training_metrics["epoch_times"].append(None)

        self.training_metrics["learning_rates"].append(learning_rate)

    def log_epoch(self, epoch, total_loss, loss_dict, grad_dict, weights, learning_rate=None):
        """
        Log complete epoch information to CSV file.
        GPU-OPTIMIZED: Batches all tensor conversions together.
        """
        if not self.log_dir:
            return

        import torch

        # Helper to convert tensor to float
        def to_float(x):
            return x.detach().cpu().item() if isinstance(x, torch.Tensor) else x

        current_time = time.time()
        elapsed_time = current_time - self.start_time
        epoch_time = current_time - self.epoch_start_time if hasattr(self, 'epoch_start_time') else None

        total_loss_val = to_float(total_loss)
        loss_vals = {k: to_float(v) for k, v in loss_dict.items()}
        grad_vals = {k: to_float(v) for k, v in grad_dict.items()}

        row = {
            "epoch": epoch,
            "timestamp": datetime.fromtimestamp(current_time).isoformat(),
            "elapsed_time": elapsed_time,
            "total_loss": total_loss_val,
            "pde_loss": loss_vals.get("pde", 0),
            "flux_bc_loss": loss_vals.get("flux_bc", 0),
            "wt_head_loss": loss_vals.get("wt_head", 0),
            "data_loss": loss_vals.get("data", 0),
            "ic_h_loss": loss_vals.get("ic_h", 0),
            "ic_zb_loss": loss_vals.get("ic_zb", 0),
            "zb_smooth_loss": loss_vals.get("zb_smooth", 0),
            "zb_curv_loss": loss_vals.get("zb_curv", 0),
            "pde_grad": grad_vals.get("pde", 0),
            "flux_bc_grad": grad_vals.get("flux_bc", 0),
            "wt_head_grad": grad_vals.get("wt_head", 0),
            "data_grad": grad_vals.get("data", 0),
            "ic_h_grad": grad_vals.get("ic_h", 0),
            "ic_zb_grad": grad_vals.get("ic_zb", 0),
            "total_grad": grad_vals.get("total", 0),
            "pde_weight": weights.get("pde", 0),
            "flux_bc_weight": weights.get("flux_bc", 0),
            "wt_head_weight": weights.get("wt_head", 0),
            "data_weight": weights.get("data", 0),
            "ic_h_weight": weights.get("ic_h", 0),
            "ic_zb_weight": weights.get("ic_zb", 0),
            "zb_smooth_weight": weights.get("zb_smooth", 0),
            "zb_curv_weight": weights.get("zb_curv", 0),
            "learning_rate": learning_rate,
            "epoch_time": epoch_time
        }

        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writerow(row)

    def print_progress(self, epoch, n_epochs, total_loss, loss_dict, grad_dict, weights):
        """Print training progress."""
        import torch

        def to_float(x):
            return x.detach().cpu().item() if isinstance(x, torch.Tensor) else x

        total_loss_val = to_float(total_loss)
        loss_vals = {k: to_float(v) for k, v in loss_dict.items()}
        grad_vals = {k: to_float(v) for k, v in grad_dict.items()}

        print(f"\n[{epoch+1:4d}/{n_epochs}]")
        print(f"  Total loss: {total_loss_val:.3e}")
        print(f"  Components:")
        print(f"    PDE={loss_vals.get('pde', 0):.3e}, FluxBC={loss_vals.get('flux_bc', 0):.3e}")
        print(f"    WTHead={loss_vals.get('wt_head', 0):.3e}, Data={loss_vals.get('data', 0):.3e}")
        print(f"    IC(h)={loss_vals.get('ic_h', 0):.3e}, IC(zb)={loss_vals.get('ic_zb', 0):.3e}")
        print(f"    Smooth={loss_vals.get('zb_smooth', 0):.3e}, Curv={loss_vals.get('zb_curv', 0):.3e}")

        print(f"  Gradients:")
        print(f"    PDE={grad_vals.get('pde', 0):.3e}, FluxBC={grad_vals.get('flux_bc', 0):.3e}")
        print(f"    Data={grad_vals.get('data', 0):.3e}, Total={grad_vals.get('total', 0):.3e}")

    def print_final_summary(self, total_grad_norm, weights):
        """Print final training summary."""
        import torch

        def to_float(x):
            return x.detach().cpu().item() if isinstance(x, torch.Tensor) else x

        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Final gradient norm: {to_float(total_grad_norm):.3e}")
        print(f"Final weights:")
        for key in weights:
            print(f"  {key}: {weights[key]:.3e}")

    def save_final_summary(self, model=None, final_metrics=None):
        """Save final training summary to JSON."""
        if not self.log_dir:
            return

        end_time = time.time()
        total_training_time = end_time - self.start_time

        # Helper to safely get last value
        def safe_last(lst):
            if not lst:
                return None
            val = lst[-1]
            if hasattr(val, 'item'):
                return val.item()
            return val

        summary = {
            "experiment_name": self.experiment_name,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
            "total_training_time_seconds": total_training_time,
            "total_epochs": len(self.losses),
            "final_losses": {
                "total": self._to_numpy([self.losses[-1]])[0] if self.losses else None,
                **{k: self._to_numpy([v[-1]])[0] if v else None for k, v in self.comps.items()}
            },
            "final_gradients": {
                k: self._to_numpy([v[-1]])[0] if v else None for k, v in self.grads.items()
            },
            "final_weights": {
                k: safe_last(v) for k, v in self.weights_history.items()
            },
            "training_statistics": {
                "avg_epoch_time": sum(filter(None, self.training_metrics["epoch_times"])) / len([x for x in self.training_metrics["epoch_times"] if x]) if self.training_metrics["epoch_times"] else None,
                "min_total_loss": min(self._to_numpy(self.losses)) if self.losses else None,
                "convergence_info": {
                    "epochs_to_min_loss": int(self._to_numpy(self.losses).argmin()) + 1 if self.losses else None,
                }
            }
        }

        if final_metrics:
            summary["final_metrics"] = final_metrics

        summary_file = os.path.join(self.exp_dir, "training_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Training summary saved: {summary_file}")
