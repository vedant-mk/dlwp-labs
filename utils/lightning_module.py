from dataclasses import replace
from functools import partial

import einops
import lightning as L
import torch
from torch.utils.data import DataLoader, Subset

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from utils import loss_fn
from utils.components import build_network
from utils.config import Config, build_world
from utils.dataset import WeatherDataset


class ForecastModule(L.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.to_dict())

        assert self.cfg.val_time_slice is not None, "the validation years are named in the configuration"
        self.train_dataset = WeatherDataset(self.data_cfg)
        self.val_dataset = WeatherDataset(replace(self.data_cfg, time_slice=self.cfg.val_time_slice,
                                               sequence_length=self.cfg.rollout_steps + 1))
        if self.config.world is None:
            grid = self.train_dataset.dataset.sizes
            self.config = replace(self.config, world=build_world(
                len(self.data_cfg.variables), (grid["latitude"], grid["longitude"]), self.model_cfg.patch_size))
        self.model = build_network(self.model_cfg, self.world)
        self.loss_fn = partial(getattr(loss_fn, f"f_{self.objective.name}"), **self.objective.kwargs)

    @property
    def cfg(self):
        return self.config.trainer

    @property
    def data_cfg(self):
        return self.config.dataset

    @property
    def model_cfg(self):
        return self.config.network

    @property
    def world(self):
        return self.config.world

    @property
    def objective(self):
        return self.config.objective

    @property
    def per_variable_weights(self) -> torch.FloatTensor:
        weights = self.objective.weights or {}
        w = torch.as_tensor([weights.get(var, 1.) for var in self.data_cfg.variables], dtype=torch.float32, device=self.device)
        return einops.repeat(w, f"(v vv) -> {self.world.field_pattern}",
                             **self.world.token_sizes, **self.world.patch_sizes)

    @property
    def area_weights(self) -> torch.FloatTensor:
        latitude = torch.as_tensor(self.train_dataset.dataset.latitude.values, dtype=torch.float32, device=self.device)
        w = torch.cos(torch.deg2rad(latitude))
        return einops.repeat(w / w.mean(), f"(h hh) -> {self.world.field_pattern}",
                             **self.world.token_sizes, **self.world.patch_sizes)

    @property
    def residual(self) -> bool:
        # persistence already returns its input; adding it again would double the state
        return self.model_cfg.residual and self.model_cfg.name != "persistence"

    def forward(self, state: torch.FloatTensor) -> torch.FloatTensor:
        prediction = self.model(state)
        return state + prediction if self.residual else prediction

    def rollout(self, state: torch.FloatTensor, steps: int) -> torch.FloatTensor:
        out = []
        for _ in range(steps):
            state = self.forward(state)
            out.append(state)
        return torch.stack(out, dim=2)

    def step_loss(self, observation: torch.FloatTensor, prediction: torch.FloatTensor) -> torch.FloatTensor:
        score = self.loss_fn(observation, prediction)
        return score.mul(self.per_variable_weights).mul(self.area_weights).mean()

    @staticmethod
    def validation_loss(observation: torch.FloatTensor, prediction: torch.FloatTensor) -> torch.FloatTensor:
        # plain MSE whatever the training objective, so val/loss_step* is one quantity across runs
        return loss_fn.f_mse(observation, prediction).mean()

    def training_step(self, batch, batch_idx):
        states, _ = batch
        steps = 1 if self.global_step < self.cfg.pre_steps else self.cfg.train_rollout_steps
        prediction = self.rollout(states[:, :, 0], steps)
        loss = sum(self.step_loss(states[:, :, k + 1], prediction[:, :, k]) for k in range(steps)) / steps
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        states, _ = batch
        prediction = self.rollout(states[:, :, 0], self.cfg.rollout_steps)
        for k in range(self.cfg.rollout_steps):
            self.log(f"val/loss_step{k + 1}", self.validation_loss(states[:, :, k + 1], prediction[:, :, k]),
                     prog_bar=(k == 0))

    def predict_step(self, batch, batch_idx):
        states, _ = batch
        return self.rollout(states[:, :, 0], self.cfg.predict_steps).cpu()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
                                      betas=tuple(self.cfg.betas))
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": self.create_scheduler(optimizer), "interval": "step"}}

    def create_scheduler(self, optimizer):
        schedulers, milestones, total = [], [], 0
        for sch_cfg in self.cfg.schedulers:
            typ, steps = sch_cfg["type"].lower(), sch_cfg["steps"]
            total += steps
            milestones.append(total)
            if typ == "linear":
                sched = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=sch_cfg.get("start_factor", 1.0),
                    end_factor=sch_cfg.get("end_factor", 1.0), total_iters=steps)
            elif typ == "constant":
                sched = torch.optim.lr_scheduler.ConstantLR(
                    optimizer, factor=sch_cfg.get("factor", 1.0), total_iters=steps)
            elif typ == "cosine":
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=steps, eta_min=sch_cfg.get("eta_min", 0.0))
            else:
                raise ValueError(f"Unknown scheduler type: {typ}")
            schedulers.append(sched)
        return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=schedulers, milestones=milestones[:-1])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True,
                          num_workers=self.cfg.num_workers, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.subset(self.val_dataset, self.cfg.rollout_steps, self.cfg.val_stride),
                          batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.subset(self.val_dataset, self.cfg.predict_steps, self.cfg.predict_stride),
                          batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers)

    @staticmethod
    def subset(dataset: WeatherDataset, steps: int, stride: int) -> Subset:
        stop = min(dataset.dataset.sizes["time"] - steps, len(dataset))
        return Subset(dataset, range(0, stop, stride))


def forecasts_to_xarray(module, predictions, path=None):
    """The list Trainer.predict returns as one Dataset in the forecast schema of lab 1:
    (time, prediction_timedelta, latitude, longitude), physical units."""
    dataset, cfg = module.val_dataset, module.config.trainer
    times = dataset.dataset.time.values
    inits = times[range(0, len(times) - cfg.predict_steps, cfg.predict_stride)]
    leads = np.arange(1, cfg.predict_steps + 1) * (times[1] - times[0])
    batches, start = [], 0
    for prediction in predictions:                                   # (b, v, steps, H, W) each
        b = prediction.shape[0]
        x = einops.rearrange(prediction, "b v t h w -> v b t h w")
        batches.append(dataset.to_xarray(x, time=inits[start: start + b],
                                         prediction_timedelta=leads).astype("float32"))
        start += b
    ds = xr.concat(batches, dim="time")
    if path is not None:
        ds.chunk({"time": 16}).to_zarr(path, mode="w")
        ds = xr.open_zarr(path)
    return ds


def plot_metrics(log_dir, ax=None, label=None):
    """Draw the training and validation loss of one run from its metrics.csv onto the current axes."""
    df = pd.read_csv(Path(log_dir) / "metrics.csv")
    ax = ax or plt.gca()

    train = df.dropna(subset=["train/loss"]) if "train/loss" in df else df.iloc[:0]
    if len(train):
        line, = ax.plot(train["step"], train["train/loss"].rolling(10, min_periods=1).mean(),
                        lw=1.8, label=f"{label} train" if label else "train")
        ax.plot(train["step"], train["train/loss"], lw=0.7, alpha=0.25, color=line.get_color())

    if "val/loss_step1" in df:
        val = df.dropna(subset=["val/loss_step1"])
        if len(val):
            ax.plot(val["step"], val["val/loss_step1"], "o--", ms=4, lw=1.0,
                    label=f"{label} val step1" if label else "val step1")

    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (standardised units)")
    ax.grid(alpha=0.3, which="both")
    return ax


class LossCurve(L.Callback):
    """Writes the loss curve of a run when training ends."""

    def on_train_end(self, trainer, module):
        if trainer.logger is None:
            return
        trainer.logger.save()
        plot_metrics(trainer.logger.log_dir)
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(trainer.logger.log_dir) / "loss_curves.png")
        plt.close()
