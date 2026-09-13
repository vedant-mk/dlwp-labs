"""Justify the two protocol decisions on the validation year 2016 alone.

Both decisions were first motivated by numbers that should not have driven them: the residual
check validated on December 2015 (inside the training years), and roll-out training was adopted
after seeing the 5-day blow-up on the 2017-2019 test years. This script repeats both arguments
using only the training years and 2016, so the report can justify them without the test set.

  Check 1 (residual)   train the single-step baseline 2000 steps with and without the residual;
                       one-step validation MSE on 2016 against persistence in the same units.
  Check 2 (roll-out)   roll out 5 days on 2016 from (a) that single-step residual model and
                       (b) the roll-out-trained baseline (runs/baseline17/seed0/model.ckpt);
                       latitude-weighted RMSE of Z500 and T850 against persistence and climatology.

    python explore/validate_on_2016.py
"""
import sys
import time
from dataclasses import replace
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.config import Config
from utils.dataset import WeatherDataset
from utils.lightning_module import ForecastModule, forecasts_to_xarray
from utils.metrics import climatology_at, rmse, truth_at

xr.set_options(use_bottleneck=False)
torch.set_float32_matmul_precision("medium")

CONFIG = ROOT / "configs" / "baseline17.yaml"
EVAL, CLIM = ROOT / "data" / "era5_eval_5p6.zarr", ROOT / "data" / "clim_wb2_5p6.zarr"
OUT = ROOT / "explore" / "figures"
STEPS = 2000
LEADS = [6, 24, 72, 120]


def config(residual: bool, single_step: bool) -> Config:
    cfg = yaml.safe_load(open(CONFIG))
    cfg["network"]["residual"] = residual
    if single_step:   # the 2000-step single-step protocol the decisions were first taken on
        cfg["dataset"]["sequence_length"] = 2
        cfg["trainer"].update(max_steps=STEPS, pre_steps=0, train_rollout_steps=1,
                              schedulers=[{"type": "linear", "steps": 100, "start_factor": 0.01},
                                          {"type": "cosine", "steps": STEPS - 100}])
    c = Config.from_dict(cfg)
    return replace(c, trainer=replace(c.trainer, val_time_slice={"start": "2015-12", "stop": "2015-12"}))


def use_2016(module: ForecastModule) -> ForecastModule:
    module.val_dataset = WeatherDataset(replace(module.config.dataset, path=str(EVAL),
                                                time_slice={"start": "2016", "stop": "2016"},
                                                sequence_length=module.config.trainer.rollout_steps + 1))
    return module


def one_step_mse(module: ForecastModule, identity: bool = False) -> float:
    module.eval()
    losses = []
    with torch.no_grad():
        for states, _ in module.val_dataloader():
            states = states.to(module.device)
            prediction = states[:, :, 0] if identity else module(states[:, :, 0])
            losses.append(ForecastModule.validation_loss(states[:, :, 1], prediction).item())
    return float(np.mean(losses))


def trainer() -> L.Trainer:
    return L.Trainer(max_steps=STEPS, accelerator="auto", logger=False, enable_checkpointing=False,
                     enable_model_summary=False, enable_progress_bar=False, num_sanity_val_steps=0,
                     limit_val_batches=0, val_check_interval=None, check_val_every_n_epoch=None)


def five_day_scores(module: ForecastModule, name: str) -> pd.DataFrame:
    module = use_2016(module)
    forecasts = forecasts_to_xarray(module, trainer().predict(module))[["Z500", "T850"]].load()
    era5 = xr.open_zarr(EVAL)
    clim = xr.open_zarr(CLIM)[["Z500", "T850"]].load()
    truth = truth_at(era5, forecasts).load()
    persistence = era5[["Z500", "T850"]].sel(time=forecasts.time).broadcast_like(forecasts).load()
    rows = []
    for label, field in ((name, forecasts), ("persistence", persistence), ("climatology", climatology_at(clim, forecasts))):
        score = rmse(field, truth)
        for var in ("Z500", "T850"):
            for lead in LEADS:
                rows.append(dict(model=label, variable=var, lead_hours=lead,
                                 rmse=float(score[var].sel(prediction_timedelta=np.timedelta64(lead, "h")))))
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = []

    # ----------------------------------------------------------- check 1: residual, on 2016
    print(f"check 1: {STEPS}-step single-step baseline with and without the residual, validated on 2016\n")
    models = {}
    rows = []
    for residual in (False, True):
        L.seed_everything(0, workers=True)
        module = use_2016(ForecastModule(config(residual=residual, single_step=True)))
        started = time.time()
        trainer().fit(module)
        models[residual] = module
        model_loss, bar = one_step_mse(module), one_step_mse(module, identity=True)
        rows.append(dict(variant="residual" if residual else "plain", persistence=bar, model=model_loss,
                         ratio=model_loss / bar, train_s=round(time.time() - started)))
    del models[False]   # only the residual model is needed for check 2; free its training tensor
    check1 = pd.DataFrame(rows)
    print(check1.round(4).to_string(index=False))
    check1.to_csv(OUT / "validation2016_residual.csv", index=False)
    report.append("### Check 1: residual (one-step MSE on 2016, standardised units)\n\n" + "```\n" + check1.round(4).to_string(index=False) + "\n```")

    # ----------------------------------------------------------- check 2: roll-out, on 2016
    print("\ncheck 2: five-day roll-out on 2016, single-step vs roll-out-trained baseline\n")
    single = five_day_scores(models[True], "single-step (2000 steps)")
    ckpt = ROOT / "runs" / "baseline17" / "seed0" / "model.ckpt"
    trained = ForecastModule.load_from_checkpoint(ckpt, config=config(residual=True, single_step=False),
                                                  map_location="cpu")
    rollout = five_day_scores(trained, "roll-out-trained (6000 steps)")
    check2 = pd.concat([single, rollout[~rollout.model.isin(["persistence", "climatology"])]])
    table = check2.pivot_table(index=["variable", "model"], columns="lead_hours", values="rmse")
    print(table.round(2).to_string())
    check2.to_csv(OUT / "validation2016_rollout.csv", index=False)
    report.append("### Check 2: five-day roll-out on 2016 (latitude-weighted RMSE)\n\n" + "```\n" + table.round(2).to_string() + "\n```")

    (OUT / "validation2016.md").write_text("# Protocol decisions checked on the 2016 validation year\n\n"
                                         + "\n\n".join(report) + "\n")
    print(f"\nwrote {OUT}/validation2016.md")


def equal_steps_control() -> None:
    """Check 2 compared 2000 single-step steps with 6000 roll-out-trained steps, mixing training length
    with roll-out training. This trains single-step for the same 6000 steps and rolls it out on 2016."""
    global STEPS
    STEPS = 6000
    print("check 3: single-step baseline trained 6000 steps (same length as roll-out training), on 2016\n")
    L.seed_everything(0, workers=True)
    module = ForecastModule(config(residual=True, single_step=True))
    started = time.time()
    trainer().fit(module)
    scores = five_day_scores(module, "single-step (6000 steps)")
    scores = scores[~scores.model.isin(["persistence", "climatology"])]
    table = scores.pivot_table(index=["variable", "model"], columns="lead_hours", values="rmse")
    print(table.round(2).to_string(), f"\n(trained in {time.time() - started:.0f} s)")
    scores.to_csv(OUT / "validation2016_rollout_equal_steps.csv", index=False)
    with open(OUT / "validation2016.md", "a") as f:
        f.write("\n### Check 3: equal-length control (single-step, 6000 steps) on 2016\n\n```\n"
                + table.round(2).to_string() + "\n```\n")


if __name__ == "__main__":
    equal_steps_control() if "--equal-steps" in sys.argv else main()
