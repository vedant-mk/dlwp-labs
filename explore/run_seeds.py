"""Repeat every model with seeds 1 and 2, after the seed-0 batch has finished.

Waits for the running seed-0 batch to exit so the GPU is not shared, then runs each config/seed
through experiment.py. After each seed run it keeps what the report needs - scores.csv, the loss
curves, run.json, and a small forecast slice for the spectrum figure - and deletes the full
forecast store (2.4 GB) and the checkpoint, so six runs fit on the disk.

Order: a complete seed-1 set first, then seed 2, so a partial night still yields full comparisons.

    python explore/run_seeds.py --wait-pid <pid of the seed-0 batch>
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ["baseline17", "pyramid", "multiscale"]
SEEDS = [1, 2]
LOG = ROOT / "explore" / "seed_runs.log"

# the slice explore/extra_figures.py reads: its variables, leads and initialisation stride
SLICE_VARIABLES = ["Z500", "T850", "Q850", "TP6h"]
SLICE_LEADS = [24, 120]
SLICE_STRIDE = 4


def say(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def keep_slice_and_prune(run_dir: Path) -> None:
    store = run_dir / "forecasts.zarr"
    if not store.exists():
        say(f"  no forecasts in {run_dir}; nothing to prune")
        return
    forecasts = xr.open_zarr(store)[SLICE_VARIABLES]
    part = forecasts.isel(time=slice(0, None, SLICE_STRIDE)).sel(
        prediction_timedelta=[np.timedelta64(h, "h") for h in SLICE_LEADS]).load()
    for variable in part.variables.values():
        variable.encoding.clear()
    part.to_zarr(run_dir / "forecasts_slice.zarr", mode="w")
    shutil.rmtree(store)
    (run_dir / "model.ckpt").unlink(missing_ok=True)
    say(f"  kept forecasts_slice.zarr, removed forecasts.zarr and model.ckpt from {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    args = parser.parse_args()

    say(f"waiting for the seed-0 batch (pid {args.wait_pid}) to finish")
    while alive(args.wait_pid):
        time.sleep(60)
    say("seed-0 batch finished")

    batch_log = (ROOT / "explore" / "rollout_runs.log").read_text(errors="ignore")
    multiscale_ok = "multiscale seed 0:" in batch_log
    if not multiscale_ok:
        say("WARNING: multiscale seed 0 did not complete; its seed runs are skipped")

    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    for seed in SEEDS:
        for config in CONFIGS:
            if config == "multiscale" and not multiscale_ok:
                continue
            run_dir = ROOT / "runs" / config / f"seed{seed}"
            if (run_dir / "run.json").exists():
                say(f"{config} seed {seed} already done, skipping")
                continue
            free = shutil.disk_usage(ROOT).free / 1e9
            if free < 5:
                say(f"STOPPING: only {free:.1f} GB free, a run needs ~2.5 GB of headroom")
                return
            say(f"starting {config} seed {seed} ({free:.1f} GB free)")
            started = time.time()
            with open(ROOT / "explore" / f"seed_{config}_{seed}.log", "w") as out:
                code = subprocess.run([sys.executable, "experiment.py", "--config", f"configs/{config}.yaml",
                                       "--seed", str(seed)], cwd=ROOT, env=env, stdout=out,
                                      stderr=subprocess.STDOUT).returncode
            if code != 0:
                say(f"FAILED: {config} seed {seed} exited {code}; see explore/seed_{config}_{seed}.log")
                continue
            say(f"finished {config} seed {seed} in {(time.time() - started) / 60:.0f} min")
            keep_slice_and_prune(run_dir)
    say("all seed runs done")


if __name__ == "__main__":
    main()
