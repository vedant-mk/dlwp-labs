"""Why does the baseline ViT lose to persistence at six hours?

Three candidates, each trained briefly on the same data and scored on the same validation
batches, against the persistence loss measured through the identical code path:

  plain      the baseline as it ships
  residual   the network predicts the tendency: forward(x) returns x + net(x)
  no_tp6h    the baseline without precipitation, which carries ~31% of the loss budget
             while being close to unpredictable at six hours

Validation loss in standardised units; lower is better, and persistence is the bar.
"""
import sys
import time
from pathlib import Path

import lightning as L
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import replace

from utils.components import build_network
from utils.config import Config
from utils.lightning_module import ForecastModule

STEPS = 2000
CONFIG = "configs/baseline17.yaml"


class Residual(torch.nn.Module):
    """Wraps a network so it predicts the change rather than the whole next state."""

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x):
        return x + self.network(x)


def build(name: str) -> ForecastModule:
    cfg = yaml.safe_load(open(CONFIG))
    cfg["trainer"]["max_steps"] = STEPS
    cfg["trainer"]["schedulers"] = [{"type": "linear", "steps": 50, "start_factor": 0.01},
                                    {"type": "cosine", "steps": STEPS - 50}]
    if name == "no_tp6h":
        cfg["dataset"]["variables"] = [v for v in cfg["dataset"]["variables"] if v != "TP6h"]

    config = Config.from_dict(cfg)
    last = "2015-12"
    module = ForecastModule(replace(config, trainer=replace(config.trainer,
                                                            val_time_slice={"start": last, "stop": last})))
    if name == "residual":
        module.model = Residual(module.model)
    return module


def validation_loss(module: ForecastModule) -> float:
    """The mean single-step validation loss, in the module's own units."""
    module.eval()
    losses = []
    with torch.no_grad():
        for batch in module.val_dataloader():
            states, _ = batch
            states = states.to(module.device)
            prediction = module.model(states[:, :, 0])
            losses.append(module.step_loss(states[:, :, 1], prediction).item())
    module.train()
    return sum(losses) / len(losses)


def persistence_loss(module: ForecastModule) -> float:
    """The same measurement with the identity as the model: the bar to beat."""
    losses = []
    with torch.no_grad():
        for batch in module.val_dataloader():
            states, _ = batch
            states = states.to(module.device)
            losses.append(module.step_loss(states[:, :, 1], states[:, :, 0]).item())
    return sum(losses) / len(losses)


def main() -> None:
    print(f"{STEPS} training steps each, single-step validation loss in standardised units\n")
    print(f"{'variant':12} {'persistence':>12} {'model':>10} {'ratio':>8}   verdict")
    print("-" * 62)
    for name in ("plain", "residual"):
        torch.manual_seed(0)
        module = build(name)
        bar = persistence_loss(module)
        trainer = L.Trainer(max_steps=STEPS, accelerator="auto", logger=False,
                            enable_checkpointing=False, enable_model_summary=False,
                            enable_progress_bar=False, num_sanity_val_steps=0,
                            limit_val_batches=0, val_check_interval=None,
                            check_val_every_n_epoch=None)
        started = time.time()
        trainer.fit(module)
        got = validation_loss(module)
        verdict = "BEATS persistence" if got < bar else "loses to persistence"
        print(f"{name:12} {bar:12.4f} {got:10.4f} {got / bar:8.2f}   {verdict}  "
              f"({time.time() - started:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
