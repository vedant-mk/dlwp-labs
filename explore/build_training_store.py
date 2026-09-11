"""Build the training store: ERA5 2009-2015 at 5.625 degrees, from WeatherBench 2.

Written in a single pass. An earlier version appended one year at a time, which failed:
a year is 1460 steps against a 100-step zarr chunk, so each append had to rewrite a
partial chunk, and a failure mid-append left T2M at two years and the rest at one.
One write has no intermediate state to desynchronise; dask streams it, so memory is flat.

    python explore/build_training_store.py
"""
import shutil
import sys
import time
from pathlib import Path

import xarray as xr
from dask.diagnostics import ProgressBar

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # the repo root, for the modules below

from prepare_eval_data import ANON, STORES, select

VARIABLES = ["T2M", "Z850", "Z500", "Z250", "T850", "T500", "T250",
             "Q850", "Q500", "Q250", "U850", "U500", "U250", "V850", "V500", "V250", "TP6h"]
YEARS = ("2009", "2015")           # the brief caps training at 2015
OUT = Path("data/era5_5p6.zarr")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        print(f"removing the incomplete {OUT}", flush=True)
        shutil.rmtree(OUT)

    store = xr.open_zarr(STORES[(64, 32)][0], storage_options=ANON)
    data = xr.Dataset({v: select(store, v).sel(time=slice(*YEARS)) for v in VARIABLES})
    data = data.transpose("time", "latitude", "longitude").chunk({"time": 100})
    for variable in data.variables.values():          # cloud codecs do not carry over
        variable.encoding.clear()

    print(f"writing {data.sizes['time']} states of {len(VARIABLES)} variables "
          f"({data.nbytes / 1e9:.2f} GB) to {OUT}", flush=True)
    started = time.time()
    with ProgressBar(dt=20.0):                        # a line every 20 s, not a spinner
        data.to_zarr(OUT, mode="w")

    final = xr.open_zarr(OUT)
    assert final.sizes["time"] == data.sizes["time"], "the store is short"
    assert all(final[v].sizes["time"] == final.sizes["time"] for v in VARIABLES), \
        "variables disagree on the length of time"
    print(f"\ndone in {time.time() - started:.0f} s: {final.sizes['time']} states, "
          f"{len(final.data_vars)} variables, "
          f"{str(final.time.values[0])[:10]} to {str(final.time.values[-1])[:10]}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
