"""Step 3 — the variables, and the spatial scales they occupy (zonal power spectra)."""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

ds = xr.open_dataset("cache/data.nc").sel(time=slice("2015-01-01", "2015-12-31T18"))
lat = ds.latitude.values
w = np.cos(np.deg2rad(lat))
w = w / w.mean()

META = {
    "Z500": ("500 hPa", "~5.5 km", "m^2/s^2", "geopotential - the steering flow; troughs and ridges"),
    "T850": ("850 hPa", "~1.5 km", "K", "temperature above the boundary layer; air-mass marker"),
    "Q700": ("700 hPa", "~3.0 km", "kg/kg", "specific humidity; moisture available to storms"),
    "TP6h": ("surface", "0 km", "m/6h", "total precipitation accumulated over 6 hours"),
    "T2M": ("surface", "2 m", "K", "screen-level temperature; strong daily cycle"),
    "U10M": ("surface", "10 m", "m/s", "eastward surface wind"),
    "V10M": ("surface", "10 m", "m/s", "northward surface wind"),
    "U250": ("250 hPa", "~10.4 km", "m/s", "eastward wind at jet-stream level"),
    "V250": ("250 hPa", "~10.4 km", "m/s", "northward wind at jet-stream level"),
}

print(f"{'var':6} {'level':8} {'altitude':10} {'unit':8} "
      f"{'mean':>12} {'std':>12}   meaning")
print("-" * 118)
for v in ds.data_vars:
    lev, alt, unit, mean_of = META[v]
    x = ds[v]
    m = float((x * xr.DataArray(w, dims="latitude")).mean())
    s = float(x.std())
    print(f"{v:6} {lev:8} {alt:10} {unit:8} {m:12.4g} {s:12.4g}   {mean_of}")

# ------------------------------------------------------------ zonal spectra
SUB = ds.isel(time=slice(0, None, 15))          # ~97 samples across 2015
print(f"\nspectra averaged over {SUB.sizes['time']} times spanning 2015")

nlon = ds.sizes["longitude"]
k = np.fft.rfftfreq(nlon, d=1.0 / nlon)          # zonal wavenumber 0..32

spectra = {}
for v in ds.data_vars:
    x = SUB[v].values.astype(np.float64)          # (t, lat, lon)
    x = x - x.mean(axis=-1, keepdims=True)        # remove zonal mean -> drop k=0
    P = np.abs(np.fft.rfft(x, axis=-1)) ** 2      # (t, lat, k)
    P = (P * w[None, :, None]).mean(axis=(0, 1))  # area-weighted, time-averaged
    spectra[v] = P / P.sum()                      # normalise: fraction of variance

# fraction of variance at large (k<=3), synoptic (4-8), small (>=9) scales
print(f"\n{'var':6} {'large k<=3':>11} {'synoptic 4-8':>13} {'small k>=9':>11}   dominant")
print("-" * 62)
bands = {}
for v, P in spectra.items():
    large = P[1:4].sum()
    syn = P[4:9].sum()
    small = P[9:].sum()
    bands[v] = (large, syn, small)
    dom = ["large", "synoptic", "small"][int(np.argmax([large, syn, small]))]
    print(f"{v:6} {large:11.3f} {syn:13.3f} {small:11.3f}   {dom}")

# ------------------------------------------------------------------ figure
SNAP = "2015-01-15T00:00"
show = ["Z500", "T850", "Q700", "TP6h"]
scale = {"Z500": (1 / 9.81, "geopotential height (m)"),
         "T850": (1.0, "K"), "Q700": (1e3, "g/kg"), "TP6h": (1e3, "mm / 6 h")}

fig = plt.figure(figsize=(15, 9))
for i, v in enumerate(show):
    ax = fig.add_subplot(3, 2, i + 1) if i < 2 else fig.add_subplot(3, 2, i + 1)
    f, unit = scale[v]
    field = ds[v].sel(time=SNAP).values * f
    cmap = "Blues" if v == "TP6h" else "RdBu_r"
    im = ax.imshow(field, cmap=cmap, aspect="auto",
                   extent=[0, 360, lat.min(), lat.max()], origin="lower")
    plt.colorbar(im, ax=ax, label=unit, fraction=0.03, pad=0.02)
    lg, sy, sm = bands[v]
    ax.set_title(f"{v}  ({META[v][0]}, {META[v][1]})   "
                 f"large {lg:.0%} / synoptic {sy:.0%} / small {sm:.0%}", fontsize=10)
    ax.set_xticks([0, 90, 180, 270, 360])
    if i >= 2: ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

# spectra, log-log
ax = fig.add_subplot(3, 1, 3)
for v in ds.data_vars:
    style = dict(lw=2.2) if v in show else dict(lw=1.0, alpha=0.45, ls="--")
    ax.loglog(k[1:], spectra[v][1:], label=v, **style)
ref = k[1:] ** -3.0
ax.loglog(k[1:], ref / ref.sum() * 0.9, "k:", lw=1.2, label="$k^{-3}$ reference")
ax.axvspan(1, 3, alpha=0.07, color="tab:green")
ax.axvspan(4, 8, alpha=0.07, color="tab:orange")
ax.axvspan(9, 32, alpha=0.07, color="tab:red")
ax.text(1.7, 3e-6, "large\n(planetary)", fontsize=8, ha="center")
ax.text(6, 3e-6, "synoptic\n(storms)", fontsize=8, ha="center")
ax.text(17, 3e-6, "small\n(sub-grid-ish)", fontsize=8, ha="center")
ax.set_xlabel("zonal wavenumber k  (waves around a latitude circle)")
ax.set_ylabel("fraction of variance")
ax.set_title("zonal power spectra, 2015 average — where each variable keeps its variance",
             fontsize=11)
ax.legend(ncol=5, fontsize=8, loc="upper right")
ax.grid(alpha=0.3, which="both")

fig.suptitle("Step 3 — different variables live at different spatial scales", fontsize=13)
fig.tight_layout()
fig.savefig("explore/figures/step3_variables_scales.png", dpi=140, bbox_inches="tight")
print("\nsaved explore/figures/step3_variables_scales.png")
