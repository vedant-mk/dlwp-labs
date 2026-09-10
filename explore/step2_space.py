"""Step 2 — the geometry of the 32x64 grid: cell size, cos-latitude area, longitude wrap."""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

R = 6371.0  # km

ds = xr.open_dataset("cache/data.nc").sel(time=slice("2015-01-01", "2015-12-31T18"))
lat = ds.latitude.values
lon = ds.longitude.values
nlat, nlon = lat.size, lon.size
dlat = 180.0 / nlat
dlon = 360.0 / nlon

# exact spherical cell area: R^2 * dlon_rad * (sin(lat_top) - sin(lat_bot))
edges = np.linspace(90.0, -90.0, nlat + 1)          # lat decreasing north->south
top, bot = edges[:-1], edges[1:]
area = (R**2) * np.deg2rad(dlon) * (np.sin(np.deg2rad(top)) - np.sin(np.deg2rad(bot)))
area = np.abs(area)

# order rows so index matches the data's latitude ordering
if lat[0] < lat[-1]:
    area = area[::-1]

eq = np.argmin(np.abs(lat))
po = np.argmax(np.abs(lat))
ns_km = R * np.deg2rad(dlat)
ew_eq = R * np.deg2rad(dlon) * np.cos(np.deg2rad(lat[eq]))
ew_po = R * np.deg2rad(dlon) * np.cos(np.deg2rad(lat[po]))

print(f"grid                : {nlat} lat x {nlon} lon, spacing {dlat:g}deg x {dlon:g}deg")
print(f"north-south extent  : {ns_km:.0f} km  (same at every latitude)")
print(f"east-west  equator  : {ew_eq:.0f} km  (lat {lat[eq]:+.2f})")
print(f"east-west  polemost : {ew_po:.0f} km  (lat {lat[po]:+.2f})")
print(f"cell area  equator  : {area[eq]:,.0f} km^2")
print(f"cell area  polemost : {area[po]:,.0f} km^2")
print(f"AREA RATIO eq/pole  : {area[eq]/area[po]:.1f} x")
print(f"cos(lat) ratio      : {np.cos(np.deg2rad(lat[eq]))/np.cos(np.deg2rad(lat[po])):.1f} x")
print(f"total area check    : {area.sum()*nlon:,.0f} km^2 vs sphere {4*np.pi*R**2:,.0f} km^2")

# ViT token level: patch_size [4,4] -> 8 x 16 tokens, each 22.5deg x 22.5deg
p = 4
tok_area = area.reshape(nlat // p, p).sum(axis=1) * p
print(f"\nViT tokens          : {nlat//p} x {nlon//p}, each {p*dlat:g}deg x {p*dlon:g}deg")
print(f"token area equator  : {tok_area[len(tok_area)//2]:,.0f} km^2")
print(f"token area polemost : {tok_area[0]:,.0f} km^2")
print(f"TOKEN AREA RATIO    : {tok_area[len(tok_area)//2]/tok_area[0]:.1f} x")

# ---------------------------------------------------------------- figure
z = ds.Z500.sel(time="2015-01-15T00:00").values / 9.81  # geopotential -> metres

fig = plt.figure(figsize=(13, 9))

# (a) grid on a globe
ax = fig.add_subplot(2, 2, 1, projection=ccrs.Orthographic(0, 35))
ax.set_global()
ax.coastlines(linewidth=0.4, color="0.4")
for y in np.linspace(-90, 90, nlat + 1):
    ax.plot(np.linspace(-180, 180, 361), np.full(361, y),
            transform=ccrs.PlateCarree(), color="tab:red", lw=0.35)
for x in np.linspace(-180, 180, nlon + 1):
    ax.plot(np.full(181, x), np.linspace(-90, 90, 181),
            transform=ccrs.PlateCarree(), color="tab:red", lw=0.35)
ax.set_title("(a) the real grid on a sphere\ncells shrink toward the poles", fontsize=10)

# (b) the same grid flat -- what the ViT sees
ax = fig.add_subplot(2, 2, 2)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(0, nlon); ax.set_ylim(0, nlat)
for i in range(nlat + 1):
    ax.axhline(i, color="tab:red", lw=0.35)
for j in range(nlon + 1):
    ax.axvline(j, color="tab:red", lw=0.35)
ax.set_aspect("equal")
ax.set_title("(b) what the ViT sees: a flat 32x64 rectangle\nevery cell identical", fontsize=10)

# (c) cell area vs latitude
ax = fig.add_subplot(2, 2, 3)
ax.plot(lat, area, "o-", ms=3, color="tab:blue")
ax.set_xlabel("latitude (deg)"); ax.set_ylabel("cell area (km$^2$)")
ax.grid(alpha=0.3)
ax.set_title(f"(c) true cell area vs latitude\nequator/pole ratio = {area[eq]/area[po]:.1f}x",
             fontsize=10)
ax.annotate(f"{area[eq]:,.0f}", (lat[eq], area[eq]), textcoords="offset points",
            xytext=(0, -18), ha="center", fontsize=8)
ax.annotate(f"{area[po]:,.0f}", (lat[po], area[po]), textcoords="offset points",
            xytext=(-8, 10), ha="right", fontsize=8)

# (d) longitude wrap
ax = fig.add_subplot(2, 2, 4)
ax.imshow(np.concatenate([z, z], axis=1), cmap="RdBu_r", aspect="auto")
ax.axvline(nlon - 0.5, color="k", lw=2, ls="--")
ax.set_xticks([0, nlon, 2 * nlon]); ax.set_xticklabels(["0", "360", "720"])
ax.set_xlabel("longitude (deg), field repeated twice")
ax.set_yticks([]); ax.set_ylabel("latitude")
ax.set_title("(d) Z500 tiled twice: the seam is invisible\nlongitude wraps; the ViT does not know",
             fontsize=10)

fig.suptitle("Step 2 — the 32x64 grid is a flat picture of a round Earth", fontsize=13)
fig.tight_layout()
fig.savefig("explore/figures/step2_space.png", dpi=140, bbox_inches="tight")
print("\nsaved explore/figures/step2_space.png")
