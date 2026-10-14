"""Turn the raw satellite time series into per-field-season feature vectors.

    python src/extract_features.py

WHAT THIS DOES, AND THE CHOICES THAT MATTER
-------------------------------------------
For each field-season it masks every acquisition down to the TARGET plot's
pixels only -- the patch contains several surveyed plots and the mask's
band 0 identifies them -- then reduces each acquisition to a few band means
over those pixels, then reduces the season to a fixed-length vector.

Three decisions worth stating because they bound everything downstream:

1. A plot is roughly 15 Sentinel-2 pixels. Per-acquisition spatial statistics
   beyond the mean are noise at that size, so only the mean is taken, plus
   the pixel count as a feature in its own right.

2. Cloud. Sentinel-2 ships SCL and MSK_CLDPRB. An acquisition is dropped when
   more than half the plot's pixels are flagged cloud, shadow, snow or
   saturated. The number of acquisitions surviving is itself recorded,
   because it is a legitimate inference-time quality signal and is needed
   later for the reliability phase.

3. Temporal reduction. Sequences vary from 30 to 146 acquisitions and are
   irregularly spaced, so a fixed-length summary is used rather than
   resampling to a grid: season-level statistics of each index, plus the
   value at the peak of the vegetation curve and when that peak occurs.

4. Mixed resolutions. One Sentinel-2 acquisition stores bands on three
   different grids -- B2/B3/B4/B8 at 10 m (34x33), B5-B8A/B11/B12/SCL at
   20 m (18x17), B1/B9 at 60 m (7x6) -- and Sentinel-1 sits on its own
   33x33 grid. Rather than upsample the coarse bands, which would invent
   detail the sensor never recorded, the PLOT MASK is resampled down onto
   each band's native grid by nearest neighbour. A 20 m band therefore
   contributes the handful of its own pixels that overlap the plot. The
   60 m bands B1 and B9 are not used at all: at that scale a plot is a
   fraction of one pixel and the value is dominated by its neighbours.

Nothing here uses the yield label, and the script asserts that.

Output: results/features.csv, one row per field-season.
"""

from __future__ import annotations

import csv
import glob
import json
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJ = Path(__file__).resolve().parents[1]
# dataset root; SICKLE_DATA_ROOT as in sickle-project/.env.example
ROOT = Path(__import__("os").environ.get(
    "SICKLE_DATA_ROOT", "D:/FPD/sickledataset extracted"))
OUT = PROJ / "results" / "features.csv"

# Sentinel-2 Scene Classification values to reject.
# 1 saturated, 3 shadow, 8/9 cloud medium/high, 10 cirrus, 11 snow.
SCL_BAD = {1, 3, 8, 9, 10, 11}


def acq_date(path: str) -> datetime | None:
    m = re.search(r"(\d{8})", Path(path).stem)
    return datetime.strptime(m.group(1), "%Y%m%d") if m else None


def plot_pixels(uid: str, pid: float, res: str):
    """Boolean mask of the target plot inside this patch."""
    f = ROOT / f"masks/{res}/{uid}.tif"
    if not f.exists():
        return None
    return tifffile.imread(f)[0] == pid


def fit_mask(sel: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resample of the plot mask onto a band's own grid.

    The mask is moved to the data rather than the data to the mask, so a
    coarse band is read at the resolution it was actually acquired. Returns
    an all-False array only if the plot vanishes at that scale, which the
    caller treats as "this band cannot see this plot".
    """
    if sel.shape == shape:
        return sel
    yi = (np.arange(shape[0]) * sel.shape[0] / shape[0]).astype(int)
    xi = (np.arange(shape[1]) * sel.shape[1] / shape[1]).astype(int)
    yi = np.clip(yi, 0, sel.shape[0] - 1)
    xi = np.clip(xi, 0, sel.shape[1] - 1)
    return sel[np.ix_(yi, xi)]


def s2_series(uid: str, sel):
    """Per-acquisition means over the plot, with cloud screening.

    Each band is read on its own grid with the mask resampled to match, so
    10 m and 20 m bands are both used without resampling any imagery.
    """
    recs = []
    for f in sorted(glob.glob(str(ROOT / f"images/S2/npy/{uid}/*.npz"))):
        d = np.load(f)

        def band(name):
            if name not in d:
                return np.nan
            a = d[name]
            m = fit_mask(sel, a.shape)
            if m.sum() == 0:
                return np.nan
            return float(np.nanmean(a[m]))

        # Cloud screen on SCL, read at its own 20 m grid.
        if "SCL" in d:
            a = d["SCL"]
            m = fit_mask(sel, a.shape)
            if m.sum() and np.isin(a[m].astype(int), list(SCL_BAD)).mean() > 0.5:
                continue

        red, nir, g2, b2 = band("B4"), band("B8"), band("B3"), band("B2")
        swir, re1 = band("B11"), band("B5")
        if not np.isfinite([red, nir, g2, b2]).all():
            continue
        rec = {
            "date": acq_date(f),
            "ndvi": (nir - red) / (nir + red + 1e-9),
            "ndwi": (g2 - nir) / (g2 + nir + 1e-9),
            "evi": 2.5 * (nir - red) / (nir + 6 * red - 7.5 * b2 + 1e4),
            "red": red, "nir": nir,
        }
        # 20 m indices only where that grid actually sees the plot.
        rec["ndmi"] = ((nir - swir) / (nir + swir + 1e-9)
                       if np.isfinite(swir) else np.nan)
        rec["reci"] = (nir / (re1 + 1e-9) - 1.0
                       if np.isfinite(re1) else np.nan)
        rec["swir"] = swir
        recs.append(rec)
    return recs


def s1_series(uid: str, sel):
    """Sentinel-1 backscatter means over the plot, on S1's own grid."""
    recs = []
    for f in sorted(glob.glob(str(ROOT / f"images/S1/npy/{uid}/*.npz"))):
        d = np.load(f)
        if "VV" not in d or "VH" not in d:
            continue
        m = fit_mask(sel, d["VV"].shape)
        if m.sum() == 0:
            continue
        vv = float(np.nanmean(d["VV"][m]))
        vh = float(np.nanmean(d["VH"][fit_mask(sel, d["VH"].shape)]))
        if not np.isfinite([vv, vh]).all():
            continue
        recs.append({"date": acq_date(f), "vv": vv, "vh": vh,
                     "rvi": 4 * vh / (vv + vh + 1e-9)})
    return recs


def reduce_series(recs, keys, prefix) -> dict:
    """Fixed-length season summary of an irregular time series."""
    out = {}
    if not recs:
        return {f"{prefix}_{k}_{s}": np.nan
                for k in keys for s in ("mean", "max", "min", "sd", "auc")}
    for k in keys:
        v = np.array([r[k] for r in recs], float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            for s in ("mean", "max", "min", "sd", "auc"):
                out[f"{prefix}_{k}_{s}"] = np.nan
            continue
        out[f"{prefix}_{k}_mean"] = float(v.mean())
        out[f"{prefix}_{k}_max"] = float(v.max())
        out[f"{prefix}_{k}_min"] = float(v.min())
        out[f"{prefix}_{k}_sd"] = float(v.std())
        out[f"{prefix}_{k}_auc"] = float(v.sum() / len(v))
    return out


def phenology(recs) -> dict:
    """Where the greenness peak sits, and how fast it rises and falls."""
    if len(recs) < 4:
        return {"peak_frac": np.nan, "rise": np.nan, "fall": np.nan,
                "n_acq": len(recs)}
    recs = [r for r in recs if r["date"] is not None]
    if len(recs) < 4:
        return {"peak_frac": np.nan, "rise": np.nan, "fall": np.nan,
                "n_acq": len(recs)}
    recs = sorted(recs, key=lambda r: r["date"])
    nd = np.array([r["ndvi"] for r in recs])
    t = np.array([(r["date"] - recs[0]["date"]).days for r in recs], float)
    i = int(np.argmax(nd))
    span = max(t[-1] - t[0], 1.0)
    return {
        "peak_frac": float(t[i] / span),
        "rise": float((nd[i] - nd[0]) / max(t[i] - t[0], 1.0) * 100),
        "fall": float((nd[i] - nd[-1]) / max(t[-1] - t[i], 1.0) * 100),
        "n_acq": len(recs),
    }


def main() -> int:
    split = json.loads((PROJ / "configs/splits.json").read_text(encoding="utf8"))
    rows = []
    with (ROOT / "sickle_dataset_tabular.csv").open(encoding="utf8") as fh:
        for r in csv.DictReader(fh):
            try:
                y, pid = float(r["YIELD"]), float(r["PLOT_ID"])
            except (ValueError, KeyError):
                continue
            if y > 0:
                rows.append((r, y, pid))
    assign = {u: s for s, v in split["field_seasons"].items() for u in v}
    print(f"{len(rows)} field-seasons to process")

    recs_out, skipped = [], []
    for n, (r, y, pid) in enumerate(rows, 1):
        uid = r["UNIQUE_ID"]
        sel = plot_pixels(uid, pid, "10m")
        if sel is None or sel.sum() == 0:
            skipped.append((uid, "no plot pixels in 10 m mask"))
            continue
        s2 = s2_series(uid, sel)
        s1 = s1_series(uid, sel)
        f = {"uid": uid, "plot": pid, "split": assign.get(uid, "?"),
             "season": r["STANDARD_SEASON"], "year": r["YEAR"],
             "variety": r["VARIETY"],
             "area": float(r["AREA"]) if r["AREA"].strip() else np.nan,
             "n_px": int(sel.sum()),
             "yield": y}
        f.update(reduce_series(s2, ["ndvi", "ndwi", "ndmi", "evi", "reci",
                                    "red", "nir", "swir"], "s2"))
        f.update(reduce_series(s1, ["vv", "vh", "rvi"], "s1"))
        ph = phenology(s2)
        f.update({f"ph_{k}": v for k, v in ph.items()})
        f["s2_n_clear"] = len(s2)
        f["s1_n_acq"] = len(s1)
        recs_out.append(f)
        if n % 40 == 0:
            print(f"  {n}/{len(rows)}")

    # The target must not have leaked into a feature.
    feat_cols = [c for c in recs_out[0]
                 if c not in ("uid", "plot", "split", "season", "year",
                              "variety", "yield")]
    yv = np.array([r["yield"] for r in recs_out])
    for c in feat_cols:
        v = np.array([r[c] for r in recs_out], float)
        ok = np.isfinite(v)
        if ok.sum() > 10 and abs(np.corrcoef(v[ok], yv[ok])[0, 1]) > 0.999:
            raise SystemExit(f"feature {c!r} is collinear with the target")

    cols = ["uid", "plot", "split", "season", "year", "variety", "yield"] + feat_cols
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in recs_out:
            w.writerow({c: r.get(c, "") for c in cols})

    print(f"\nwrote {OUT}: {len(recs_out)} rows x {len(feat_cols)} features")
    if skipped:
        print(f"SKIPPED {len(skipped)}: {skipped[:5]}")
    miss = {c: sum(1 for r in recs_out if not np.isfinite(float(r[c] or np.nan)))
            for c in feat_cols}
    worst = sorted(miss.items(), key=lambda kv: -kv[1])[:5]
    print(f"missing-value counts, worst 5: {worst}")
    print(f"clear S2 acquisitions per sample: "
          f"min {min(r['s2_n_clear'] for r in recs_out)}, "
          f"median {int(np.median([r['s2_n_clear'] for r in recs_out]))}, "
          f"max {max(r['s2_n_clear'] for r in recs_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
