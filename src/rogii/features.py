"""Per-well features: recovered baseline context + FLE3N-style GR statistics.

Only the supplied prefix TVT_input and the reference typewell TVT are available.
Horizontal TVT/formation columns never cross this module's input boundary.
All GR/geometry observations are assumed available for offline trajectory inference.
"""
import numpy as np
import pandas as pd

INPUT_COLUMNS = ["MD", "X", "Y", "Z", "GR", "TVT_input"]


def target_mask(horizontal):
    missing = horizontal["TVT_input"].isna().to_numpy()
    if not missing.any() or missing.all():
        raise ValueError("Require a nonempty known prefix and prediction suffix")
    start = int(np.flatnonzero(missing)[0])
    if not missing[start:].all():
        raise ValueError("TVT_input must be a contiguous prefix followed by missing targets")
    if not np.isfinite(horizontal.loc[:start - 1, "TVT_input"]).all():
        raise ValueError("Known TVT context must be finite")
    return missing


def build_features(horizontal, typewell, well_id):
    # Explicit whitelist is essential: do not copy the target and drop it later.
    h = horizontal.loc[:, INPUT_COLUMNS].reset_index(drop=True).copy()
    tw = typewell.loc[:, ["TVT", "GR"]].copy()
    if not np.isfinite(h[["MD", "X", "Y", "Z"]].to_numpy()).all():
        raise ValueError("Geometry must be finite")
    if not h.MD.is_monotonic_increasing or h.MD.duplicated().any():
        raise ValueError("MD must be strictly increasing in original CSV row order")
    mask = target_mask(h)
    idx = np.flatnonzero(mask)
    start = int(idx[0])
    known = h.iloc[:start]
    last = known.iloc[-1]
    recent = known.tail(100)
    slope = (np.polyfit(recent.MD - last.MD, recent.TVT_input, 1)[0]
             if len(recent) > 1 else 0.0)
    f = h.loc[mask, ["MD", "X", "Y", "Z", "GR"]].copy()
    f["row_index"] = idx
    f["rows_after_ps"] = idx - start
    f["frac_after_ps"] = (idx - start) / max(len(h) - start, 1)
    f["well_total_rows"] = len(h)
    f["last_known_tvt"] = float(last.TVT_input)
    f["recent_tvt_md_slope"] = float(slope)
    for c in ["MD", "X", "Y", "Z", "GR"]:
        f[c.lower() + "_at_ps"] = last[c]
        f[c.lower() + "_delta_from_ps"] = h.loc[mask, c] - last[c]
    f["linear_tvt_baseline"] = last.TVT_input + slope * (f.MD - last.MD)
    # Filling uses only this well's observed GR, never its missing TVT labels.
    gr = h.GR.replace([np.inf, -np.inf], np.nan)
    gr = gr.fillna(gr.median() if gr.notna().any() else 0.0)
    for w in (5, 21, 51):
        roll = gr.rolling(w, min_periods=1, center=True)
        for name, values in (("mean", roll.mean()), ("std", roll.std().fillna(0)),
                             ("range", roll.max() - roll.min())):
            f[f"GR_{name}_{w}"] = values.iloc[idx].to_numpy()
    f["GR_grad"] = np.gradient(gr.to_numpy())[idx]
    f["GR_zscore"] = ((gr - gr.mean()) / (gr.std() + 1e-8)).iloc[idx].to_numpy()
    f["dZ"] = h.Z.diff().fillna(0).iloc[idx].to_numpy()
    for c in ("TVT", "GR"):
        values = tw[c].replace([np.inf, -np.inf], np.nan)
        for stat in ("min", "max", "mean", "std"):
            f[f"tw_{c.lower()}_{stat}"] = getattr(values, stat)()
    # Nearest gamma-ray match is a proxy, not a unique geological alignment.
    valid = tw.replace([np.inf, -np.inf], np.nan).dropna().sort_values("GR", kind="stable")
    if valid.empty:
        raise ValueError("Reference typewell needs finite TVT and GR observations")
    tg, tt = valid.GR.to_numpy(), valid.TVT.to_numpy()
    hg = gr.iloc[idx].to_numpy()
    right = np.searchsorted(tg, hg).clip(0, len(tg) - 1)
    left = (right - 1).clip(0, len(tg) - 1)
    nearest = np.where(np.abs(tg[left] - hg) <= np.abs(tg[right] - hg), left, right)
    f["tw_nearest_tvt"] = tt[nearest]
    f = f.replace([np.inf, -np.inf], np.nan).astype("float32").reset_index(drop=True)
    ids = [f"{well_id}_{i}" for i in idx]
    return f, ids


def aligned_submission(sample, ids, predictions):
    if len(ids) != len(predictions) or len(set(ids)) != len(ids):
        raise ValueError("Predictions require unique IDs and equal lengths")
    if sample.id.duplicated().any() or set(sample.id) != set(ids):
        raise ValueError("Submission and predicted IDs must match exactly")
    values = np.asarray(predictions, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Predictions must be finite; no median fallback is allowed")
    out = sample[["id"]].merge(pd.DataFrame({"id": ids, "tvt": values}),
                               on="id", how="left", validate="one_to_one", sort=False)
    if out.id.tolist() != sample.id.tolist():
        raise ValueError("Submission order changed")
    return out
