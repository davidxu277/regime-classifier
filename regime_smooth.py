"""Step 1: smooth regime labels into persistent regimes.

The raw labels flip every ~7 days (median run 3 days), which is almost certainly
labeling noise rather than real regime switching. We consolidate short runs into
their longer neighbor so each regime segment lasts at least `min_dur` trading days,
then report how choppiness and class balance change.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "stock_market_regimes_2000_2026.csv")


def _runs(lab: np.ndarray) -> list[list]:
    """Return [start, end_inclusive, value] runs."""
    runs, i, n = [], 0, len(lab)
    while i < n:
        j = i
        while j + 1 < n and lab[j + 1] == lab[i]:
            j += 1
        runs.append([i, j, lab[i]])
        i = j + 1
    return runs


def smooth(lab: np.ndarray, min_dur: int) -> np.ndarray:
    """Iteratively absorb the shortest sub-min_dur run into its longer neighbor."""
    lab = lab.copy()
    while True:
        runs = _runs(lab)
        if len(runs) <= 1:
            break
        lengths = [r[1] - r[0] + 1 for r in runs]
        k = int(np.argmin(lengths))
        if lengths[k] >= min_dur:
            break
        left = runs[k - 1] if k > 0 else None
        right = runs[k + 1] if k < len(runs) - 1 else None
        if left and right:
            new = left[2] if (left[1] - left[0]) >= (right[1] - right[0]) else right[2]
        else:
            new = (left or right)[2]
        lab[runs[k][0]: runs[k][1] + 1] = new
    return lab


def choppiness(df: pd.DataFrame, col: str) -> dict:
    flips, runs = [], []
    for _, g in df.groupby("ticker"):
        lab = g.sort_values("date")[col].to_numpy()
        flips.append((lab[1:] != lab[:-1]).mean())
        runs.extend([r[1] - r[0] + 1 for r in _runs(lab)])
    runs = np.array(runs)
    return {
        "flip_%": np.mean(flips) * 100,
        "median_run": np.median(runs),
        "mean_run": runs.mean(),
        "frac_<=5d_%": (runs <= 5).mean() * 100,
    }


def main() -> None:
    df = pd.read_csv(DATA, usecols=["date", "ticker", "regime_label"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    print("=== RAW ===")
    raw = choppiness(df, "regime_label")
    print({k: round(v, 1) for k, v in raw.items()})
    print("class mix:", (df["regime_label"].value_counts(normalize=True) * 100).round(1).to_dict())

    print("\n=== after min-duration smoothing ===")
    print(f"{'min_dur':>8} | {'flip%':>6} | {'median_run':>10} | {'mean_run':>8} | {'<=5d%':>6} | class mix")
    for md in (3, 5, 10, 15):
        col = f"sm{md}"
        df[col] = df.groupby("ticker")["regime_label"].transform(
            lambda s: smooth(s.to_numpy(), md)
        )
        c = choppiness(df, col)
        mix = (df[col].value_counts(normalize=True) * 100).round(1).to_dict()
        print(f"{md:>8} | {c['flip_%']:>6.1f} | {c['median_run']:>10.0f} | "
              f"{c['mean_run']:>8.0f} | {c['frac_<=5d_%']:>6.0f} | {mix}")

    # save the full dataset with a chosen smoothed column for downstream use
    out = os.path.join(HERE, "regimes_smoothed.csv")
    full = pd.read_csv(DATA)
    full["date"] = pd.to_datetime(full["date"])
    full = full.sort_values(["ticker", "date"]).reset_index(drop=True)
    for md in (10,):
        full[f"regime_smooth{md}"] = full.groupby("ticker")["regime_label"].transform(
            lambda s: smooth(s.to_numpy(), md)
        )
    full.to_csv(out, index=False)
    print(f"\nsaved smoothed dataset (regime_smooth10) -> {out}")


if __name__ == "__main__":
    main()
