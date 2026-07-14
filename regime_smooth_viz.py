"""Visualize raw vs smoothed regime labels on the S&P 500 (^GSPC)."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
COLORS = {"Bull": "#2E7D32", "Bear": "#C62828", "Crisis": "#6A1B9A",
          "Sideways": "#F9A825", "High-volatility": "#607D8B"}

df = pd.read_csv(os.path.join(HERE, "regimes_smoothed.csv"), parse_dates=["date"])
g = df[df["ticker"] == "^GSPC"].sort_values("date")
g = g[(g["date"] >= "2018-01-01") & (g["date"] <= "2023-12-31")]
dates = g["date"].to_numpy()
price = g["close"].to_numpy()


def shade(ax, lab):
    i, n = 0, len(lab)
    while i < n:
        j = i
        while j + 1 < n and lab[j + 1] == lab[i]:
            j += 1
        ax.axvspan(dates[i], dates[min(j + 1, n - 1)], color=COLORS.get(lab[i], "#999"), alpha=0.35, lw=0)
        i = j + 1


fig, ax = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
fig.suptitle("^GSPC (S&P 500) — regime labels before vs after smoothing (2018–2023)",
             fontsize=14, fontweight="bold")

for a, col, title in [(ax[0], "regime_label", "RAW  (flips every ~7 days)"),
                      (ax[1], "regime_smooth10", "SMOOTHED  min_dur=10 (persistent regimes)")]:
    a.plot(dates, price, color="black", lw=1.0, zorder=3)
    shade(a, g[col].to_numpy())
    a.set_title(title, loc="left", fontsize=11)
    a.set_ylabel("S&P 500")
    a.margins(x=0)

handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.35) for c in
           [COLORS[k] for k in ["Bull", "Sideways", "Bear", "Crisis"]]]
ax[0].legend(handles, ["Bull", "Sideways", "Bear", "Crisis"], ncol=4, loc="upper left", framealpha=0.9)

fig.tight_layout(rect=[0, 0, 1, 0.96])
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
out = os.path.join(HERE, "figures", "regime_smoothing_comparison.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
