"""Plot the raw regime labels as colored bands over the S&P 500 (^GSPC)."""
from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "stock_market_regimes_2000_2026.csv")
COLORS = {"Bull": "#2E7D32", "Bear": "#C62828", "Crisis": "#6A1B9A",
          "Sideways": "#F9A825", "High-volatility": "#607D8B"}

df = pd.read_csv(DATA, parse_dates=["date"])
g = df[df["ticker"] == "^GSPC"].sort_values("date")
g = g[(g["date"] >= "2018-01-01") & (g["date"] <= "2023-12-31")]
dates = g["date"].to_numpy()
price = g["close"].to_numpy()
lab = g["regime_label"].to_numpy()

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(dates, price, color="black", lw=1.0, zorder=3)
i, n = 0, len(lab)
while i < n:
    j = i
    while j + 1 < n and lab[j + 1] == lab[i]:
        j += 1
    ax.axvspan(dates[i], dates[min(j + 1, n - 1)], color=COLORS.get(lab[i], "#999"), alpha=0.35, lw=0)
    i = j + 1

ax.set_title("^GSPC (S&P 500) with daily regime labels (2018–2023)", fontsize=13, fontweight="bold")
ax.set_ylabel("S&P 500")
ax.margins(x=0)
handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k], alpha=0.35)
           for k in ["Bull", "Sideways", "Bear", "Crisis"]]
ax.legend(handles, ["Bull", "Sideways", "Bear", "Crisis"], ncol=4, loc="upper left", framealpha=0.9)

fig.tight_layout()
out = os.path.join(HERE, "figures", "regime_bands.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
