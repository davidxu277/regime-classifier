"""Dashboard of what the nowcast regime model learned (trained on smoothed labels)."""
from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from regime_classifier import load_and_prepare, FEATURES, TEST_START

HERE = os.path.dirname(os.path.abspath(__file__))
COLORS = {"Bull": "#2E7D32", "Bear": "#C62828", "Crisis": "#6A1B9A", "Sideways": "#F9A825"}

df = load_and_prepare()
train = df[df["date"] < TEST_START]
test = df[df["date"] >= TEST_START]
clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=50,
                             class_weight="balanced", n_jobs=-1, random_state=7)
clf.fit(train[FEATURES], train["regime"])
labels = sorted(df["regime"].unique())

fig, ax = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("Regime Nowcaster — what it learned (smoothed labels; train 2000–2018, test 2019–2026)",
             fontsize=15, fontweight="bold")

# (0,0) scatter: trend vs volatility, colored by regime
samp = df.sample(9000, random_state=1)
for reg in labels:
    s = samp[samp["regime"] == reg]
    ax[0, 0].scatter(s["px_ma60"] * 100, s["vol_20"] * 100, s=6, alpha=0.35, c=COLORS[reg], label=reg)
ax[0, 0].set_xlabel("Trend  (price vs 60-day MA, %)")
ax[0, 0].set_ylabel("Volatility  (20-day return std, %)")
ax[0, 0].set_title("Regimes separate on trend × volatility")
ax[0, 0].set_xlim(-40, 40); ax[0, 0].set_ylim(0, 8)
ax[0, 0].legend(markerscale=3, framealpha=0.9)
ax[0, 0].axvline(0, color="gray", lw=0.7, ls="--")

# (0,1) feature importance
imp = pd.Series(clf.feature_importances_, index=FEATURES).sort_values()
ax[0, 1].barh(imp.index, imp.values, color="#1565C0")
ax[0, 1].set_title("Feature importance (what the model relies on)")
ax[0, 1].set_xlabel("importance")

# (1,0) confusion matrix (row-normalized = recall)
pred = clf.predict(test[FEATURES])
cm = confusion_matrix(test["regime"], pred, labels=labels).astype(float)
cm_norm = cm / cm.sum(axis=1, keepdims=True)
ax[1, 0].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax[1, 0].set_xticks(range(len(labels)), labels, rotation=30)
ax[1, 0].set_yticks(range(len(labels)), labels)
ax[1, 0].set_xlabel("predicted"); ax[1, 0].set_ylabel("true")
ax[1, 0].set_title("Confusion matrix (out-of-sample, row-normalized)")
for i in range(len(labels)):
    for j in range(len(labels)):
        ax[1, 0].text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                      color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=10)

# (1,1) regime feature profile (standardized means)
z = (df[FEATURES] - df[FEATURES].mean()) / df[FEATURES].std()
z["regime"] = df["regime"].to_numpy()
profile = z.groupby("regime")[FEATURES].mean().T[labels]
im2 = ax[1, 1].imshow(profile.values, cmap="RdBu_r", vmin=-1.3, vmax=1.3, aspect="auto")
ax[1, 1].set_xticks(range(len(labels)), labels, rotation=30)
ax[1, 1].set_yticks(range(len(FEATURES)), FEATURES)
ax[1, 1].set_title("Regime profile (mean feature, std units)")
for i in range(len(FEATURES)):
    for j in range(len(labels)):
        ax[1, 1].text(j, i, f"{profile.values[i, j]:+.1f}", ha="center", va="center",
                      color="white" if abs(profile.values[i, j]) > 0.7 else "black", fontsize=8)
fig.colorbar(im2, ax=ax[1, 1], fraction=0.046, pad=0.04)

fig.tight_layout(rect=[0, 0, 1, 0.97])
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
out = os.path.join(HERE, "figures", "regime_dashboard.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
