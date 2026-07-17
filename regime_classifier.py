"""
Market-regime nowcaster.

Given trailing price/macro features as of day t, classify the CURRENT market regime
(Bull / Bear / Sideways / Crisis). This is nowcasting, not forecasting: features end at
day t and the label is day t's regime.

Pipeline:
  1. Engineer per-ticker technical features (momentum / MA deviation / volatility / price
     position) + macro features (vix, rates, unemployment, yield curve).
  2. Time-based train/test split (no look-ahead, no shuffle).
  3. RandomForest with balanced class weights.
  4. Report accuracy + per-class metrics + confusion matrix, feature importance, and a
     per-regime feature profile ("what characterizes each regime").
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "stock_market_regimes_2000_2026.csv")
TEST_START = "2019-01-01"   # out-of-sample: covers COVID crash + 2022 bear


# ── feature engineering (per ticker, trailing windows only) ───────────────────
def add_technical_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    c, r = g["close"], g["returns"]
    g["ret_1"]   = c.pct_change(1)
    g["ret_5"]   = c.pct_change(5)
    g["ret_20"]  = c.pct_change(20)
    g["ret_60"]  = c.pct_change(60)
    g["px_ma20"]  = c / c.rolling(20).mean()  - 1
    g["px_ma60"]  = c / c.rolling(60).mean()  - 1
    g["px_ma200"] = c / c.rolling(200).mean() - 1
    g["vol_20"] = r.rolling(20).std()
    g["vol_60"] = r.rolling(60).std()
    hi20, lo20 = c.rolling(20).max(), c.rolling(20).min()
    g["px_hi20"]  = c / hi20 - 1
    g["px_hi60"]  = c / c.rolling(60).max() - 1
    g["stoch20"]  = (c - lo20) / (hi20 - lo20).replace(0, np.nan)
    return g


TECH = ["ret_1", "ret_5", "ret_20", "ret_60",
        "px_ma20", "px_ma60", "px_ma200",
        "vol_20", "vol_60", "px_hi20", "px_hi60", "stoch20"]
MACRO = ["vix", "fed_funds_rate", "unemployment_rate", "yield_spread"]
FEATURES = TECH + MACRO


def load_and_prepare() -> pd.DataFrame:
    """Load data and engineer features."""
    df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["yield_spread"] = df["10y_treasury"] - df["2y_treasury"]
    df["regime"] = df["regime_label"]
    df = df[df["regime"] != "High-volatility"].copy()  # only 16 rows, cannot learn

    df = df.groupby("ticker", group_keys=False).apply(add_technical_features)
    return df.dropna(subset=FEATURES + ["regime"])


def main() -> None:
    df = load_and_prepare()
    print(f"usable rows after feature warmup: {len(df):,}")

    train = df[df["date"] < TEST_START]
    test  = df[df["date"] >= TEST_START]
    Xtr, ytr = train[FEATURES].to_numpy(), train["regime"].to_numpy()
    Xte, yte = test[FEATURES].to_numpy(),  test["regime"].to_numpy()
    print(f"train: {len(Xtr):,} rows (<{TEST_START})   test: {len(Xte):,} rows (>={TEST_START})")
    print("train label mix:", dict(train["regime"].value_counts()))

    clf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=50,
        class_weight="balanced", n_jobs=-1, random_state=7,
    )
    clf.fit(Xtr, ytr)

    pred = clf.predict(Xte)
    labels = sorted(np.unique(ytr))
    print("\n=== OUT-OF-SAMPLE PERFORMANCE (2019-2026) ===")
    print(f"accuracy   : {(pred == yte).mean():.3f}")
    print(f"macro F1   : {f1_score(yte, pred, average='macro'):.3f}")
    print("\n" + classification_report(yte, pred, digits=3))
    print("confusion matrix (rows=true, cols=pred):", labels)
    print(confusion_matrix(yte, pred, labels=labels))

    print("\n=== FEATURE IMPORTANCE (what the model uses to separate regimes) ===")
    imp = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    for name, val in imp.items():
        print(f"  {name:20} {val:.4f}")

    print("\n=== REGIME FEATURE PROFILE (mean in std units; +high / -low vs overall) ===")
    z = (df[FEATURES] - df[FEATURES].mean()) / df[FEATURES].std()
    z["regime"] = df["regime"].to_numpy()
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(z.groupby("regime")[FEATURES].mean().T.round(2))


if __name__ == "__main__":
    main()
