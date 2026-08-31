#!/usr/bin/env python3
"""Preprocess raw Bitcoin minute data into hourly windows for forecasting.

The raw datasets hold one row per minute with OHLC prices, traded volumes and
a weighted price.  This script aggregates them to an hourly resolution,
handles missing intervals, performs a chronological train/validation split,
scales the features with training statistics only and stores everything in a
compressed NumPy archive consumed by ``forecast_btc.py``.
"""
import os
import sys

import numpy as np
import pandas as pd


FEATURES = ["Open", "High", "Low", "Close",
            "Volume_(BTC)", "Volume_(Currency)", "Weighted_Price"]
TARGET = "Close"
WINDOW = 24
TRAIN_FRACTION = 0.8
PREFERRED = "bitstamp"
OUTPUT = "btc_preprocessed.npz"


def find_dataset(argv):
    """Return the path to the raw CSV to use.

    A path given on the command line wins.  Otherwise the known project
    locations are scanned and the ``PREFERRED`` exchange is chosen when
    several candidates exist.
    """
    if len(argv) > 1:
        return argv[1]

    here = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [
        here,
        os.path.join(here, "data"),
        os.path.join(here, "..", "..", "pipeline", "pandas"),
        os.path.join(here, "..", ".."),
    ]
    candidates = []
    for directory in search_dirs:
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            low = name.lower()
            if low.endswith(".csv") and "1-min" in low and (
                    "bitstamp" in low or "coinbase" in low):
                candidates.append(os.path.join(directory, name))

    if not candidates:
        return None
    for path in candidates:
        if PREFERRED in os.path.basename(path).lower():
            return path
    return sorted(candidates)[0]


def load_raw(path):
    """Load the raw minute CSV and return it indexed by datetime."""
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
    df = df.set_index("Timestamp").sort_index()
    return df


def to_hourly(df):
    """Aggregate the minute data to hourly OHLCV rows.

    Missing minutes inside an hour are ignored by the aggregation; hours with
    no trade at all are forward filled for prices (last known price carries
    over) and set to zero for volumes.
    """
    agg = df.resample("1h").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume_(BTC)": "sum",
        "Volume_(Currency)": "sum",
        "Weighted_Price": "mean",
    })

    # Drop the leading span before the very first trade.
    agg = agg.loc[agg["Close"].first_valid_index():]

    # Hours without any trade: carry the last close forward for all prices.
    agg["Close"] = agg["Close"].ffill()
    for col in ("Open", "High", "Low", "Weighted_Price"):
        agg[col] = agg[col].fillna(agg["Close"])
    for col in ("Volume_(BTC)", "Volume_(Currency)"):
        agg[col] = agg[col].fillna(0.0)

    agg = agg.dropna()
    return agg[FEATURES]


def main():
    """Run the full preprocessing pipeline."""
    path = find_dataset(sys.argv)
    if path is None or not os.path.isfile(path):
        print("Raw Bitcoin CSV not found. Pass its path as an argument, e.g.")
        print("  ./preprocess_data.py path/to/bitstampUSD_1-min_data.csv")
        sys.exit(1)

    raw = load_raw(path)
    rows_loaded = len(raw)

    hourly = to_hourly(raw)
    data = hourly.to_numpy(dtype="float64")
    rows_after = data.shape[0]

    split = int(rows_after * TRAIN_FRACTION)
    train_data = data[:split]
    val_data = data[split:]

    feature_mean = train_data.mean(axis=0)
    feature_std = train_data.std(axis=0)
    feature_std[feature_std == 0] = 1.0

    train_scaled = (train_data - feature_mean) / feature_std
    val_scaled = (val_data - feature_mean) / feature_std

    target_index = FEATURES.index(TARGET)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT)
    np.savez_compressed(
        out_path,
        train_data=train_scaled,
        val_data=val_scaled,
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_names=np.array(FEATURES),
        target_index=target_index,
        window=WINDOW,
    )

    assert not np.isnan(train_scaled).any()
    assert not np.isnan(val_scaled).any()

    print("rows loaded:            {}".format(rows_loaded))
    print("rows after preprocessing: {}".format(rows_after))
    print("selected features:     {}".format(", ".join(FEATURES)))
    print("train size:            {}".format(len(train_scaled)))
    print("validation size:       {}".format(len(val_scaled)))
    print("window (hours):        {}".format(WINDOW))
    print("output file:           {}".format(out_path))


if __name__ == "__main__":
    main()
