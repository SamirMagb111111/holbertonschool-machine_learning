# Time Series Forecasting

Forecast the Bitcoin (BTC/USD) close price for the **next hour** from the
**previous 24 hours** of market data, using a recurrent neural network.

## Concepts

- **Time-series forecasting** — predicting future values of a sequence from
  its own past (and related) observations, where order in time matters.
- **Stationary process** — one whose statistical properties (mean, variance,
  autocorrelation) do not change over time. Raw BTC prices are *not*
  stationary (strong trend); scaling and using short 24-hour windows keeps
  the model focused on local dynamics.
- **Sliding window** — a fixed-length span (`24` hours here) that slides one
  step at a time over the series, each position giving one training sample:
  the 24 hours are the input, the 25th hour's close is the target.

## Data

- **Source used:** `bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv`
  (Bitstamp). It is the longer and more recent of the two available files;
  the exchanges are **not** concatenated, to avoid mixing order books.
- **Features (7):** `Open`, `High`, `Low`, `Close`, `Volume_(BTC)`,
  `Volume_(Currency)`, `Weighted_Price`. `Close` is kept both as an input and
  as the prediction target. `Timestamp` is used only to order and index the
  data.

## Preprocessing (`preprocess_data.py`)

1. Parse `Timestamp` (Unix seconds) to datetime, sort chronologically, index.
2. **Hourly aggregation** of the minute rows with OHLC-correct rules:
   `Open=first`, `High=max`, `Low=min`, `Close=last`,
   volumes `=sum`, `Weighted_Price=mean`.
3. **Missing data:** drop the span before the first trade; for hours with no
   trade, forward-fill `Close` and derive `Open/High/Low/Weighted_Price` from
   it, set volumes to `0` (no trade = no volume). No future value is ever
   used to fill a past one.
4. **Chronological split:** oldest 80 % → train, newest 20 % → validation.
   No shuffling of the raw series.
5. **Normalization:** `(x - mean) / std` with **training-set** mean/std only
   (zero std guarded); validation is scaled with those same statistics.
6. Save `btc_preprocessed.npz`: scaled `train_data`, `val_data`,
   `feature_mean`, `feature_std`, `feature_names`, `target_index`, `window`.
   This file is generated locally and is not committed.

Window length is `24` because each aggregated step is one hour, so 24 steps
span exactly the "previous 24 hours" the task asks for.

## Forecasting (`forecast_btc.py`)

- Loads the archive and builds `(24, 7)` windows → next-hour `Close` targets
  with `tf.keras.utils.timeseries_dataset_from_array`, wrapped in
  **`tf.data.Dataset`** pipelines: the training pipeline shuffles *after*
  windowing, then `batch(64)` and `prefetch(AUTOTUNE)`; validation is not
  shuffled.
- **Model (RNN):**
  `Input(24, 7)` → `GRU(64, return_sequences=True)` → `GRU(32)` →
  `Dense(16, relu)` → `Dense(1)`.
- **Loss:** mean squared error (`mse`); `mae` is also reported.
  Optimizer: Adam. `EarlyStopping(monitor="val_loss",
  restore_best_weights=True)`, up to 20 epochs.
- Evaluation prints validation MSE/MAE (scaled) plus MAE/RMSE converted back
  to USD, and a handful of `predicted vs actual` close prices.

## Data-leakage safeguards

Chronological split, training-only scaling statistics, windows never contain
their own target, validation is strictly the later observations, no
future-to-past filling.

## Files

| File | Purpose |
| --- | --- |
| `preprocess_data.py` | Build the hourly, scaled, split dataset archive. |
| `forecast_btc.py` | Train and evaluate the GRU forecaster via `tf.data`. |
