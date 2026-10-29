"""Rolling-origin backtesting for time series."""
import pandas as pd


def rolling_origin_splits(n_rows, initial_train, horizon, step=None):
    """Yield (train_index, test_index) pairs that always train on the past
    and test on the future. Never shuffles the data.

    n_rows        total number of rows in the series
    initial_train size of the first training window
    horizon       how many rows ahead each test window covers
    step          how far the window moves each time (default = horizon)
    """
    step = step or horizon
    start = initial_train
    while start + horizon <= n_rows:
        yield list(range(0, start)), list(range(start, start + horizon))
        start += step


def backtest(series, fit_predict, initial_train, horizon, step=None):
    """Run a model over rolling splits.

    fit_predict(train_series, horizon) must return a list of `horizon` predictions.
    Returns a DataFrame with columns: actual, predicted, fold.
    """
    rows = []
    for fold, (tr, te) in enumerate(rolling_origin_splits(len(series), initial_train, horizon, step)):
        preds = fit_predict(series.iloc[tr], horizon)
        for idx, p in zip(te, preds):
            rows.append({"date": series.index[idx], "actual": series.iloc[idx],
                         "predicted": p, "fold": fold})
    return pd.DataFrame(rows).set_index("date")
