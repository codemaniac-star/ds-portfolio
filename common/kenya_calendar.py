"""Kenya public holidays as features for forecasting.

Fixed-date holidays are listed below. Add Eid and any new holidays yourself
and check dates against the official Kenya Gazette each year.
"""
import pandas as pd
from dateutil.easter import easter

FIXED = [  # (month, day, name)
    (1, 1, "New Year"),
    (5, 1, "Labour Day"),
    (6, 1, "Madaraka Day"),
    (10, 20, "Mashujaa Day"),
    (12, 12, "Jamhuri Day"),
    (12, 25, "Christmas"),
    (12, 26, "Boxing Day"),
]


def kenya_holidays(year):
    """Return a dict {date: name} for the given year."""
    days = {pd.Timestamp(year, m, d): name for m, d, name in FIXED}
    e = pd.Timestamp(easter(year))
    days[e - pd.Timedelta(days=2)] = "Good Friday"
    days[e + pd.Timedelta(days=1)] = "Easter Monday"
    return days


def add_holiday_features(df, date_col="date"):
    """Add is_holiday and days_to_holiday columns to a DataFrame."""
    df = df.copy()
    dates = pd.to_datetime(df[date_col])
    years = range(dates.dt.year.min() - 1, dates.dt.year.max() + 2)
    all_hols = sorted(d for y in years for d in kenya_holidays(y))
    df["is_holiday"] = dates.isin(all_hols).astype(int)
    hol_series = pd.Series(all_hols)
    df["days_to_holiday"] = dates.apply(
        lambda d: int((hol_series[hol_series >= d].min() - d).days)
        if (hol_series >= d).any() else -1
    )
    return df
