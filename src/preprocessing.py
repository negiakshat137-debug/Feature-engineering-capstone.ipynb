"""
preprocessing.py
Reusable preprocessing utilities for StaySmart Hotels Feature Engineering Capstone.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# ── Constants ─────────────────────────────────────────────────────────────────

LEAKY_COLS = ['reservation_status', 'reservation_status_date']

NUMERIC_COLS = [
    'lead_time', 'adr', 'stays_in_week_nights', 'stays_in_weekend_nights',
    'adults', 'children', 'babies', 'previous_cancellations',
    'booking_changes', 'total_of_special_requests'
]

CATEGORICAL_COLS = [
    'hotel', 'meal', 'market_segment', 'distribution_channel',
    'deposit_type', 'customer_type', 'reserved_room_type'
]

LOG_TRANSFORM_COLS = ['lead_time', 'adr', 'previous_cancellations']

MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_hotel_data(url: str = None) -> pd.DataFrame:
    """Load hotel bookings dataset and drop leaky columns."""
    if url is None:
        url = (
            'https://raw.githubusercontent.com/rfordatascience/tidytuesday'
            '/master/data/2020/2020-02-11/hotels.csv'
        )
    df = pd.read_csv(url)
    df = df.drop(columns=[c for c in LEAKY_COLS if c in df.columns])
    df['children'] = df['children'].fillna(0)
    return df


# ── Preprocessing Builders ────────────────────────────────────────────────────

def make_numeric_transformer(scaler: str = 'robust') -> Pipeline:
    """
    Build a numeric transformer pipeline.
    scaler: 'robust' | 'standard' | 'minmax'
    """
    scaler_map = {
        'robust': RobustScaler(),
        'standard': StandardScaler(),
        'minmax': MinMaxScaler()
    }
    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', scaler_map[scaler])
    ])


def make_log_numeric_transformer(scaler: str = 'robust') -> Pipeline:
    """Numeric transformer with log1p before scaling (for skewed features)."""
    scaler_map = {
        'robust': RobustScaler(),
        'standard': StandardScaler(),
        'minmax': MinMaxScaler()
    }
    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('log1p', FunctionTransformer(np.log1p, validate=True)),
        ('scale', scaler_map[scaler])
    ])


def make_categorical_transformer() -> Pipeline:
    """One-hot encode categorical columns with unknown handling."""
    return Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])


def build_preprocessor(
    log_cols=None,
    std_cols=None,
    cat_cols=None,
    scaler: str = 'robust'
) -> ColumnTransformer:
    """
    Build a ColumnTransformer with separate transformers for
    log-skewed numerics, standard numerics, and categoricals.
    """
    log_cols = log_cols or LOG_TRANSFORM_COLS
    std_cols = std_cols or [c for c in NUMERIC_COLS if c not in log_cols]
    cat_cols = cat_cols or CATEGORICAL_COLS

    return ColumnTransformer([
        ('log_num', make_log_numeric_transformer(scaler), log_cols),
        ('std_num', make_numeric_transformer(scaler),     std_cols),
        ('cat',     make_categorical_transformer(),        cat_cols)
    ])
