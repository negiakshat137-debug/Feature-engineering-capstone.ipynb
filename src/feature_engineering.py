"""
feature_engineering.py
Reusable feature extraction and construction utilities.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import MONTH_MAP


# ── Date / Time Features ──────────────────────────────────────────────────────

def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from arrival date columns."""
    df = df.copy()
    df['arrival_month_num'] = df['arrival_date_month'].map(MONTH_MAP)
    df['arrival_season'] = pd.cut(
        df['arrival_month_num'], bins=[0, 3, 6, 9, 12],
        labels=['Winter', 'Spring', 'Summer', 'Autumn'],
        include_lowest=True
    )
    df['arrival_quarter'] = df['arrival_month_num'].apply(lambda m: f'Q{(m - 1) // 3 + 1}')
    df['is_peak_season'] = df['arrival_month_num'].isin([6, 7, 8, 12]).astype(int)
    df['lead_time_bucket'] = pd.cut(
        df['lead_time'], bins=[-1, 7, 30, 90, 180, 10000],
        labels=['LastMinute', 'Short', 'Medium', 'Long', 'VeryLong']
    )
    return df


# ── Binning / Binarization ────────────────────────────────────────────────────

def add_binned_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add equal-width and quantile-binned versions of key columns."""
    df = df.copy()
    df['lead_time_bin_ew'] = pd.cut(
        df['lead_time'], bins=5,
        labels=['VeryShort', 'Short', 'Medium', 'Long', 'VeryLong']
    )
    df['adr_bin_q'] = pd.qcut(
        df['adr'], q=4,
        labels=['Budget', 'Economy', 'Standard', 'Premium'],
        duplicates='drop'
    )
    adr_threshold = df['adr'].quantile(0.75)
    df['high_value_customer'] = (df['adr'] > adr_threshold).astype(int)
    return df


# ── Ratio & Interaction Features ─────────────────────────────────────────────

def add_constructed_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio, interaction, and domain-derived features."""
    df = df.copy()

    # Ratio features
    df['price_per_person'] = df['adr'] / (
        df['adults'] + df['children'].fillna(0) + df['babies'] + 1
    )
    df['special_request_rate'] = df['total_of_special_requests'] / (
        df['stays_in_week_nights'] + df['stays_in_weekend_nights'] + 1
    )

    # Interaction features
    df['adr_x_lead_time']    = df['adr'] * df['lead_time']
    df['changes_x_requests'] = df['booking_changes'] * df['total_of_special_requests']

    # Aggregated features
    df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']

    # Binary flags
    df['is_family']        = ((df['children'].fillna(0) + df['babies']) > 0).astype(int)
    df['has_deposit']      = (df['deposit_type'] != 'No Deposit').astype(int)
    df['prev_cancel_flag'] = (df['previous_cancellations'] > 0).astype(int)

    return df


def add_polynomial_features(df: pd.DataFrame, cols=None, degree: int = 2) -> pd.DataFrame:
    """Add polynomial/interaction features for specified columns."""
    if cols is None:
        cols = ['lead_time', 'adr']
    df = df.copy()
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    poly_arr = poly.fit_transform(df[cols].fillna(0))
    poly_names = poly.get_feature_names_out(cols)
    for i, name in enumerate(poly_names):
        df[f'poly_{name}'] = poly_arr[:, i]
    return df


def add_group_features(df: pd.DataFrame, train_index=None) -> pd.DataFrame:
    """
    Add group-aggregated features.
    IMPORTANT: Pass train_index to avoid leakage — aggregates computed on
    training rows only, then mapped to all rows.
    """
    df = df.copy()
    ref = df.iloc[train_index] if train_index is not None else df

    country_avg_adr = ref.groupby('country')['adr'].mean()
    df['country_avg_adr'] = df['country'].map(country_avg_adr).fillna(ref['adr'].mean())

    return df


# ── TF-IDF Pseudo-text Features ───────────────────────────────────────────────

def add_tfidf_features(df: pd.DataFrame, max_features: int = 10) -> pd.DataFrame:
    """Combine hotel/market/meal into pseudo-text and apply TF-IDF."""
    df = df.copy()
    df['booking_description'] = (
        df['hotel'].fillna('') + ' ' +
        df['market_segment'].fillna('') + ' ' +
        df['meal'].fillna('')
    )
    tfidf = TfidfVectorizer(max_features=max_features)
    matrix = tfidf.fit_transform(df['booking_description'])
    tfidf_df = pd.DataFrame(
        matrix.toarray(),
        columns=[f'tfidf_{w}' for w in tfidf.get_feature_names_out()],
        index=df.index
    )
    return pd.concat([df, tfidf_df], axis=1)


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def build_full_feature_set(df: pd.DataFrame, train_index=None) -> pd.DataFrame:
    """Apply all feature engineering steps in sequence."""
    df = extract_datetime_features(df)
    df = add_binned_features(df)
    df = add_constructed_features(df)
    df = add_polynomial_features(df)
    df = add_group_features(df, train_index=train_index)
    return df
