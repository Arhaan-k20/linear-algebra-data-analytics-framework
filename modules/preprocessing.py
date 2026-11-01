import pandas as pd
import numpy as np


def prepare_numeric_data(df):
    numeric_df = df.select_dtypes(include=np.number).copy()

    if numeric_df.empty:
        raise ValueError("No numerical columns found in the dataset.")

    numeric_df = numeric_df.dropna()

    if numeric_df.empty:
        raise ValueError(
            "No valid numerical rows remain after removing missing values."
        )

    return numeric_df


def standardize_data(df):
    mean = df.mean()
    standard_deviation = df.std(ddof=0)

    standard_deviation = standard_deviation.replace(
        0,
        1
    )

    standardized_data = (
        df - mean
    ) / standard_deviation

    return standardized_data