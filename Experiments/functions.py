import numpy as np
import pandas as pd


## Functions
def nu_D(time, x, y):
    # k = 1.380649 * 1e-23
    k = 1
    y = k * x / (y * 0.001)
    return time, x, y, dict(k=1)


def linearize(time, T, y):
    T = 1 / T
    y = np.log(y)
    return time, T, y


def delinearize(time, x, y):
    x = 1 / x
    y = np.exp(y)
    return time, x, y


## Group filters
def z_filter(data: pd.Series):
    mean = data.mean()
    s = data.std(ddof=0) + 1e-50
    z_score = np.abs((data - mean) / s) < 1
    return z_score


def whisker_iqr_filter(data: pd.Series):
    whisker_width = 0.5
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1 + 1e-50
    return (data >= q1 - whisker_width * iqr) & (data <= q3 + whisker_width * iqr)


def iqr_filter(data: pd.Series):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1 + 1e-50
    return np.abs((data - data.median()) / iqr) < 1
