import numpy as np
import pandas as pd


## Functions
def nu_D(time, x, y):
    # k = 1.380649 * 1e-23
    k = 1
    y = k * x / (y*0.001)
    return time, x, y


def nu_to_v(time, x, y):
    ro = 1.73
    y = y / ro
    return time, x, y


def K_to_C(time, x, y):
    x = x - 273.15
    return time, x, y


def C_to_K(time, x, y):
    x = x + 273.15
    return time, x, y


def linearize(time, x, y):
    x = 1 / x
    y = np.log(y)
    return time, x, y


def delinearize(time, x, y):
    x = 1 / x
    y = np.exp(y)
    return time, x, y


## Group filters
def z_filter(data: pd.Series):
    mean = data.mean()
    s = data.std(ddof=0) + 1e-50
    z_score = np.abs((data-mean) / s) < 1
    return z_score


def whisker_iqr_filter(data: pd.Series):
    whisker_width = 0.5
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1 + 1e-50
    return (data >= q1 - whisker_width*iqr) & (data <= q3 + whisker_width*iqr)


def iqr_filter(data: pd.Series):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1 + 1e-50

    return np.abs((data - data.median()) / iqr) < 1
