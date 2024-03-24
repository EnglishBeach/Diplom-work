import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm

from tools import Solver

COMPS = ['Q', 'QHH', 'DH', 'D']


def get_df(y: Solver):
    res = {'time': y.T}
    for comp in COMPS:
        res[comp] = y[comp]
    return pd.DataFrame(res)


def variant(v):
    return [v * 0.1, v, v * 10]


def get_combinations(k: dict) -> dict:
    combination_dict = {key: variant(value) for key, value in k.items()}
    keys, values = zip(*combination_dict.items())
    return (dict(zip(keys, v)) for v in itertools.product(*values))


def sweep(Y1: Solver, Y2: Solver):
    eps = 1e-3
    df0 = np.array([1, *[Y1.initial[key] for key in COMPS]]) + eps

    data = []
    k_dict = Y2.K
    desc = tqdm(total=3 ** len(k_dict))
    i = 100
    for k in get_combinations(k_dict):

        Y1.solve(K=k)
        Y2.solve(K=k)
        df1 = get_df(Y1)
        df2 = get_df(Y2)
        df_res = np.abs(df1 - df2) / df0 * 100
        res: dict = df_res.max().iloc[1:].to_dict()
        res.update(k)
        data.append(res)
        desc.update()
        if i < 0:
            break
        i -= 1
    df = pd.DataFrame(data)
    df.to_csv('result.csv')
