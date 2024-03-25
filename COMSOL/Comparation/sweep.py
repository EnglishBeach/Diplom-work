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
    data = []
    k_dict = Y2.K
    desc = tqdm(total=3 ** (len(k_dict) + 2))
    try:
        for k in get_combinations(k_dict):
            for init in get_combinations({'Q': [1, 10, 20], 'QHH': [0, 1, 2], 'DH': [1, 10, 20]}):
                Y1.solve(K=k, initial=init)
                Y2.solve(K=k, initial=init)

                df1 = get_df(Y1)
                df2 = get_df(Y2)

                df_res = np.abs(df1 - df2)
                res: dict = df_res.max().iloc[1:].to_dict()
                res.update(k)
                res.update(init)
                data.append(res)
                desc.update()
    finally:
        df = pd.DataFrame(data)
        df.to_csv('result.csv')
