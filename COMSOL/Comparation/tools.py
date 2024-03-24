import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from matplotlib.widgets import Slider, Button
from scipy.integrate import solve_ivp

plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('font', size=10)
plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=15)
plt.rc('legend', fontsize=10)
plt.rcParams["figure.figsize"] = (10, 7)


class BaseK(dict):
    def __init__(self):
        consts = {
            key: value
            for key, value in self.__class__.__dict__.items()
            if (key not in ['__module__'] + dir(dict)) and '__' not in key
        }
        super().__init__(**consts)

    def __setitem__(self, __key, __value):
        self.__setattr__(__key, __value)
        return super().__setitem__(__key, __value)


class Solver:
    def __init__(self, system, K, initial, comps, T) -> None:
        self.system = system
        self.K = K
        self.initial = initial
        self.T = T

        comps = comps.replace('[', '').replace(']', '')
        self.comp = {}
        for i, key in enumerate(comps.split(', ')):
            self.comp[key] = i
        self.solve()

    def solve(self, initial=None, K=None):
        init = initial if initial is not None else self.initial
        k = K if K is not None else self.K

        solution = solve_ivp(
            fun=self.system,
            t_span=[0, self.T.max()],
            y0=init,
            args=(k,),
            dense_output=True,
        )
        self.y = solution.sol(self.T)

    def get_specific(self, value):
        match value:
            case 'P':
                M = self.y[self.comp['M']]
                DM = self.y[self.comp['DM']]
                M0 = self.initial[self.comp['M']]
                return M0 - M - DM

    def __getitem__(self, value):
        if value in ['P']:
            return self.get_specific(value)
        return self.y[self.comp[value]]

    def get_df(self, comps: list):
        res = {'time': self.T}
        for comp in comps:
            res[comp] = self[comp]
        return pd.DataFrame(res)


import itertools


def variant(v):
    return [v * 0.1, v * 0.5, v, v * 2, v * 10]


def get_combinations(k: dict):
    combination_dict = {key: variant(value) for key, value in k.items()}
    keys, values = zip(*combination_dict.items())
    return (dict(zip(keys, v)) for v in itertools.product(*values))


# def sweep(Y1:Solver,Y2:Solver):
#     Y1.
#     K_base = Y1.K

#     for k in Y1.K:
#         Y1.solve(
#             K=
#         )


class Comparator:
    def __init__(self, y1, y2) -> None:
        self.y1: Solver = y1
        self.y2: Solver = y2
        self.K = y1.K
        self.T = y1.T

    def __getitem__(self, comp):
        sigma = 1e-9
        return self.y1[comp] - self.y2[comp]

    def solve(self, initial=None, K=None):
        self.y2.solve(initial=initial, K=K)


class Differ:
    def __init__(self, y1, y2) -> None:
        self.y1: Solver = y1
        self.y2: Solver = y2
        self.K = y1.K
        self.T = y1.T

    def __getitem__(self, comp):
        sigma = 1e-9
        return self.y1[comp] - self.y2[comp]

    def solve(self, initial=None, K=None):
        self.y1.solve(initial=initial, K=K)
        self.y2.solve(initial=initial, K=K)
