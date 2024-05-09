import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
import numpy as np
import copy

plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('font', size=10)
plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=15)
plt.rc('legend', fontsize=10)
plt.rcParams["figure.figsize"] = (10, 7)


class K_gen(dict):
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

    def copy(self) -> dict:
        return copy.deepcopy(self)


def get_init(comps_str: str, init: dict):
    comps = comps_str.replace('[', '').replace(']', '').split(', ')
    return {comp: init.get(comp, 0) for comp in comps}


class Solver:
    def __init__(self, system, K: K_gen, initial: dict, T: np.ndarray) -> None:
        self._system = system
        self.K = K
        self.T = T

        self.initial = initial
        self._comp = {key: i for i, key in enumerate(self.initial)}

        self.solve()

    def is_correct(self):
        return (0 <= self.y.min()) & (self.y.max() < 1e6)

    def solve(self, initial: dict = {}, K: dict = {}):
        init = self.initial.copy()
        init.update(initial)

        k = self.K.copy()
        for key, value in K.items():
            k[key] = value

        # print('*' * 50)
        for method in [
            'BDF',
            'Radau',
            'LSODA',
            'RK45',
        ]:
            solution = solve_ivp(
                fun=self._system,
                t_span=[0, self.T.max()],
                y0=list(init.values()),
                args=(k,),
                method=method,
                dense_output=True,
                # rtol=0.00001,
                max_step=self.T[1],
            )
            self.y = solution.sol(self.T)
            # print(method)
            if self.is_correct():
                break

    def get_specific(self, value):
        match value:
            case 'P':
                M = self.y['M']
                DM = self.y['DM']
                M0 = self.initial['M']
                return M0 - M - DM

    def __getitem__(self, value):
        if value in ['P']:
            return self.get_specific(value)
        return self.y[self._comp[value]]


class Sweeper:
    def __init__(self, y1, y2) -> None:
        self.y1: Solver = y1
        self.y2: Solver = y2
        self.K = y2.K
        self.T = y1.T

        self.initial = self.y2.initial

    def __getitem__(self, comp):
        sigma = 1e-9
        return self.y1[comp] - self.y2[comp]

    def solve(self, initial={}, K={}):
        self.y2.solve(initial=initial, K=K)


class Comparator:
    def __init__(self, y1, y2) -> None:
        self.y1: Solver = y1
        self.y2: Solver = y2
        self.K = y1.K
        self.T = y1.T

        self.initial = self.y1.initial

    def __getitem__(self, comp):
        sigma = 1e-9
        return self.y1[comp] - self.y2[comp]

    def solve(self, initial={}, K={}):
        self.y1.solve(initial=initial, K=K)
        self.y2.solve(initial=initial, K=K)
