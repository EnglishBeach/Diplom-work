import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from matplotlib.widgets import Slider
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
            if key.startswith('K') or key in ['light']
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

    def __getitem__(self, value):
        return self.y[self.comp[value]]


class Differ:
    def __init__(self, y1, y2) -> None:
        self.y1: Solver = y1
        self.y2: Solver = y2
        self.K = y1.K
        self.T = y1.T

    def __getitem__(self, comp):
        sigma = 1e-9
        return (self.y1[comp] - self.y2[comp]) / (self.y1[comp] + sigma) * 100

    def solve(self, initial=None, K=None):
        self.y1.solve(initial=initial, K=K)
        self.y2.solve(initial=initial, K=K)


def show_plot(y: Solver | Differ, set_lim=False):
    lims = 50
    fig, ax = plt.subplots()
    gs = plt.GridSpec(2, 2, figure=fig)
    fig.subplots_adjust(left=0.25, right=0.99, bottom=0.1, top=0.95, hspace=0.1, wspace=0.1)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    scal = FuncFormatter(lambda x, pos: f'{x*1000: .2f}')
    K = y.K
    y.solve()

    plots = {}

    base_ax = plt.subplot(gs[0, 0:2])
    for c in ['Q', 'DH', 'QHH']:
        plots[c] = base_ax.plot(y.T, y[c], label=c)[0]
    base_ax.legend()
    base_ax.xaxis.set_major_formatter(scal)
    if set_lim:
        base_ax.set_ylim([-lims, lims])

    d_ax = plt.subplot(gs[1, 0])
    plots['D'] = d_ax.plot(y.T, y['D'], label='D')[0]
    d_ax.legend()
    d_ax.xaxis.set_major_formatter(scal)
    if set_lim:
        d_ax.set_ylim([-lims, lims])

    other_ax = plt.subplot(gs[1, 1])
    other_ax.xaxis.set_major_formatter(scal)
    for c in ['QH', 'QHD']:
        plots[c] = other_ax.plot(y.T, y[c], label=c)[0]
    other_ax.legend()
    other_ax.xaxis.set_major_formatter(scal)
    if set_lim:
        other_ax.set_ylim([-lims, lims])

    sliders = {}
    for i, k in enumerate(K):
        k_axes = fig.add_axes([0.05, 0.95 - 0.05 * i, 0.15, 0.05])
        amp_slider = Slider(
            ax=k_axes,
            label=k,
            valmin=-3,
            valmax=3,
            valinit=0,
            valstep=0.1,
            orientation="horizontal",
        )
        sliders[k] = amp_slider

    def update(val):
        k = copy.deepcopy(K)
        for h in K:
            k[h] = K[h] * 10 ** sliders[h].val

        y.solve(K=k)
        for c, plot in plots.items():
            plot.set_ydata(y[c])
        fig.canvas.draw_idle()

    for h in K:
        sliders[h].on_changed(update)

    return fig
