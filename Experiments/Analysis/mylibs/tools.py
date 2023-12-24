import os
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from . import functions

def input_path(path=''):
    while (path == '') or (not os.path.isfile(path)):
        path = input(f"Input data path: ")
    return path

def _split_path(path=''):
    path_list = (path).split('\\')
    folder = '\\'.join(path_list[:-1])
    name = path_list[-1].split('.')[0]
    return folder, name


class Experiment:
    d: pd.DataFrame = None
    folder = None
    name = None

    def __init__(self, data=None, name=''):
        self.d = data
        self.name = name
        self.log = []
        self.info = {}

    def set_info(self, **info):
        self.info.update(info)

    def _log_wrapp(func):

        def log_wrapper(self, *args, **kwargs):
            res = func(self, *args, **kwargs)
            self.log.append(res)
            return self.d

        return log_wrapper

    @_log_wrapp
    def load_csv(self, path):
        self.folder, self.name = _split_path(path)
        self.d = pd.read_csv(path)
        self.d.rename(columns={'Temperature': 'x', 'Viscosity': 'y'}, inplace=True)
        return ('csv loaded', path)

    @_log_wrapp
    def load_hdf5(self, path):
        self.folder, self.name = _split_path(path)
        with pd.HDFStore(path) as file:
            self.d = file['data']
            self.info.update(file.get_storer('data').attrs.info)
            self.log.extend(file.get_storer('data').attrs.log)
        return ('hdf5 loaded', path)

    def save_hdf5(self, folder=None):
        path = folder if folder is not None else self.folder
        assert path is not None, 'Path not define'
        file_path = f'{path}\{self.name}.hdf5'
        with pd.HDFStore(file_path) as file:
            file.put('data', self.d)
            file.get_storer('data').attrs.log = self.log
            file.get_storer('data').attrs.info = self.info

    def copy(self):
        return copy.deepcopy(self)

    @_log_wrapp
    def apply(self, func):
        self.d['time'], self.d['x'], self.d['y'] = func(self.d['time'], self.d['x'], self.d['y'])
        return (func.__name__, [])

    @_log_wrapp
    def group_filter(self, filter, by='x', column='y'):
        group = self.d.groupby(by=by)[column]
        mask = group.apply(filter).droplevel([0]).sort_index().to_numpy()
        self.d = self.d[mask]
        return (filter.__name__, [])


def regress(experiment: Experiment):

    exp = experiment.copy()
    exp.apply(functions.C_to_K)
    exp.apply(functions.linearize)

    df = exp.d
    df['x0'] = 1
    result = sm.OLS(df['y'], df[['x', 'x0']]).fit()
    means = result.params

    D0 = np.exp(means['x0'])
    E = -8.314 * means['x']

    conf_int = result.conf_int(0.005).loc
    conf_int['x0'] = np.exp(conf_int['x0'])
    dD0 = (conf_int['x0'].max() - conf_int['x0'].min()) / 2
    conf_int['x'] = -8.314 * conf_int['x']
    dE = (conf_int['x'].max() - conf_int['x'].min()) / 2

    info = dict(
        E=E,
        D0=D0,
        dD0=dD0,
        dE=dE,
        f_statistic=result.fvalue,
        r2=result.rsquared,
    )
    func = lambda T: D0 * np.exp(-E / (8.314*T))
    return info, result, func


# Plots and load
pd.set_option('mode.chained_assignment', None)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('font', size=10)
plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=15)
plt.rc('legend', fontsize=10)
plt.rcParams["figure.figsize"] = (10, 7)

VERBOSE_COLORS = {
    'OK': 'g',
    'OK_inner': 'b',
    'image_sweep_check': 'r',
    'combine_check': 'w',
}


def temporal_plot(
    experiment,
    title='',
    ylabel='',
    interactive=False,
    save=False,
):
    fig, ax_v = plt.subplots()
    ax_T = ax_v.twinx()
    ax_v.scatter(experiment.d['time'], experiment.d['y'], color='red', marker='.')
    ax_T.scatter(experiment.d['time'], experiment.d['x'], color='blue', marker='.')

    fig.canvas.manager.set_window_title(title + ' plot')
    fig.subplots_adjust(
        top=0.9,
        bottom=0.1,
        left=0.1,
        right=0.9,
        hspace=0.2,
        wspace=0.2,
    )
    ax_v.set_xlabel('Time [s]')
    ax_T.set_ylabel('Temperature [C]', color='blue')
    ax_v.set_ylabel(ylabel, color='red')

    if interactive: plt.show()
    if save:
        os.makedirs(f'{experiment.folder}\Plots', exist_ok=True)
        fig.savefig(f'{experiment.folder}\Plots\\{title}_{experiment.name}.jpg', dpi=600)
    return experiment


def _initial_filter(df, x=(-np.inf, np.inf), y=(0, np.inf), time=(0, np.inf)):
    temperature_cond = ((x[0] < df['x']) & (df['x'] < x[1]))
    viscosity_cond = ((y[0] < df['y']) & (df['y'] < y[1]))
    time_cond = ((time[0] < df['time']) & (df['time'] < time[1]))
    return df[temperature_cond & viscosity_cond & time_cond]


def _ask_continue():
    res = None
    while res is None:
        ask = input('Continue [y] and n: ')
        if ask in ['', 'y']:
            res = True
        elif ask in ['n']:
            res = False
        else:
            print('Incorrect input!')
    return res


def configurate_data(experiment: Experiment, save=False) -> Experiment:
    while True:
        exp = experiment.copy()
        time_lim = ()
        while len(time_lim) != 2:
            time_lim = input('Time lim (space as delimiter): ')
            time_lim = [float(i) for i in time_lim.split(' ') if '' != i]
            if len(time_lim) == 1: time_lim.append(np.inf)

        y_lim = ()
        while len(y_lim) != 2:
            y_lim = input('Viscosity lim (space as delimiter): ')
            y_lim = [float(i) for i in y_lim.split(' ')]

        exp.d = _initial_filter(exp.d, time=time_lim, y=y_lim, x=(12, 42))
        exp.log.append(('initial_filter', {'time': time_lim, 'y': y_lim, 'x': (12, 42)}))

        temporal_plot(
            exp,
            title='Configurate',
            ylabel='Viscosity [cP]',
            save=save,
            interactive=True,
        )
        if _ask_continue(): break
    return exp


def temperature_plot(
    experiment: Experiment,
    title='',
    xlabel='',
    ylabel='',
    interactive=False,
    save=False,
):
    fig, ax = plt.subplots()
    colors = experiment.d['Viscosity_verbose'].replace(VERBOSE_COLORS)
    ax.scatter(x=experiment.d['x'], y=experiment.d['y'], c=colors, s=5)
    sns.lineplot(
        ax=ax,
        data=experiment.d,
        x='x',
        y='y',
        estimator='mean',
        errorbar=("sd", 1),
        label='mean',
    )
    sns.lineplot(
        ax=ax,
        data=experiment.d,
        x="x",
        y="y",
        errorbar=('pi', 50),
        estimator="median",
        label='median',
    )

    fig.canvas.manager.set_window_title(title + ' plot')
    fig.subplots_adjust(
        top=0.9,
        bottom=0.1,
        left=0.1,
        right=0.9,
        hspace=0.2,
        wspace=0.2,
    )
    ax.set_title(f"{experiment.name}: ({experiment.info['w']}% mass)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if interactive: plt.show()
    if save:
        os.makedirs(f'{experiment.folder}\Plots', exist_ok=True)
        fig.savefig(f'{experiment.folder}\Plots\\{title}_{experiment.name}.jpg', dpi=600)


def comparation_plot(
    experiment: Experiment,
    ols_exp: Experiment,
    title,
    xlabel,
    ylabel,
    interactive=False,
    save=False,
):
    fig, ax = plt.subplots()
    colors = experiment.d['Viscosity_verbose'].replace(VERBOSE_COLORS)
    ax.scatter(
        experiment.d['x'],
        experiment.d['y'],
        color=colors,
        marker='.',
        label='Data',
        alpha=0.6,
    )

    E = experiment.info['E']
    D0 = experiment.info['D0']
    ax.plot(
        ols_exp.d['x'],
        ols_exp.d['y'],
        color='black',
        label=f'OLS: \nE= {E/1000: >8.2f} kJ\nD= {D0: >8.2e} m2/s',
    )

    fig.canvas.manager.set_window_title(title + ' plot')
    fig.subplots_adjust(
        top=0.9,
        bottom=0.1,
        left=0.1,
        right=0.9,
        hspace=0.2,
        wspace=0.2,
    )

    ax.set_title(f"{experiment.name}: ({experiment.info['w']}% mass)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    if interactive: plt.show()
    if save:
        os.makedirs(f'{experiment.folder}\Plots', exist_ok=True)
        fig.savefig(f'{experiment.folder}\Plots\\{title}_{experiment.name}.jpg', dpi=600)
