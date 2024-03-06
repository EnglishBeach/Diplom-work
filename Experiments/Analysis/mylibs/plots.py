import os

import matplotlib.pyplot as plt
import seaborn as sns

from .tools import Experiment

# Plots and load
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


def temperature_plot(
    experiment: Experiment,
    title='',
    xlabel='',
    ylabel='',
    interactive=False,
    save_folder=None,
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

    if interactive:
        plt.show()
    if save_folder is not None:
        os.makedirs(f'{save_folder}\Plots', exist_ok=True)
        fig.savefig(f'{save_folder}\Plots\\{title}_{experiment.name}.jpg', dpi=600)


def comparation_plot(
    experiment: Experiment,
    ols_exp: Experiment,
    title,
    xlabel,
    ylabel,
    interactive=False,
    save_folder=None,
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

    if interactive:
        plt.show()
    if save_folder is not None:
        os.makedirs(f'{save_folder}\Plots', exist_ok=True)
        fig.savefig(f'{save_folder}\Plots\\{title}_{experiment.name}.jpg', dpi=600)
