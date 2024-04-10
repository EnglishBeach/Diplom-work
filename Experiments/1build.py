import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import Experiments.functions as f
from z_base import VERBOSE_COLORS, Experiment, input_path

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


def regress(experiment: Experiment):
    experiment = experiment.apply(functions.C_to_K).apply(functions.linearize)

    df = experiment.d
    df["x0"] = 1
    result = sm.OLS(df["y"], df[["x", "x0"]]).fit()
    means = result.params

    D0 = np.exp(means["x0"])
    E = -8.314 * means["x"]

    conf_int = result.conf_int(0.005).loc
    conf_int["x0"] = np.exp(conf_int["x0"])
    dD0 = (conf_int["x0"].max() - conf_int["x0"].min()) / 2
    conf_int["x"] = -8.314 * conf_int["x"]
    dE = (conf_int["x"].max() - conf_int["x"].min()) / 2

    info = dict(
        E=E,
        D0=D0,
        dD0=dD0,
        dE=dE,
        f_statistic=result.fvalue,
        r2=result.rsquared,
    )
    func = lambda T: D0 * np.exp(-E / (8.314 * T))
    return info, result, func


def create_OLS(exp: Experiment):
    info, result, func = regress(exp)
    exp.set_info(**info)
    x = np.linspace(13, 42, 100) + 273.15
    ols_res = Experiment(
        pd.DataFrame(
            {
                "x": x,
                "y": func(x),
                "time": x * 0,
            }
        ),
        "interpolated",
    )
    return ols_res.apply(functions.K_to_C)


def initial_filter(df, x=(-np.inf, np.inf), y=(0, np.inf), time=(0, np.inf)):
    temperature_cond = (x[0] < df["x"]) & (df["x"] < x[1])
    viscosity_cond = (y[0] < df["y"]) & (df["y"] < y[1])
    time_cond = (time[0] < df["time"]) & (df["time"] < time[1])
    return df[temperature_cond & viscosity_cond & time_cond]


def ask_continue():
    res = None
    while res is None:
        ask = input("Continue [y] and n: ")
        if ask in ["", "y"]:
            res = True
        elif ask in ["n"]:
            res = False
        else:
            print("Incorrect input!")
    return res


def configurate_data(experiment: Experiment) -> Experiment:
    while True:
        exp = experiment.copy()
        time_lim = ()
        while len(time_lim) != 2:
            time_lim = input("Time lim (space as delimiter): ")
            time_lim = [float(i) for i in time_lim.split(" ") if "" != i]
            if len(time_lim) == 1:
                time_lim.append(np.inf)

        y_lim = ()
        while len(y_lim) != 2:
            y_lim = input("Viscosity lim (space as delimiter): ")
            y_lim = [float(i) for i in y_lim.split(" ")]

        exp.d = initial_filter(exp.d, time=time_lim, y=y_lim, x=(12, 42))
        exp.info.append(("initial_filter", {"time": time_lim, "y": y_lim, "x": (12, 42)}))

        _temporal_plot(
            exp,
            title="Configurate",
            ylabel="Viscosity [cP]",
            interactive=True,
        )
        if ask_continue():
            break
    return exp


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


def temporal_plot(
    experiment: Expseriment,
    title="",
    ylabel="",
    interactive=False,
    save_folder=None,
):
    fig, ax_v = plt.subplots()
    ax_T = ax_v.twinx()
    ax_v.scatter(experiment.d["time"], experiment.d["y"], color="red", marker=".")
    ax_T.scatter(experiment.d["time"], experiment.d["x"], color="blue", marker=".")

    fig.canvas.manager.set_window_title(title + " plot")
    fig.subplots_adjust(
        top=0.9,
        bottom=0.1,
        left=0.1,
        right=0.9,
        hspace=0.2,
        wspace=0.2,
    )
    ax_T.set_title(f"{experiment.name}: ({experiment.info['w']}% mass)")
    ax_v.set_xlabel("Time [s]")
    ax_T.set_ylabel("Temperature [C]", color="blue")
    ax_v.set_ylabel(ylabel, color="red")

    if interactive:
        plt.show()
    if save_folder is not None:
        os.makedirs(f"{save_folder}\Plots", exist_ok=True)
        fig.savefig(f"{save_folder}\Plots\\{title}_{experiment.name}.jpg", dpi=600)


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


exp = Experiment()
exp.load_csv(*input_path())
temporal_plot(exp)

exp = configurate_data(exp)
temperature_plot(
    exp,
    title='Viscosity',
    xlabel='Temperature [C]',
    ylabel='Viscosity [cP]',
    interactive=True,
)


exp.apply(f.C_to_K)
exp.apply(f.nu_D)
exp.apply(f.linearize)
exp.group_filter(f.iqr_filter)
exp.apply(f.delinearize)
exp.apply(f.K_to_C)
print('Filtered')

temperature_plot(
    exp,
    title='Diffusion',
    xlabel='Temperature [C]',
    ylabel='D [m2/s]',
    interactive=True,
)

info, result, func = regress(exp)
exp.set_info(**info)
x = np.linspace(13, 42, 100) + 273.15
ols_res = Experiment(
    pd.DataFrame({'x': x, 'y': func(x), 'time': x * 0}),
    'interpolated',
)

ols_res.apply(f.K_to_C)

exp.info

tmp = exp.copy()

exp2 = exp.copy()
ols_res2 = ols_res.copy()
exp2.apply(f.C_to_K)
ols_res2.apply(f.C_to_K)
exp2.apply(f.linearize)
ols_res2.apply(f.linearize)

comparation_plot(
    exp2,
    ols_res2,
    title='OLS_Linear',
    xlabel='Temperature',
    ylabel='D',
    interactive=True,
)

comparation_plot(
    exp,
    ols_res,
    title='OLS_Diffusion',
    xlabel='Temperature [C]',
    ylabel='D [m2/s]',
    interactive=True,
)

exp.save_hdf5()
