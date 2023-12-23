import os
import copy

import pandas as pd
import numpy as np
from sklearn import linear_model
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('mode.chained_assignment', None)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('font', size=10)  #controls default text size
plt.rc('axes', titlesize=16)  #fontsize of the title
plt.rc('axes', labelsize=15)  #fontsize of the x and y labels
plt.rc('legend', fontsize=10)
plt.rcParams["figure.figsize"] = (10, 7)


def ask_continue():
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

    def log_wrapp(func):

        def log_wrapper(self, *args, **kwargs):
            res = func(self, *args, **kwargs)
            self.log.append(res)
            return self.d

        return log_wrapper

    @log_wrapp
    def load_csv(self, path=''):
        path, self.folder, self.name = self._input_path(path)
        self.d = pd.read_csv(path)
        self.d.rename(columns={'Temperature': 'x', 'Viscosity': 'y'}, inplace=True)
        return ('csv loaded', path)

    @log_wrapp
    def load_hdf5(self, path=''):
        path, self.folder, self.name = self._input_path(path)

        with pd.HDFStore(path) as file:
            self.d = file['data']
            self.info.update(file.get_storer('data').attrs.info)
            self.log.extend(file.get_storer('data').attrs.log)
        return ('hdf5 loaded', path)

    def save_hdf5(self):
        file_path = f'{self.folder}\{self.name}.hdf5'
        with pd.HDFStore(file_path) as file:
            file.put('data', self.d)
            file.get_storer('data').attrs.log = self.log
            file.get_storer('data').attrs.info = self.info

    def copy(self):
        return copy.deepcopy(self)

    @staticmethod
    def _input_path(path):
        while (path == '') or (not os.path.isfile(path)):
            path = input(f"Input data path: ")
        path_list = (path).split('\\')
        folder = '\\'.join(path_list[:-1])
        name = path_list[-1].split('.')[0]
        return path, folder, name

    @log_wrapp
    def apply(self, func):
        self.d['time'], self.d['x'], self.d['y'] = func(self.d['time'], self.d['x'], self.d['y'])
        return (func.__name__, [])

    @log_wrapp
    def group_filter(self, filter, by='x', column='y'):
        group = self.d.groupby(by=by)[column]
        mask = group.apply(filter).droplevel([0]).sort_index().to_numpy()
        self.d = self.d[mask]
        return (filter.__name__, [])

    @log_wrapp
    def mask_filter(self, filter, **kwargs):
        self.d = filter(self.d, **kwargs)
        return (filter.__name__, kwargs)


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


## Mask filters
def initial_filter(df, x=(-np.inf, np.inf), y=(0, np.inf), time=(0, np.inf)):
    temperature_cond = ((x[0] < df['x']) & (df['x'] < x[1]))
    viscosity_cond = ((y[0] < df['y']) & (df['y'] < y[1]))
    time_cond = ((time[0] < df['time']) & (df['time'] < time[1]))
    return df[temperature_cond & viscosity_cond & time_cond]


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


## Start plot
verbose_colors = {
    'OK': 'g',
    'OK_inner': 'b',
    'image_sweep_check': 'r',
    'combine_check': 'w',
}

exp = Experiment()
exp.load_csv()
exp.set_info(
    compound=input('Compound: '),
    rho=float(input('Rho: ')),
    w=float(input('W mass: ')),
)

fig, ax_v = plt.subplots()
ax_T = ax_v.twinx()
ax_v.scatter(exp.d['time'], exp.d['y'], color='red', marker='.')
ax_T.scatter(exp.d['time'], exp.d['x'], color='blue', marker='.')

fig.canvas.manager.set_window_title('Start plot')
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
ax_v.set_ylabel('Viscosity [cP]', color='red')
plt.show()

## Initial filter plot
while True:
    time_lim = ()
    while len(time_lim) != 2:
        time_lim = input('Time lim (space as delimiter): ')
        time_lim = [float(i) for i in time_lim.split(' ') if '' != i]
        if len(time_lim) == 1: time_lim.append(np.inf)

    y_lim = ()
    while len(y_lim) != 2:
        y_lim = input('Viscosity lim (space as delimiter): ')
        y_lim = [float(i) for i in y_lim.split(' ')]

    exp.mask_filter(initial_filter, time=time_lim, y=y_lim, x=(12, 42))

    plot = exp.copy()
    fig, ax_v = plt.subplots()
    ax_v.set_title(f"{plot.name}: ({plot.info['w']}% mass)")
    ax_T = ax_v.twinx()

    ax_v.scatter(plot.d['time'], plot.d['y'], color='red', marker='.')
    ax_T.scatter(plot.d['time'], plot.d['x'], color='blue', marker='.')

    fig.canvas.manager.set_window_title('Initial filter plot')
    fig.subplots_adjust(
        top=0.9,
        bottom=0.1,
        left=0.1,
        right=0.9,
        hspace=0.2,
        wspace=0.2,
    )
    ax_v.set_ylabel('Viscosity [cP]', color='red')
    ax_v.set_xlabel('Time [s]')
    ax_T.set_ylabel('Temperature [C]', color='blue')
    plt.show()

    if ask_continue(): break

tmp = exp.copy()
# fig.savefig(f'{plot.folder}\Plots\\1{plot.experiment_name}_Initial.jpg',dpi =600)

## Temperature plots
plot = exp.copy()

fig, ax = plt.subplots()
colors = plot.d['Viscosity_verbose'].replace(verbose_colors)
ax.scatter(x=plot.d['x'], y=plot.d['y'], c=colors, s=5)

sns.lineplot(
    ax=ax,
    # data=plot.d,
    x=plot.d["x"],
    y=plot.d["y"],
    estimator='mean',
    errorbar=("sd", 1),
    label='mean',
)
sns.lineplot(
    ax=ax,
    data=plot.d,
    x="x",
    y="y",
    errorbar=('pi', 50),
    estimator="median",
    label='median',
)

fig.canvas.manager.set_window_title('Temperature plots')
fig.subplots_adjust(
    top=0.9,
    bottom=0.1,
    left=0.1,
    right=0.9,
    hspace=0.2,
    wspace=0.2,
)
ax.set_title(f"{plot.name}: ({exp.info['w']}% mass)")
ax.set_xlabel('Temperature [C]')
ax.set_ylabel('Viscosity [cP]')
plt.show()

# fig.savefig(f'{plot.folder}\Plots\\2{plot.experiment_name}_Temperature.jpg',dpi =600)

exp.apply(C_to_K)
exp.apply(nu_D)
exp.apply(linearize)
exp.group_filter(iqr_filter)
exp.apply(delinearize)
# # exp.apply(nu_D)
exp.apply(K_to_C)
print('Filtered')

## Diffusion plot
plot = exp.copy()
fig, ax = plt.subplots()

colors = plot.d['Viscosity_verbose'].replace(verbose_colors)
ax.scatter(x=plot.d['x'], y=plot.d['y'], c=colors, s=5)

sns.lineplot(
    ax=ax,
    data=plot.d,
    x="x",
    y="y",
    estimator='mean',
    errorbar=("sd", 1),
    label='mean',
)
sns.lineplot(
    ax=ax,
    data=plot.d,
    x="x",
    y="y",
    errorbar=('pi', 50),
    estimator="median",
    label='median',
)

fig.canvas.manager.set_window_title('Diffusion plot')
fig.subplots_adjust(
    top=0.9,
    bottom=0.1,
    left=0.1,
    right=0.9,
    hspace=0.2,
    wspace=0.2,
)
ax.set_title(f"{plot.name}: ({plot.info['w']}% mass)")
ax.set_xlabel('Temperature [C]')
ax.set_ylabel('D [m2/s]')
plt.show()

# fig.savefig(f'{plot.folder}\Plots\\3{plot.name}_Diffusion.jpg',dpi =600)

plot = exp.copy()
plot.apply(C_to_K)
plot.apply(linearize)
plot.d


## OLS plot
def regress(data):
    reg = linear_model.LinearRegression(fit_intercept=True)
    X = np.array([data['x']]).T
    Y = np.array(data['y'])
    reg.fit(X, Y)

    w_T = reg.coef_[0]
    w_D = reg.intercept_

    D0 = np.exp(w_D)
    E = -8.314 * w_T

    def TC_func(T, E=E, D0=D0):
        return D0 * np.exp(-E / (8.314*T))

    return D0, E, TC_func


plot = exp.copy()

plot.apply(C_to_K)
plot.apply(linearize)

D0, E, OLS_func = regress(plot.d)
x = np.linspace(13, 42, 100) + 273.15
ols_res = Experiment(
    pd.DataFrame({
        'x': x, 'y': OLS_func(x), 'time': x * 0
    }),
    'interpolated',
)

ols_res.apply(K_to_C)

print(
    f'E  = {E/1000: <7.2f} kJ',
    f'D0 = {D0: <7.2e} m2*s',
    sep='\n',
)

fig, ax = plt.subplots()
ax.set_yscale('log')
ax.scatter(
    plot.d['x'],
    plot.d['y'],
    color='gray',
    marker='.',
)

sns.regplot(
    ax=ax,
    data=plot.d,
    x='x',
    y='y',
    scatter=False,
    truncate=False,
    order=1,
    label=f'E= {E/1000: >8.2f} kJ\nD= {D0: >8.2e} m2/s',
)

fig.canvas.manager.set_window_title('Fast OLS plot')
fig.subplots_adjust(
    top=0.9,
    bottom=0.1,
    left=0.1,
    right=0.9,
    hspace=0.2,
    wspace=0.2,
)
ax.set_title(f"{plot.name}: ({plot.info['w']}% mass)")
ax.set_xlabel('Temperature [C]')
ax.set_ylabel('D [m2/s]')
plt.legend()
plt.show()

# fig.savefig(f'{plot.folder}\Plots\\4{plot.name}_OLS.jpg',dpi =600)

fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4, 5], np.array([1, 2, 3, 4, 5]) / 100)
ax.set_yscale('log')

## Comparation plot
plot = exp.copy()

fig, ax = plt.subplots()
colors = plot.d['Viscosity_verbose'].replace(verbose_colors)
ax.scatter(plot.d['x'], plot.d['y'], color=colors, marker='.', label='Data')

ax.plot(
    ols_res.d['x'],
    ols_res.d['y'],
    color='black',
    label=f'OLS: \nE= {E/1000: >8.2f} kJ\nD= {D0: >8.2e} m2/s',
)

fig.canvas.manager.set_window_title('Comparation plot')
fig.subplots_adjust(
    top=0.9,
    bottom=0.1,
    left=0.1,
    right=0.9,
    hspace=0.2,
    wspace=0.2,
)

ax.set_title(f"{plot.name}: ({plot.info['w']}% mass)")
ax.set_xlabel('Temperature [C]')
ax.set_ylabel('D [m2/s]')
ax.legend()
plt.show()

# fig.savefig(f'{plot.folder}\Plots\\5{plot.name}_Comparation.jpg',dpi =600)

## Regression
plot = exp.copy()
plot.apply(C_to_K)
plot.apply(linearize)

df = plot.d
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

exp.set_info(D0=D0, d_D0=dD0, d_E=dE, f_statistic=result.fvalue, r2=result.rsquared)
print(
    f"Constants {plot.name} ({plot.info['w']}% mass):",
    f'E  = {E: >7.3e} ± {dE: <3.2e} J',
    f'D0 = {D0: >7.3e} ± {dD0: <3.2e} m2/s',
    sep='\n',
)
print(result.summary2())

exp.save_hdf5()
