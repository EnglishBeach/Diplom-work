import numpy as np
import pandas as pd
from mylibs import tools,plots, functions as funcs


exp = tools.Experiment()
exp.load_csv(*tools._split_path())
tools.temporal_plot(exp)

exp = tools.configurate_data(exp)

plots.temperature_plot(
    exp,
    title='Viscosity',
    xlabel='Temperature [C]',
    ylabel='Viscosity [cP]',
    interactive=True,
)


exp.apply(funcs.C_to_K)
exp.apply(funcs.nu_D)
exp.apply(funcs.linearize)
exp.group_filter(funcs.iqr_filter)
exp.apply(funcs.delinearize)
exp.apply(funcs.K_to_C)
print('Filtered')

plots.temperature_plot(
    exp,
    title='Diffusion',
    xlabel='Temperature [C]',
    ylabel='D [m2/s]',
    interactive=True,
)

info, result, func = tools.regress(exp)
exp.set_info(**info)
x = np.linspace(13, 42, 100) + 273.15
ols_res = tools.Experiment(
    pd.DataFrame({
        'x': x, 'y': func(x), 'time': x * 0
    }),
    'interpolated',
)

ols_res.apply(funcs.K_to_C)

exp.info

tmp=exp.copy()

exp2= exp.copy()
ols_res2 = ols_res.copy()
exp2.apply(funcs.C_to_K)
ols_res2.apply(funcs.C_to_K)
exp2.apply(funcs.linearize)
ols_res2.apply(funcs.linearize)

plots.comparation_plot(
    exp2,
    ols_res2,
    title='OLS_Linear',
    xlabel='Temperature',
    ylabel='D',
    interactive=True,
)

plots.comparation_plot(
    exp,
    ols_res,
    title='OLS_Diffusion',
    xlabel='Temperature [C]',
    ylabel='D [m2/s]',
    interactive=True,
)

exp.save_hdf5()
