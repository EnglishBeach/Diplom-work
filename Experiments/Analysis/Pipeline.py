import numpy as np
import pandas as pd
from mylibs import tools, functions as f
import os

for i in range(9):

    exp = tools.Experiment()
    path = f'D:\Works\Diplom-work\Experiments\OCM_viscosity\OCM{i}'
    file = [file for file in os.listdir(path) if '.hdf5' in file][0]
    exp.load_hdf5()
    info, result, func = tools.regress(exp)
    exp.set_info(**info)
    os.remove(f'{path}\{file}')
    exp.save_hdf5(f'{path}\{file}')
