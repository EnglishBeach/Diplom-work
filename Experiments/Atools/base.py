import copy
import os
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydantic import BaseModel

from . import functions

pd.set_option("mode.chained_assignment", None)


class Mols(Enum):
    butanol = 'BUT'
    ocm = 'OCM'
    dmag = 'DMA'
    peta = 'PET'


def input_path(path=""):
    while (path == "") or (not os.path.isfile(path)):
        path = input(f"Input data path: ")
    return path


class Experiment(BaseModel):

    class Config:
        arbitrary_types_allowed = True

    name: str = ""
    d: pd.DataFrame = None
    folder: Path
    info: dict = {}
    log: list = []

    @classmethod
    def load_csv(cls, path):
        path = Path(path)

        df = pd.read_csv(path).rename(columns={"Temperature": "x", "Viscosity": "y"})
        return Experiment(
            name=path.parent.stem,
            d=df,
            folder=path,
        )

    @classmethod
    def load_hdf5(cls, path):
        path = Path(path)
        with pd.HDFStore(path) as file:
            data = file["data"]
            info = file.get_storer("data").attrs.info
            log = file.get_storer("data").attrs.log
        return Experiment(
            name=path.parent.stem,
            d=data,
            folder=path,
            info=info,
            log=log,
        )

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return f"<Experiment {self.name} {self.info}>"

    def set_info(self, **info):
        self.info.update(info)

    def save_hdf5(self, folder=None):
        path = folder if folder is not None else self.folder
        assert path is not None, "Path not define"
        file_path = f"{path}\{self.name}.hdf5"
        with pd.HDFStore(file_path) as file:
            file.put("data", self.d)
            file.get_storer("data").attrs.log = self.log
            file.get_storer("data").attrs.info = self.info

    def apply(self, func):
        temp = self.copy()
        (
            temp.d["time"],
            temp.d["x"],
            temp.d["y"],
            comment,
        ) = func(
            self.d["time"],
            self.d["x"],
            self.d["y"],
        )
        temp.log.append({func.__name__: comment})
        return temp

    def group_filter(self, filter, by="x", column="y"):
        temp = copy.deepcopy(self)
        group = self.d.groupby(by=by)[column]
        mask = group.apply(filter).droplevel([0]).sort_index().to_numpy()
        temp.d = self.d[mask]

        temp.log.append({filter.__name__: None})
        return temp

