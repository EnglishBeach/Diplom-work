import copy
import os
from enum import Enum
from pathlib import Path
from typing import Optional, NamedTuple

import pandas as pd
from pydantic import BaseModel

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


class Experiment(NamedTuple):
    name: str = ""
    d: pd.DataFrame = None
    folder: Optional[Path] = None
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
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        file_path = folder / f"{self.name}.hdf5"
        with pd.HDFStore(file_path) as file:
            file.put("data", self.d)
            file.get_storer("data").attrs.log = self.log
            file.get_storer("data").attrs.info = self.info

    def apply(self, func, x, y, time='time'):
        temp = self.copy()
        (
            temp.d[time],
            temp.d[x],
            temp.d[y],
            comment,
        ) = func(
            self.d[time],
            self.d[x],
            self.d[y],
        )
        temp.log.append({func.__name__: comment})
        return temp

    def group_filter(self, filter, by, column):
        temp = copy.deepcopy(self)
        group = self.d.groupby(by=by)[column]
        mask = group.apply(filter).droplevel([0]).sort_index().to_numpy()
        temp.d = self.d[mask]

        temp.log.append({filter.__name__: None})
        return temp
