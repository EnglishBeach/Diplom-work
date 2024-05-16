import copy
import os
from enum import Enum
from pathlib import Path
from typing import Optional, NamedTuple
from pydantic import BaseModel

import pandas as pd

pd.set_option("mode.chained_assignment", None)
T_ZERO = 273.15


class Mols(Enum):
    butanol = 'BUT'
    ocm = 'OCM'
    dmeg = 'DME'
    peta = 'PET'


class Experiment:
    def __init__(
        self,
        name: Optional[str] = None,
        d: Optional[pd.DataFrame] = None,
        source: Optional[Path] = None,
        info: dict = {},
        lims: dict = {},
    ) -> None:

        self.name = name
        self.d = d
        self.source = source
        self.info = info
        self.lims = lims

    def __repr__(self) -> str:
        return f"<Experiment {self.name} {self.source.stem}>"

    def read_csv(self, path):
        path = Path(path)
        self.d = pd.read_csv(path)
        self.source = path

    @classmethod
    def from_hdf5(cls, path):
        exp = cls()
        path = Path(path)
        exp.source = path
        with pd.HDFStore(path) as file:
            exp.d = file.get("data")
            exp.info = file.get("info").to_dict()
            exp.lims = file.get('lims').to_dict()
        return exp

    def save_hdf5(self, folder=None):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        file_path = folder / f"{self.name}.hdf5"
        with pd.HDFStore(file_path) as file:
            file.put("data", self.d)
            file.put('info', pd.Series(self.info))
            file.put('lims', pd.Series(self.lims))

    def copy(self):
        return copy.deepcopy(self)

    def group_filter(self, filter, by, column):
        temp = copy.deepcopy(self)
        group = self.d.groupby(by=by)[column]
        mask = group.apply(filter).droplevel([0]).sort_index().to_numpy()
        temp.d = self.d[mask]

        temp.info.append({filter.__name__: None})
        return temp
