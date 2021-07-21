from pathlib import Path
from typing import Any, List

import numpy as np
from pydantic import BaseModel


class Measurement(BaseModel):
    data: Any
    label: str


class Trial(BaseModel):
    measurements: np.ndarray
    response: int

    class Config:
        arbitrary_types_allowed = True


class Session(BaseModel):
    spks: np.ndarray
    response: np.ndarray

    class Config:
        arbitrary_types_allowed = True

    def nancheck(self):
        if np.isnan(self.spks).any():
            print("SESSION CONTAINS NANS!")
        else:
            print("NO NANS")


class Dataset(BaseModel):
    sessions: List[Session]

    def save(self, path: Path):
        #  for session in self.sessions:
        #      session.nancheck()
        np.save(path, [s.dict() for s in self.sessions])
