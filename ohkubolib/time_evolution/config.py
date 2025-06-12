from functools import cached_property

import numpy as np

from ohkubolib.datamodel import datamodel, field
from ohkubolib.datamodel.serde import NdArryaListSerde


@datamodel
class TEConfig:
    x0: np.ndarray = field(serde=NdArryaListSerde)
    dt_obs: float
    t_f: float
    dt: float = 0
    t_trans: float = 0

    def __post_init__(self) -> None:
        step = int(self.t_f / self.dt_obs)
        if abs(self.dt_obs*step - self.t_f) > 1e-6:
            raise ValueError('t_f is not multiple of dt_obs')

        if self.dt > 0:
            step = int(self.dt_obs / self.dt)
            if abs(self.dt*step - self.dt_obs) > 1e-6:
                raise ValueError('dt_obs is not multiple of dt')

        if self.t_trans > 0:
            step = int(self.t_trans / self.dt_obs)
            if abs(self.dt_obs*step - self.t_trans) > 1e-6:
                raise ValueError('t_trans is not multiple of dt_obs')

    @cached_property
    def time_list(self) -> 'TimeList':
        dt = self.dt_obs if self.dt == 0 else self.dt
        num_obs=int(self.t_f / self.dt_obs) + 1
        num_dt=int(self.t_f / dt) + 1

        step = int(self.dt_obs / dt)
        obs_indices = set(range(0, num_dt, step))

        return TimeList(
            obs=np.linspace(self.t_trans, self.t_f + self.t_trans, num_obs),
            dt=np.linspace(self.t_trans, self.t_f + self.t_trans, num_dt),
            obs_indices=obs_indices,
            num_obs=num_obs,
        )


@datamodel
class TimeList:
    obs: np.ndarray
    num_obs: int

    dt: np.ndarray
#    transient: np.ndarray
    obs_indices: set[int]

    def is_obs_indices(self, i: int) -> bool:
        """
        Returns whether `i` is observation index or not.
        """
        return i in set(self.obs_indices)
