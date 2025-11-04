from abc import ABC, abstractmethod

from achql.stl import Expression, Var


class TaskBase(ABC):
    def __init__(self, path_max_len: int, time_limit: int, backend: str):
        self.path_max_len = path_max_len
        self.time_limit = time_limit
        self.env = self._build_env(backend)

        # Stateful and ugly but oh well
        self._create_vars()

        # high level constraints
        self.hi_spec = self._build_hi_spec(self.wp_var)

        # low level constraints
        self.lo_spec = self._build_lo_spec(self.obs_var)

    @abstractmethod
    def _build_env(self):
        raise NotImplementedError()

    @abstractmethod
    def _create_vars(self):
        raise NotImplementedError()

    @abstractmethod
    def _build_hi_spec(self, wp_var: Var) -> Expression:
        raise NotImplementedError()

    @abstractmethod
    def _build_lo_spec(self, obs_var: Var) -> Expression:
        raise NotImplementedError()
