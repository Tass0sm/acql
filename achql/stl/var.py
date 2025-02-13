import jax.numpy as jnp


class Var:
    def __init__(self, name, idx: int = 0, dim: int = 1, index = None, **kwargs):
        self.name = name
        self.idx = 0
        self.dim = dim
        self.index = index

        for sub_name, sub_index in kwargs.items():
            self._register_sub_var(sub_name, sub_index)

    def _register_sub_var(self, sub_name, sub_index):
        if isinstance(sub_index, int):
            sub_index = (sub_index, sub_index+1)
            sub_dim = 1
        elif isinstance(sub_index, tuple):
            sub_dim = sub_index[1] - sub_index[0]
        else:
            raise NotImplementedError

        setattr(self, sub_name, Var(self.name, self.idx, dim=sub_dim, index=sub_index))

    def get_value(self, env: dict):
        x = env[self.idx]

        if self.index is not None:
            x = x[..., self.index[0]:self.index[1]]

        # if x.shape[-1] != self.dim:
        #     raise ValueError(f'Variable {self.name} expected a value with dimension {self.dim}, but x had shape {x.shape} with dimension {x.shape[-1]}.')

        # if x.ndim == 2:
        #     x = jnp.expand_dims(x, 1)

        return x
