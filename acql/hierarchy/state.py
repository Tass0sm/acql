"""A state struct for options framework."""

from brax import base
from flax import struct
import jax


@struct.dataclass
class OptionState(base.Base):
  """Environment state for training and inference."""

  option: jax.Array
  option_beta: jax.Array
