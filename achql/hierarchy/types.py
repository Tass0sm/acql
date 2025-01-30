from typing import Protocol, Tuple

from brax.training.types import Observation, PRNGKey, Extra, Action

from achql.hierarchy.state import OptionState


class HierarchicalPolicy(Protocol):

  def __call__(
      self,
      observation: Observation,
      option_state: OptionState,
      key: PRNGKey,
  ) -> Tuple[Action, Extra]:
    pass
