from abc import ABCMeta, abstractmethod


class Expression(metaclass=ABCMeta):
    '''
    NOTE: All the inputs are assumed to be TIME REVERSED. The outputs are also TIME REVERSED
    All STL formulas have the following functions:
    robustness_trace: Computes the robustness trace.
    robustness: Computes the robustness value of the trace
    eval_trace: Computes the robustness trace and returns True in each entry if the robustness value is > 0
    eval: Computes the robustness value and returns True if the robustness value is > 0
    forward: The forward function of this STL_formula PyTorch module (default to the robustness_trace function)

    Inputs to these functions:
    trace: the input signal assumed to be TIME REVERSED. If the formula has two subformulas (e.g., And), then it is a tuple of the two inputs. An input can be a tensor of size [batch_size, time_dim,...], or an Expression with a .value (Tensor) associated with the expression.
    pscale: predicate scale. Default: 1
    scale: scale for the max/min function.  Default: -1
    keepdim: Output shape is the same as the input tensor shapes. Default: True
    agm: Use arithmetic-geometric mean. (In progress.) Default: False
    distributed: Use the distributed mean. Default: False
    '''

    # def __init__(self):
    #     super().__init__()

    # def _reverse_env_inputs(self, env):
    #     reversed_input_env = {}
    #     for k, v in env.items():
    #         assert v.ndim == 3, "Inputs to STL expression must have shape (batch x time x signal_dim)"
    #         reversed_input_env[k] = torch.flip(v, [1])
    #     return reversed_input_env

    # @abstractmethod
    # def robustness_trace(
    #         self,
    #         env,
    #         # time=0,
    #         # pscale=1,
    #         # scale=-1,
    #         # keepdim=True,
    #         # agm=False,
    #         # distributed=False,
    #         # **kwargs
    # ):
    #     raise NotImplementedError

    # @abstractmethod
    # def robustness(
    #         self,
    #         env,
    #         # time=0,
    #         # pscale=1,
    #         # scale=-1,
    #         # keepdim=True,
    #         # agm=False,
    #         # distributed=False,
    #         # **kwargs
    # ):
    #     raise NotImplementedError

    # @abstractmethod
    # def eval_trace(
    #         self,
    #         env,
    #         # pscale=1,
    #         # scale=-1,
    #         # keepdim=True,
    #         # agm=False,
    #         # distributed=False,
    #         # **kwargs
    # ):
    #     raise NotImplementedError

    # @abstractmethod
    # def eval(
    #         self,
    #         env,
    #         # time=0,
    #         # pscale=1,
    #         # scale=-1,
    #         # keepdim=True,
    #         # agm=False,
    #         # distributed=False,
    #         # **kwargs
    # ):
    #     raise NotImplementedError
