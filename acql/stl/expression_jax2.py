"""Defines a tree structure for constructing and evaluating STL formulae"""
from abc import ABC, abstractmethod
from functools import partial
from typing import Optional, Callable, List

import jax
from jax.nn import logsumexp
import jax.numpy as jnp

from .signal import max1d

from .expression import Expression
from .var import Var


class STLFormula(Expression):
    """STLFormula defines a formula in bounded-time signal temporal logic (STL) using a
    tree data structure.

    STLFormula is an abstract base class that defines the interface all STL formulae
    should implement.

    NOTE: Assumes that signals are piecewise affine for the purposes of interpolating
    """

    def __init__(self):
        super(STLFormula, self).__init__()

    @property
    @abstractmethod
    def children(self):
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, env: dict[str: jnp.ndarray], smoothing: float = 100.0) -> jnp.ndarray:
        """Evaluates this formula on the given signal, returning its robustness trace.

        The robustness trace is an array of the same length as s but with only one value
        dimension. Each sample represents the robustness of the subsignal of s starting
        at that sample and continuing to the end of s.
        I.e. if the robustness trace has sample (t_i, r_i), then the subsignal s[i:] has
        robustness r_i with respect to this formula.

        If robustness is positive, then the formula is satisfied. If the robustness
        is negative, then the formula is not satisfied.

        args:
            s: the signal upon which this formula is evaluated.
            smoothing: the parameter determining how much to smooth non-continuous
                elements of this formula (e.g. mins and maxes). Uses the log-sum-exp
                smoothing for max and min.
        returns:
            single-element array of the robustness value for this formula and the given
            signal.
        """
        raise NotImplementedError()


class STLPredicate(STLFormula):
    """Represents an STL predicate (function from signal to reals)"""

    def __init__(
            self,
            var: Var,
            predicate: Callable[[jnp.ndarray], jnp.ndarray],
            lower_bound: float,
            default_params: Optional[jnp.ndarray] = None,
            info: dict = {},
    ):
        """Initialize an STL formula for a predicate.

        args:
            predicate: a callable that takes a single sample and returns a scalar value
            lower_bound: the predicate is satisfied if its value is greater than this
                bound.
        """
        super(STLPredicate, self).__init__()

        self.var = var
        self.default_params = default_params
        self.predicate = predicate
        self.lower_bound = lower_bound
        self.info = info

    @property
    def children(self):
        return []

    @partial(jax.jit, static_argnames=["self"])
    def __call__(self, env: dict[str: jnp.ndarray], smoothing: float = 100.0, end_time: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Evaluates this formula on the given signal, returning its robustness trace.

        The robustness trace is an array of the same length as s where each
        sample represents the robustness of the subsignal of s starting at that sample
        and continuing to the end of s. I.e. if the robustness trace has sample
        (t_i, r_i), then the subsignal s[i:] has robustness r_i with respect to this
        formula.

        If robustness is positive, then the formula is satisfied. If the robustness
        is negative, then the formula is not satisfied.

        A predicate has robustness value equal to the scalar result of its function
        minus the lower bound. The complexity of this call is linear in the length
        of the signal.

        args:
            s: the signal upon which this formula is evaluated.
            smoothing: the parameter determining how much to smooth non-continuous
                elements of this formula (e.g. mins and maxes). Uses the log-sum-exp
                smoothing for max and min.
        returns:
            The robustness trace for this formula and the given signal.
        """

        # FIXME: Currently hard coding params input as being under key "1"

        # if this predicate has been registered in a spot automaton, get this id
        # and lookup if params have been passed
        if "ap_i" in self.info:
            ap_id = self.info["ap_i"]
            params = env.get(32 + ap_id, self.default_params)
        else:
            params = self.default_params

        s = self.var.get_value(env)

        # from IPython.core.debugger import set_trace; set_trace()

        # predicate :: [a], x -> a
        # predicate'' :: [t, a], x -> [t]

        # Apply the predicate to each sampled value
        mu_x = jax.vmap(self.predicate, in_axes=(0, None))(s, params)

        # Compute the robustness over all subsignals
        robustness = mu_x - self.lower_bound

        return robustness


class STLNegation(STLFormula):
    """Represents an STL negation"""

    def __init__(self, child: STLFormula):
        """Initialize an STL formula for a negation.

        args:
            child: the STLFormula to negate
        """
        super(STLNegation, self).__init__()

        self.child = child

    @property
    def children(self):
        return [self.child]

    @partial(jax.jit, static_argnames=["self"])
    def __call__(self, env: dict[str: jnp.ndarray], smoothing: float = 100.0, end_time: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Evaluates this formula on the given signal, returning its robustness trace.

        The robustness trace is an array of the same length as s where each
        sample represents the robustness of the subsignal of s starting at that sample
        and continuing to the end of s. I.e. if the robustness trace has sample
        (t_i, r_i), then the subsignal s[i:] has robustness r_i with respect to this
        formula.

        If robustness is positive, then the formula is satisfied. If the robustness
        is negative, then the formula is not satisfied.

        A negation has robustness value equal to the negative of its child's robustness
        value

        args:
            s: the signal upon which this formula is evaluated.
            smoothing: the parameter determining how much to smooth non-continuous
                elements of this formula (e.g. mins and maxes). Uses the log-sum-exp
                smoothing for max and min.
        returns:
            the robustness trace for this formula and the given signal.
        """
        # Negate the robustness value
        return -self.child(env, smoothing, end_time)


def soft_max(signals, smoothing):
    stacked_signals = jnp.stack(signals, axis=0)
    smoothed_max = 1 / smoothing * logsumexp(smoothing * stacked_signals, axis=0)
    return smoothed_max


class STLAnd(STLFormula):
    """Represents an STL conjunction"""

    def __init__(
        self, child_1: STLFormula, child_2: STLFormula
    ):
        """Initialize an STL formula for a conjunction

        args:
            child_1, child_2: the two formulae to and together.
        """
        super(STLAnd, self).__init__()

        self._children = [child_1, child_2]

    @property
    def children(self):
        return self._children

    @partial(jax.jit, static_argnames=["self"])
    def __call__(self, env: dict[str: jnp.ndarray], smoothing: float = 100.0, end_time: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Evaluates this formula on the given signal, returning its robustness trace.

        The robustness trace is an array of the same length as s where each
        sample represents the robustness of the subsignal of s starting at that sample
        and continuing to the end of s. I.e. if the robustness trace has sample
        (t_i, r_i), then the subsignal s[i:] has robustness r_i with respect to this
        formula.

        If robustness is positive, then the formula is satisfied. If the robustness
        is negative, then the formula is not satisfied.

        A conjunction has robustness value equal to the minimum of its children's
        robustness values

        args:
            s: the signal upon which this formula is evaluated.
            smoothing: the parameter determining how much to smooth non-continuous
                elements of this formula (e.g. mins and maxes). Uses the log-sum-exp
                smoothing for max and min.
        returns:
            the robustness trace for this formula and the given signal.
        """
        # Get children's robustness values
        child1_r = self._children[0](env, smoothing, end_time)
        child2_r = self._children[1](env, smoothing, end_time)

        # Use the trick where min(x, y) = -max(-x, -y)
        r = -soft_max([-child1_r, -child2_r], smoothing)

        return r


class STLOr(STLFormula):
    """Represents an STL disjunction"""

    def __init__(
        self, child_1: STLFormula, child_2: STLFormula
    ):
        """Initialize an STL formula for a disjunction

        args:
            child_1, child_2: the two formulae to and together.
        """
        super(STLOr, self).__init__()

        self._children = [child_1, child_2]

    @property
    def children(self):
        return self._children

    @partial(jax.jit, static_argnames=["self"])
    def __call__(self, env: dict[str: jnp.ndarray], smoothing: float = 100.0, end_time: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Evaluates this formula on the given signal, returning its robustness trace.

        The robustness trace is an array of the same length as s where each
        sample represents the robustness of the subsignal of s starting at that sample
        and continuing to the end of s. I.e. if the robustness trace has sample
        (t_i, r_i), then the subsignal s[i:] has robustness r_i with respect to this
        formula.

        If robustness is positive, then the formula is satisfied. If the robustness
        is negative, then the formula is not satisfied.

        A disjunction has robustness value equal to the maximum of its children's
        robustness values

        args:
            s: the signal upon which this formula is evaluated.
            smoothing: the parameter determining how much to smooth non-continuous
                elements of this formula (e.g. mins and maxes). Uses the log-sum-exp
                smoothing for max and min.
        returns:
            the robustness trace for this formula and the given signal.
        """
        # Get children's robustness values
        child1_r = self._children[0](env, smoothing, end_time)
        child2_r = self._children[1](env, smoothing, end_time)

        # Compute the maximum
        r = soft_max([child1_r, child2_r], smoothing)

        return r


class STLDisjunction(STLFormula):
    """Represents an STL disjunction"""

    def __init__(
        self, children: List[STLFormula]
    ):
        """Initialize an STL formula for a disjunction

        args:
            child_1, child_2: the two formulae to and together.
        """
        super(STLDisjunction, self).__init__()

        self._children = children

    @property
    def children(self):
        return self._children

    @partial(jax.jit, static_argnames=["self"])
    def __call__(self, env: dict[str: jnp.ndarray], smoothing: float = 100.0, end_time: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Evaluates this formula on the given signal, returning its robustness trace.

        The robustness trace is an array of the same length as s where each
        sample represents the robustness of the subsignal of s starting at that sample
        and continuing to the end of s. I.e. if the robustness trace has sample
        (t_i, r_i), then the subsignal s[i:] has robustness r_i with respect to this
        formula.

        If robustness is positive, then the formula is satisfied. If the robustness
        is negative, then the formula is not satisfied.

        A disjunction has robustness value equal to the maximum of its children's
        robustness values

        args:
            s: the signal upon which this formula is evaluated.
            smoothing: the parameter determining how much to smooth non-continuous
                elements of this formula (e.g. mins and maxes). Uses the log-sum-exp
                smoothing for max and min.
        returns:
            the robustness trace for this formula and the given signal.
        """
        # Get children's robustness values
        children_rs = [c(env, smoothing, end_time) for c in self._children]

        # Compute the maximum
        r = soft_max(children_rs, smoothing)

        return r


class STLImplies(STLFormula):
    """Represents an STL implication"""

    def __init__(
        self, premise: STLFormula, conclusion: STLFormula, interpolate: bool = True
    ):
        """Initialize an STL formula for a implication

        args:
            premise: an STL formula
            conclusion: an STL formula
            interpolate: if True, will look for intersections between child traces
        """
        super(STLImplies, self).__init__()

        self.premise = premise
        self.conclusion = conclusion
        self.interpolate = interpolate

    @property
    def children(self):
        return [self.premise, self.conclusion]

    @partial(jax.jit, static_argnames=["self"])
    def __call__(self, env: dict[str: jnp.ndarray], smoothing: float = 100.0) -> jnp.ndarray:
        """Evaluates this formula on the given signal, returning its robustness trace.

        The robustness trace is an array of the same length as s where each
        sample represents the robustness of the subsignal of s starting at that sample
        and continuing to the end of s. I.e. if the robustness trace has sample
        (t_i, r_i), then the subsignal s[i:] has robustness r_i with respect to this
        formula.

        If robustness is positive, then the formula is satisfied. If the robustness
        is negative, then the formula is not satisfied.

        A implication has robustness value equal to the maximum of the negative of
        the premise's robustness and the conclusions' robustness. This relates to
        the fact that A implies B is equivalent to (not A) or B.

        args:
            s: the signal upon which this formula is evaluated.
            smoothing: the parameter determining how much to smooth non-continuous
                elements of this formula (e.g. mins and maxes). Uses the log-sum-exp
                smoothing for max and min.
        returns:
            the robustness trace for this formula and the given signal.
        """
        # Get robustness of premise and conclusion
        r_premise = self.premise(s, smoothing)
        r_conclusion = self.conclusion(s, smoothing)

        # Compute the smoothed maximum
        r = max1d(
            r_premise.at[1:].multiply(-1), r_conclusion, smoothing, self.interpolate
        )

        return r


class STLNext(STLFormula):
    """Represents an STL Next Expression"""

    def __init__(
        self, child: STLFormula
    ):
        """Initialize an STL formula for a disjunction

        args:
            child_1, child_2: the two formulae to and together.
            interpolate: if True, will look for intersections between child traces
        """
        super(STLNext, self).__init__()

        self.child = child

    @property
    def children(self):
        return [self.child]

    @partial(jax.jit, static_argnames=["self"])
    def __call__(self, env: dict[str: jnp.ndarray], smoothing: float = 100.0, end_time: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Evaluates this formula on the given signal, returning its robustness trace.

        The robustness trace is an array of the same length as s where each
        sample represents the robustness of the subsignal of s starting at that sample
        and continuing to the end of s. I.e. if the robustness trace has sample
        (t_i, r_i), then the subsignal s[i:] has robustness r_i with respect to this
        formula.

        If robustness is positive, then the formula is satisfied. If the robustness
        is negative, then the formula is not satisfied.

        A disjunction has robustness value equal to the maximum of its children's
        robustness values

        args:
            s: the signal upon which this formula is evaluated.
            smoothing: the parameter determining how much to smooth non-continuous
                elements of this formula (e.g. mins and maxes). Uses the log-sum-exp
                smoothing for max and min.
        returns:
            the robustness trace for this formula and the given signal.
        """

        # Get the robustness of the child formula
        child_trace = self.child(env, smoothing, end_time)

        zero = jnp.zeros_like(child_trace[-1:])

        # Shift by one and append zero
        next_trace = jnp.concatenate([child_trace[1:], zero], axis=0)

        return next_trace


@jax.jit
def accumulate_max(max_so_far, next_element, smoothing):
    comparands = jnp.stack((max_so_far, next_element), axis=-1)
    max_so_far = 1 / smoothing * logsumexp(smoothing * comparands, axis=-1)
    return max_so_far, max_so_far


class STLUntimedEventually(STLFormula):
    """Represents an STL temporal operator for eventually without a time bound"""

    def __init__(self, child: STLFormula):
        """Initialize an STL formula for an untimed eventually. This formula is satisfied
        for a given signal if its child is satisfied at some point in the future

        args:
            child: an STL formula that should eventually hold
        """
        super(STLUntimedEventually, self).__init__()

        self.child = child

    @property
    def children(self):
        return [self.child]

    @partial(jax.jit, static_argnames=["self"])
    def __call__(self, env: dict[str: jnp.ndarray], smoothing: float = 100.0, end_time: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Evaluates this formula on the given signal, returning its robustness trace.

        The robustness trace is an array of the same length as s where each
        sample represents the robustness of the subsignal of s starting at that sample
        and continuing to the end of s. I.e. if the robustness trace has sample
        (t_i, r_i), then the subsignal s[i:] has robustness r_i with respect to this
        formula.

        If robustness is positive, then the formula is satisfied. If the robustness
        is negative, then the formula is not satisfied.

        An untimed eventually operator has robustness value at any time equal to the
        maximum of its child's robustness over the rest of the signal

        args:
            s: the signal upon which this formula is evaluated.
            smoothing: the parameter determining how much to smooth non-continuous
                elements of this formula (e.g. mins and maxes). Uses the log-sum-exp
                smoothing for max and min.
        returns:
            the robustness trace for this formula and the given signal.
        """

        # Get the robustness of the child formula
        child_robustness = self.child(env, smoothing, end_time)

        t, = child_robustness.shape

        # mask the child expression's robustness past end_time
        if end_time is not None:
            idx = jnp.arange(t)
            child_robustness = jnp.where(
                idx < end_time,
                child_robustness,
                -100
            )

        # Get the robustness trace as the accumulated maximum (backwards from the
        # end of the array). See Donze et al. "Efficient Robust Monitoring for STL"
        f = lambda carry, x: accumulate_max(carry, x, smoothing)
        _, eventually_robustness = jax.lax.scan(
            jax.jit(f), -jnp.inf, child_robustness.T, reverse=True
        )

        return eventually_robustness.T


class STLTimedEventually(STLFormula):
    """Represents an STL temporal operator for eventually with a time bound"""

    def __init__(self, child: STLFormula, t_start: float, t_end: float):
        """Initialize an STL formula for an untimed eventually. This formula is satisfied
        for a given signal if its child is satisfied at some point in the future

        args:
            child: an STL formula that should eventually hold
            t_start: start point (inclusive) of interval
            t_end: end point (inclusive) of interval
        """
        super(STLTimedEventually, self).__init__()

        self.child = child

        if t_start < 0.0 or t_end < t_start:
            raise ValueError(
                "Start time must be positive and end time must be after start"
            )

        self.t_start = t_start
        self.t_end = t_end

    @property
    def children(self):
        return [self.child]

    @partial(jax.jit, static_argnames=["self"])
    def __call__(self, env: dict[str: jnp.ndarray], smoothing: float = 100.0) -> jnp.ndarray:
        """Evaluates this formula on the given signal, returning its robustness trace.

        The robustness trace is an array of the same length as s where each
        sample represents the robustness of the subsignal of s starting at that sample
        and continuing to the end of s. I.e. if the robustness trace has sample
        (t_i, r_i), then the subsignal s[i:] has robustness r_i with respect to this
        formula.

        If robustness is positive, then the formula is satisfied. If the robustness
        is negative, then the formula is not satisfied.

        A timed eventually operator has robustness value at any time equal to the
        maximum of its child's robustness over the interval

        args:
            s: the signal upon which this formula is evaluated.
            smoothing: the parameter determining how much to smooth non-continuous
                elements of this formula (e.g. mins and maxes). Uses the log-sum-exp
                smoothing for max and min.
        returns:
            the robustness trace for this formula and the given signal.
        """
        # Get the robustness of the child formula
        child_r = self.child(s, smoothing)

        # Define a function to get the soft max of child_r in a window between
        # t_start and t_end from some given time t
        @jax.jit
        def masked_soft_max(t):
            # This mask is a smoothed indicator for the interval t + [t_start, t_end]
            mask = jnp.logical_and(
                child_r[0] >= t + self.t_start,
                child_r[0] <= t + self.t_end,
            )

            # Mask out the child's robustness to send all values outside the interval
            # to -inf (so they don't affect the maximum)
            masked_robustness = jnp.where(mask, child_r[1], -jnp.inf)

            # Get the timestamps at the start and end of the interval so we can
            # interpolate
            i_start, i_end = jnp.nonzero(jnp.diff(mask), size=2, fill_value=-1)[0]
            t_start, t_end = child_r[0, i_start], child_r[0, i_end]

            # Interpolate within the timesteps at the start and end of the interval
            dt_start = child_r[0, i_start + 1] - t_start
            dt_end = child_r[0, i_end + 1] - t_end

            x_start = (self.t_start + t - t_start) / dt_start * child_r[1, i_start + 1]
            x_start += (1 - (self.t_start + t - t_start) / dt_start) * child_r[
                1, i_start
            ]
            x_end = (self.t_end + t - t_end) / dt_end * child_r[1, i_end + 1]
            x_end += (1 - (self.t_end + t - t_end) / dt_end) * child_r[1, i_end]

            # Pad the interpolated values onto the end
            masked_robustness = jnp.concatenate(
                (
                    masked_robustness,
                    jnp.array(x_start).reshape(-1),
                    jnp.array(x_end).reshape(-1),
                )
            )

            # Run this through a smoothed max
            return 1 / smoothing * logsumexp(smoothing * masked_robustness)

        # vmap this masked maximum function over the time range we're interested in
        robustness = jax.vmap(masked_soft_max)(child_r[0])

        return jnp.vstack((child_r[0], robustness))


class STLUntimedUntil(STLFormula):
    """Represents an STL temporal operator for until without a time bound"""

    def __init__(
        self, invariant: STLFormula, release: STLFormula, interpolate: bool = True
    ):
        """Initialize an STL formula for an untimed until. This formula is satisfied
        for a given signal if invariant is satisfied at all times until release is
        satisfied

        args:
            invariant: an STL formula that should hold until release holds
            release: an STL formula that should hold eventually
            interpolate: if True, will look for intersections between child traces
        """
        super(STLUntimedUntil, self).__init__()

        self.invariant = invariant
        self.release = release
        self.interpolate = interpolate

    @property
    def children(self):
        return [self.invariant, self.release]

    @partial(jax.jit, static_argnames=["self"])
    def __call__(self, env: dict[str: jnp.ndarray], smoothing: float = 100.0) -> jnp.ndarray:
        """Evaluates this formula on the given signal, returning its robustness trace.

        The robustness trace is an array of the same length as s where each
        sample represents the robustness of the subsignal of s starting at that sample
        and continuing to the end of s. I.e. if the robustness trace has sample
        (t_i, r_i), then the subsignal s[i:] has robustness r_i with respect to this
        formula.

        If robustness is positive, then the formula is satisfied. If the robustness
        is negative, then the formula is not satisfied.

        An untimed until operator has robustness value at any time t equal to

            sup_{tau in [t, infty)} min(release(tau), inf_{s in [t, tau] invariant(s)})

        args:
            s: the signal upon which this formula is evaluated.
            smoothing: the parameter determining how much to smooth non-continuous
                elements of this formula (e.g. mins and maxes). Uses the log-sum-exp
                smoothing for max and min.
        returns:
            the robustness trace for this formula and the given signal.
        """
        # Get the robustness of the child formulae
        invariant_robustness = self.invariant(env, smoothing)
        release_robustness = self.release(env, smoothing)

        # Do a forward pass to get the robustness of the invariant holding up to each
        # timestep (minimize by taking negative of max of negative)
        f = lambda carry, x: accumulate_max(carry, x, smoothing)
        _, r_invariant_always = jax.lax.scan(
            jax.jit(f), -jnp.inf, -invariant_robustness, reverse=False
        )
        r_invariant_always = -r_invariant_always

        # Get the elementwise min with the release robustness
        # first, stack the two signals
        release_stacked_with_invariant_always = jnp.vstack((release_robustness, r_invariant_always))

        # compute the element wise max of the negated stacked signals
        smoothed_max_of_stacked_signals = 1 / smoothing * \
            logsumexp(smoothing * -release_stacked_with_invariant_always, axis=0)

        # negate the max
        r_min = -smoothed_max_of_stacked_signals

        # Get the robustness trace as the accumulated maximum (backwards from the
        # end of the array) to compute the supremum
        _, until_robustness = jax.lax.scan(f, -jnp.inf, r_min, reverse=True)

        return until_robustness


class STLTimedUntil(STLFormula):
    """Represents an STL temporal operator for until with a time bound"""

    def __init__(
        self,
        invariant: STLFormula,
        release: STLFormula,
        t_start: float,
        t_end: float,
        interpolate: bool = True,
    ):
        """Initialize an STL formula for a timed until. This formula is satisfied
        for a given signal if invariant is satisfied at all times until release is
        satisfied, which must happen during the given interval.

        args:
            invariant: an STL formula that should hold until release holds
            release: an STL formula that should hold eventually
            t_start: start point (inclusive) of interval
            t_end: end point (inclusive) of interval
            interpolate: if True, will look for intersections between child traces
        """
        super(STLTimedUntil, self).__init__()

        self.invariant = invariant
        self.release = release

        # Make use of the fact that x U_[a, b] y is equivalent to
        # (eventually_[a, b] y) and (always_[0, a] x U y)
        self.child = STLAnd(
            STLTimedEventually(release, t_start, t_end),
            STLTimedAlways(
                STLUntimedUntil(invariant, release, interpolate), 0, t_start
            ),
            interpolate,
        )

    @property
    def children(self):
        return [self.invariant, self.release]

    @partial(jax.jit, static_argnames=["self"])
    def __call__(self, env: dict[str: jnp.ndarray], smoothing: float = 100.0) -> jnp.ndarray:
        """Evaluates this formula on the given signal, returning its robustness trace.

        The robustness trace is an array of the same length as s where each
        sample represents the robustness of the subsignal of s starting at that sample
        and continuing to the end of s. I.e. if the robustness trace has sample
        (t_i, r_i), then the subsignal s[i:] has robustness r_i with respect to this
        formula.

        If robustness is positive, then the formula is satisfied. If the robustness
        is negative, then the formula is not satisfied.

        An timed until operator has robustness value at any time t equal to that of
        its equivalent child. Recall that we re-write the timed until as an equivalent
        child formula

        x U_[a, b] y <-> (eventually_[a, b] y) and (always_[0, a] x U y)

        args:
            s: the signal upon which this formula is evaluated.
            smoothing: the parameter determining how much to smooth non-continuous
                elements of this formula (e.g. mins and maxes). Uses the log-sum-exp
                smoothing for max and min.
        returns:
            the robustness trace for this formula and the given signal.
        """
        # Just evaluate the child, which is a rewritten form of the timed until
        return self.child(s, smoothing)


class STLUntimedAlways(STLFormula):
    """Represents an STL temporal operator for always without a time bound"""

    def __init__(self, child: STLFormula):
        """Initialize an STL formula for an untimed always. This formula is satisfied
        for a given signal if its child is satisfied at all points in the future.

        Implemented as not eventually not child.

        args:
            child: an STL formula that should always hold
        """
        super(STLUntimedAlways, self).__init__()

        self._children = [child]
        self.child = STLNegation(STLUntimedEventually(STLNegation(child)))

    @property
    def children(self):
        return self._children

    @partial(jax.jit, static_argnames=["self"])
    def __call__(self, env: dict[str: jnp.ndarray], smoothing: float = 100.0, end_time: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Evaluates this formula on the given signal, returning its robustness trace.

        The robustness trace is an array of the same length as s where each
        sample represents the robustness of the subsignal of s starting at that sample
        and continuing to the end of s. I.e. if the robustness trace has sample
        (t_i, r_i), then the subsignal s[i:] has robustness r_i with respect to this
        formula.

        If robustness is positive, then the formula is satisfied. If the robustness
        is negative, then the formula is not satisfied.

        An untimed always operator is equivalent to not (eventually not child)

        args:
            s: the signal upon which this formula is evaluated.
            smoothing: the parameter determining how much to smooth non-continuous
                elements of this formula (e.g. mins and maxes). Uses the log-sum-exp
                smoothing for max and min.
        returns:
            the robustness trace for this formula and the given signal.
        """
        return self.child(env, smoothing, end_time)


class STLTimedAlways(STLFormula):
    """Represents an STL temporal operator for always with a time bound"""

    def __init__(self, child: STLFormula, t_start: float, t_end: float):
        """Initialize an STL formula for an untimed always. This formula is satisfied
        for a given signal if its child is satisfied at all points in the future.

        Implemented as not eventually not child.

        args:
            child: an STL formula that should always hold
            t_start: start point (inclusive) of interval
            t_end: end point (inclusive) of interval
        """
        super(STLTimedAlways, self).__init__()

        self._children = [child]
        self.child = STLNegation(STLTimedEventually(STLNegation(child), t_start, t_end))

    @property
    def children(self):
        return self._children

    @partial(jax.jit, static_argnames=["self"])
    def __call__(self, env: dict[str: jnp.ndarray], smoothing: float = 100.0) -> jnp.ndarray:
        """Evaluates this formula on the given signal, returning its robustness trace.

        The robustness trace is an array of the same length as s where each
        sample represents the robustness of the subsignal of s starting at that sample
        and continuing to the end of s. I.e. if the robustness trace has sample
        (t_i, r_i), then the subsignal s[i:] has robustness r_i with respect to this
        formula.

        If robustness is positive, then the formula is satisfied. If the robustness
        is negative, then the formula is not satisfied.

        An untimed always operator is equivalent to not (eventually not child)

        args:
            s: the signal upon which this formula is evaluated.
            smoothing: the parameter determining how much to smooth non-continuous
                elements of this formula (e.g. mins and maxes). Uses the log-sum-exp
                smoothing for max and min.
        returns:
            the robustness trace for this formula and the given signal.
        """
        return self.child(s, smoothing)
