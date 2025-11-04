import jax
import jax.numpy as jnp


def argmax_with_random_tiebreak(array, key, axis=-1):
    max_values = jnp.max(array, axis=axis, keepdims=True)
    is_max = jnp.where(array == max_values, 1, 0)

    # Generate random noise for tie-breaking
    noise = jax.random.uniform(key, shape=array.shape)
    perturbed_array = array + is_max * noise

    return jnp.argmax(perturbed_array, axis=axis)

def argmax_with_safest_tiebreak(array, safety_or_cost, axis=-1, use_sum_cost_critic=False):
    """Note this is typically given an array where all unsafe actions are set to
    have reward = -999"""
    max_value = jnp.max(array, axis=axis, keepdims=True)
    max_mask = (array == max_value)
    max_masked_array = jnp.where(max_mask, array, -jnp.inf)

    if use_sum_cost_critic:
        # TODO: Use axis kwarg?
        cost = safety_or_cost
        cost_ranks = jnp.argsort(jnp.argsort(cost))
        cost_ordered_max_array = max_masked_array - cost_ranks
        return jnp.argmax(cost_ordered_max_array, axis=axis)
    else:
        # TODO: Use axis kwarg?
        safety = safety_or_cost
        safety_ranks = jnp.argsort(jnp.argsort(safety))
        safety_ordered_max_array = max_masked_array + safety_ranks
        return jnp.argmax(safety_ordered_max_array, axis=axis)
