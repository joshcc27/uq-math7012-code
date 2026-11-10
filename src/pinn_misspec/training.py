"""Configuration checks and training helpers shared by the two models."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax


def finite(name, value):
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value}")
    return number


def integral(name, value):
    number = finite(name, value)
    if number != int(number):
        raise ValueError(f"{name} must be a whole number, got {value}")
    return int(number)


def apply_overrides(config, overrides, optional=None):
    """Apply ``overrides`` to a default configuration.

    Args:
        config: The default configuration, updated in place.
        overrides: Entries to set. An unknown key raises ``KeyError``, so a
            mistyped setting fails instead of being ignored.
        optional: Maps each optional key to its default. An optional key is
            kept only when it differs from its default, so leaving it unset
            gives the same configuration, and the same run id, as before the
            key existed.

    Returns:
        The updated configuration.
    """
    optional = optional or {}
    unknown = set(overrides) - set(config) - set(optional)
    if unknown:
        raise KeyError(f"Unknown configuration entries: {sorted(unknown)}")
    config.update(overrides)
    for key, default in optional.items():
        if key in config and config[key] == default:
            del config[key]
    return config


def cosine_adam(config, steps):
    """Adam for the network and the coefficients, each rate decaying to zero over ``steps`` updates."""
    return {
        "net": optax.adam(optax.cosine_decay_schedule(config["network_learning_rate"], steps)),
        "theta": optax.adam(optax.cosine_decay_schedule(config["theta_learning_rate"], steps)),
    }


def resume(checkpoint, steps, initial_params):
    """Return ``(params, optimizer_state, steps_done)`` to start or continue a run.

    Args:
        checkpoint: ``None``, or ``(params, optimizer_state, steps_done)`` from
            an earlier part of the same run.
        steps: The full budget of the run.
        initial_params: The starting parameters, to check the checkpoint against.
    """
    if checkpoint is None:
        return initial_params, None, 0
    params, optimizer_state, steps_done = checkpoint
    steps_done = int(steps_done)
    if not 0 <= steps_done <= steps:
        raise ValueError(
            f"checkpoint reports {steps_done} completed updates, outside the budget of {steps}"
        )
    if jax.tree.structure(params) != jax.tree.structure(initial_params):
        raise ValueError("checkpoint parameters do not match this configuration")
    return params, optimizer_state, steps_done


def relative_l2(estimate, reference):
    estimate, reference = np.asarray(estimate), np.asarray(reference)
    return float(np.linalg.norm(estimate - reference) / np.linalg.norm(reference))


def relative_error(value, reference):
    return abs(value - reference) / abs(reference)


def build_network(config, in_size):
    """Return the tanh MLP of a configuration: the residuals need second derivatives, which ReLU lacks."""
    return eqx.nn.MLP(
        in_size=in_size,
        out_size=1,
        width_size=int(config["network_width"]),
        depth=int(config["network_depth"]),
        activation=jnp.tanh,
        key=jax.random.key(int(config["network_seed"])),
    )
