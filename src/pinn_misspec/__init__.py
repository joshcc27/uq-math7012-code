"""Parameter compensation in inverse PINNs under model misspecification.

JAX defaults to single precision and downcasts silently, so double precision is
switched on here, before any submodule creates an array.
"""

import jax

jax.config.update("jax_enable_x64", True)
