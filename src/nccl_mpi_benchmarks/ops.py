import jax
import jax.numpy as jnp
import numpy as np
import jax.extend as jex


def add_element(a, scaler=1):

    if a.dtype != jnp.float32:
        raise ValueError("Only float32 is supported")

    out_type = jax.ShapeDtypeStruct(a.shape, a.dtype)

    return jex.ffi.ffi_call("add_element",
                            out_type,
                            a,
                            scaler=scaler,
                            vectorized=False)
