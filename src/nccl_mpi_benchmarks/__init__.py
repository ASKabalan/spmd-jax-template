from nccl_mpi_lib import gpu_ops
import jax.extend as jex

jex.ffi.register_ffi_target("collective_call",
                            jex.ffi.pycapsule(gpu_ops.collective_call),
                            platform="cuda")
jex.ffi.register_ffi_target("add_element",
                            jex.ffi.pycapsule(gpu_ops.add_element),
                            platform="cuda")
