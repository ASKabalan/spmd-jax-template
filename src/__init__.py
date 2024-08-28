from nccl_mpi_lib import gpu_ops
import jax.extend as jex

jex.ffi.register_ffi_target("CollectiveCall",
                            jex.ffi.pycapsule(gpu_ops.collective_call),
                            platform="cuda")
