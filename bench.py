import jax
jax.distributed.initialize()
rank = jax.process_index()
size = jax.process_count()
import os
# os.environ['ENABLE_PERFO_STEP'] = 'CUDA'
# os.environ['PERFO_STEP_OUT_FILE'] = 'perfo_step_out.md'
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh , PartitionSpec as P , NamedSharding
from jax.experimental.shard_map import shard_map
import nccl_mpi_benchmarks as nmb
from functools import partial
import numpy as np


print(f"rank {rank} size {size} device count {jax.device_count()}")

devices = mesh_utils.create_device_mesh((jax.device_count(),))
mesh = Mesh(devices, ('gpus',))
sharding = NamedSharding(mesh , P('gpus',))
def create_global_array(global_shape=(32,)):

  local_shape = (global_shape[0] // jax.process_count(),)
  gspmd_array = jax.make_array_from_callback(global_shape , sharding, lambda _: jnp.ones(local_shape, dtype=jnp.float32) * jax.process_index())

  return gspmd_array 



@partial(shard_map , mesh=mesh, in_specs=P('gpus') , out_specs=P('gpus'),check_rep=False)
def all_reduce_nccl(operand):
      return nmb.ops.collective_call(operand , nmb.Backend.NCCL, nmb.Collective.AllReduce, nmb.Mode.OutOfPlace)
  
@partial(shard_map , mesh=mesh, in_specs=P('gpus') , out_specs=P('gpus'),check_rep=False)
def all_reduce_mpi(operand):
        return nmb.ops.collective_call(operand , nmb.Backend.MPI, nmb.Collective.AllReduce, nmb.Mode.OutOfPlace)

@partial(shard_map , mesh=mesh, in_specs=P('gpus') , out_specs=P('gpus'),check_rep=False)
def p2p_nccl(operand):
        return nmb.ops.collective_call(operand , nmb.Backend.NCCL, nmb.Collective.Peer2Peer, nmb.Mode.OutOfPlace)

@partial(shard_map , mesh=mesh, in_specs=P('gpus') , out_specs=P('gpus'),check_rep=False)
def p2p_mpi(operand):
        return nmb.ops.collective_call(operand , nmb.Backend.MPI, nmb.Collective.Peer2Peer, nmb.Mode.OutOfPlace)

@partial(shard_map , mesh=mesh, in_specs=P('gpus') , out_specs=P('gpus'),check_rep=False)
def all2all_nccl(operand):
        return nmb.ops.collective_call(operand , nmb.Backend.NCCL, nmb.Collective.AllToAll, nmb.Mode.OutOfPlace)

def all2all_mpi(operand):
        return nmb.ops.collective_call(operand , nmb.Backend.MPI, nmb.Collective.AllToAll, nmb.Mode.OutOfPlace)

sizes_in_gb = np.arange(1, 10, 1)

for size_in_gb in sizes_in_gb:
  element_count = size_in_gb * 1024 * 1024 * 1024 // 4
  element_count *= size
  global_shape = (element_count,)
  gspmd_array = create_global_array(global_shape)

  result_nccl = all_reduce_nccl(gspmd_array)
  del result_nccl
  result_mpi = all_reduce_mpi(gspmd_array)
  del result_mpi
  result_p2p_nccl = p2p_nccl(gspmd_array)
  del result_p2p_nccl
  result_p2p_mpi = p2p_mpi(gspmd_array)
  del result_p2p_mpi
  result_all2all_nccl = all2all_nccl(gspmd_array)
  del result_all2all_nccl
  result_all2all_mpi = all2all_mpi(gspmd_array)
  del result_all2all_mpi
