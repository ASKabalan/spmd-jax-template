#include "gpu_ops.cuh"
#include "mpi_ops.hpp"
#include "nanobind/nanobind.h"
#include "nccl_ops.hpp"
#include "xla/ffi/api/api.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <type_traits>

namespace ffi = xla::ffi;
namespace nb = nanobind;

enum class Backend { NCCL, MPI };

enum class Mode { InPlace, OutOfPlace };

enum class Collective { AllReduce, AllGather, AllToAll, Peer2Peer };

ffi::Error CollectiveImpl(cudaStream_t stream, int iBackend, int iCollective,
                          int iMode, ffi::Buffer<ffi::DataType::F32> x,
                          ffi::Result<ffi::Buffer<ffi::DataType::F32>> y) {

  MPIOps mpi_ops;
  // int iCollective = 0, iMode = 0, iBackend = 0;
  Backend backend = static_cast<Backend>(iBackend);
  Collective collective = static_cast<Collective>(iCollective);
  Mode mode = static_cast<Mode>(iMode);
  const int &rank = mpi_ops.get_rank();
  const int &size = mpi_ops.get_size();
  // Pair ranks permute with Odd ranks
  // So if rank is even, next_rank is rank + 1
  // If rank is odd, next_rank is rank - 1
  int next_rank = rank % 2 == 0 ? (rank + 1) % size : (rank - 1 + size) % size;
  int prev_rank = rank % 2 == 0 ? (rank - 1 + size) % size : (rank + 1) % size;
  if (backend == Backend::MPI) {
    switch (collective) {
    case Collective::AllReduce:
      if (mode == Mode::InPlace) {
        MPI_Allreduce(MPI_IN_PLACE, x.untyped_data(), y->element_count(),
                      MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      } else {
        MPI_Allreduce(x.untyped_data(), x.untyped_data(), y->element_count(),
                      MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      }
      break;
    case Collective::AllGather:
      MPI_Allgather(x.untyped_data(), x.element_count(), MPI_FLOAT,
                    y->untyped_data(), x.element_count(), MPI_FLOAT,
                    MPI_COMM_WORLD);
      break;
    case Collective::AllToAll:
      if (mode == Mode::InPlace) {
        MPI_Alltoall(x.untyped_data(), x.element_count(), MPI_FLOAT,
                     y->untyped_data(), x.element_count(), MPI_FLOAT,
                     MPI_COMM_WORLD);
        break;
      } else {
        MPI_Alltoall(MPI_IN_PLACE, 0, MPI_FLOAT, x.untyped_data(),
                     x.element_count(), MPI_FLOAT, MPI_COMM_WORLD);
      }
      break;
    case Collective::Peer2Peer:
      MPI_Sendrecv(x.untyped_data(), x.element_count(), MPI_FLOAT, next_rank, 0,
                   y->untyped_data(), y->element_count(), MPI_FLOAT, rank, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      break;
    }
  } else {
    // NCCL
    NCCLOps nccl_ops;
    const int &rank = nccl_ops.get_rank();
    const int &size = nccl_ops.get_size();
    ncclComm_t comm = nccl_ops.get_comm();
    size_t chunk_size = x.element_count() / size;
    switch (collective) {
    case Collective::AllReduce:
      ncclAllReduce(x.untyped_data(), y->untyped_data(), x.element_count(),
                    ncclFloat32, ncclSum, comm, stream);
      break;
    case Collective::AllGather:
      ncclAllGather(x.untyped_data(), y->untyped_data(), x.element_count(),
                    ncclFloat32, comm, stream);
      break;
    case Collective::AllToAll:
      ncclGroupStart();
      for (int r = 0; r < size; r++) {
        ncclSend((void *)(x.typed_data() + r * chunk_size), chunk_size,
                 ncclFloat32, r, comm, stream);
        ncclRecv((void *)(y->typed_data() + r * chunk_size), chunk_size,
                 ncclFloat32, r, comm, stream);
      }
      ncclGroupEnd();
      break;
    case Collective::Peer2Peer:
      ncclGroupStart();
      ncclSend(x.untyped_data(), x.element_count(), ncclFloat32, next_rank,
               comm, stream);
      ncclRecv(y->untyped_data(), y->element_count(), ncclFloat32, prev_rank,
               comm, stream);
      break;
    }
  }
  return ffi::Error::Success();
}

ffi::Error AddElementImpl(cudaStream_t stream, float scaler,
                          ffi::Buffer<ffi::DataType::F32> x,
                          ffi::Result<ffi::Buffer<ffi::DataType::F32>> y) {

  add_element(scaler, x.typed_data(), y->typed_data(), x.element_count(),
              stream);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CollectiveCall, CollectiveImpl,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::PlatformStream<cudaStream_t>>()
                           .Attr<int>("iBackend")
                           .Attr<int>("iCollective")
                           .Attr<int>("iMode")
                           .Arg<ffi::Buffer<ffi::DataType::F32>>() // x
                           .Ret<ffi::Buffer<ffi::DataType::F32>>() // y
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(AddElement, AddElementImpl,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::PlatformStream<cudaStream_t>>()
                           .Attr<float>("scaler") // scaler
                           .Arg<ffi::Buffer<ffi::DataType::F32>>()
                           .Ret<ffi::Buffer<ffi::DataType::F32>>() // y
);

template <typename T> nb::capsule EncapsulateFfiCall(T *fn) {
  // This check is optional, but it can be helpful for avoiding invalid
  // handlers.
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be and XLA FFI handler");
  return nb::capsule(reinterpret_cast<void *>(fn));
}

nb::dict Registrations() {
  nb::dict d;
  d["collective_call"] = EncapsulateFfiCall(CollectiveCall);
  d["add_element"] = EncapsulateFfiCall(AddElement);
  return d;
}

NB_MODULE(nccl_mpi_lib, m) {
  m.def("registrations", &Registrations);
  nb::enum_<Backend>(m, "Backend")
      .value("NCCL", Backend::NCCL)
      .value("MPI", Backend::MPI)
      .export_values();

  nb::enum_<Mode>(m, "Mode")
      .value("InPlace", Mode::InPlace)
      .value("OutOfPlace", Mode::OutOfPlace)
      .export_values();

  nb::enum_<Collective>(m, "Collective")
      .value("AllReduce", Collective::AllReduce)
      .value("AllGather", Collective::AllGather)
      .value("AllToAll", Collective::AllToAll)
      .value("Peer2Peer", Collective::Peer2Peer)
      .export_values();
}
