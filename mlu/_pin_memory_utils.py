import torch
from torch_mlu.utils import gorilla
from torch.cuda import _pin_memory_utils as cuda_pin_memory_utils


@gorilla.patch(cuda_pin_memory_utils)
def pin_memory(data_ptr: int, size: int) -> None:
    # Modify by CAMBRICON
    # cudart = torch.cuda.cudart()
    # succ = int(
    #     cudart.cudaHostRegister(
    #         data_ptr,
    #         size,
    #         1,  # lines up with 'cudaHostRegisterPortable'
    #     )
    # )
    cnrt = torch.mlu.cnrt()
    succ = int(
        cnrt.mluHostRegister(
            data_ptr,
            size,
            1,  # lines up with 'cudaHostRegisterPortable'
        )
    )
    # end Modify by CAMBRICON
    # Add by CAMBRICON
    # cnrtErrorNotSupport
    cnrtErrorNotSupport = 100050
    if succ == cnrtErrorNotSupport:
        raise NotImplementedError("Registering memory failed with cnrtErrorNotSupport.")
    # end Add by CAMBRICON

    # Modify by CAMBRICON
    # if succ != 0:
    #     raise RuntimeError(
    #         f"Registering memory failed with cudaError: {succ}."
    #         " It's possible that this is an asynchronous error raised from a previous cuda operation."
    #         " Consider launching with CUDA_LAUNCH_BLOCKING=1 to debug."
    #     )
    if succ != 0:
        raise RuntimeError(
            f"Registering memory failed with cnrtError: {succ}."
            " It's possible that this is an asynchronous error raised from a previous cnrt operation."
            " Consider launching with CN_INVOKE_BLOCKING=1 to debug."
        )
    # end Modify by CAMBRICON


@gorilla.patch(cuda_pin_memory_utils)
def unpin_memory(data_ptr: int) -> None:
    # Modify by CAMBRICON
    # succ = int(torch.cuda.cudart().cudaHostUnregister(data_ptr))
    succ = int(torch.mlu.cnrt().mluHostUnregister(data_ptr))
    # end Modify by CAMBRICON
    if succ != 0:
        raise AssertionError(f"Unpinning shared memory failed with error-code: {succ}")
