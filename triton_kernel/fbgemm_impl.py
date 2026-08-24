from typing import List, Optional, Tuple
import torch


fbgemm_lib = torch.library.Library("fbgemm", "FRAGMENT")


@torch.library.impl(fbgemm_lib, "dense_to_jagged_forward", "PrivateUse1")
def torch_dense_to_jagged_forward(
    dense: torch.Tensor,
    offsets: List[torch.Tensor],
    total_L: Optional[int] = None,
) -> torch.Tensor:
    from .fbgemm_kernels import dense_to_jagged

    with torch.mlu.device(dense.device.index):
        return dense_to_jagged(dense, offsets, total_L)[0]


@torch.library.impl(fbgemm_lib, "jagged_to_padded_dense_forward", "PrivateUse1")
def torch_jagged_to_padded_dense_forward(
    values: torch.Tensor,
    offsets: List[torch.Tensor],
    max_lengths: List[int],
    padding_value: float = 0,
) -> torch.Tensor:
    from .fbgemm_kernels import jagged_to_dense

    with torch.mlu.device(values.device.index):
        return jagged_to_dense(values, offsets, max_lengths, padding_value)


@torch.library.impl(fbgemm_lib, "jagged_to_padded_dense_backward", "PrivateUse1")
def torch_jagged_to_padded_dense_backward(
    grad_output: torch.Tensor,
    offsets: List[torch.Tensor],
    total_L: Optional[int] = None,
) -> torch.Tensor:
    from .fbgemm_kernels import dense_to_jagged

    with torch.mlu.device(grad_output.device.index):
        return dense_to_jagged(grad_output, offsets, total_L)[0]
