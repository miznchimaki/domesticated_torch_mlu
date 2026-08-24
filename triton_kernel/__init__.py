import torch

try:
    aten_lib = torch.library.Library("aten", "IMPL")

    from ._transform_bias_rescale_qkv_impl import _transform_bias_rescale_qkv_mlu
    aten_lib.impl("_transform_bias_rescale_qkv", _transform_bias_rescale_qkv_mlu, "PrivateUse1")

    from .replication_pad3d_impl import replication_pad3d_mlu
    from .replication_pad3d_impl import replication_pad3d_out_mlu
    aten_lib.impl("replication_pad3d", replication_pad3d_mlu, "PrivateUse1")
    aten_lib.impl("replication_pad3d.out", replication_pad3d_out_mlu, "PrivateUse1")

    from .replication_pad3d_backward_impl import replication_pad3d_backward_mlu
    from .replication_pad3d_backward_impl import replication_pad3d_backward_grad_input_mlu
    aten_lib.impl("replication_pad3d_backward", replication_pad3d_backward_mlu, "PrivateUse1")
    aten_lib.impl("replication_pad3d_backward.grad_input", replication_pad3d_backward_grad_input_mlu, "PrivateUse1")
except ModuleNotFoundError:
    pass

try:
    from .fbgemm_impl import (
            torch_dense_to_jagged_forward,
            torch_jagged_to_padded_dense_forward,
            torch_jagged_to_padded_dense_backward
        )
except ModuleNotFoundError:
    pass
