from ._shard.tensor_ops import apply_tensor_ops_patch
from .nn.functional import apply_functional_patch
from .tensor._attention import *  
from .tensor._matrix_ops import *  

def apply_distributed_patch():
    apply_tensor_ops_patch()
    apply_functional_patch()
    apply_context_parallel_patch()
