import contextlib
import torch
from torch.distributed._local_tensor import (
    LocalTensor,
)

from ...utils import gorilla


@gorilla.patch(torch.distributed._local_tensor._LocalOffsetBasedRNGTracker)
@contextlib.contextmanager
def _distribute_region(self, spec, generator=None):
    """Context manager for LocalTensor mode distribute region."""
    lm = enabled_local_tensor_mode()
    assert lm is not None

    # get base state
    if generator is not None:
        base_state_tensor = generator.get_state()
        per_rank_states = {rank: base_state_tensor.clone() for rank in lm.ranks}
        # pyrefly: ignore [bad-argument-type, bad-argument-count]
        base_state_tensor = LocalTensor(per_rank_states)
    else:
        base_state_tensor = self._device_handle.get_rng_state()

    state = _LocalPhiloxState(base_state_tensor)

    if self.distribute_region_enabled:
        # sync to rank 0's state if no explicit generator
        if generator is None:
            any_rank_state = lm._any_local_rng_state()
            any_rank_cpu, any_rank_cuda = any_rank_state

            # Modify by CAMBRICON
            # if self._device.type == "cuda":
            if self._device.type == "cuda" or self._device.type == "mlu":
                # end Modify by CAMBRICON
                assert self._device.index in any_rank_cuda
                any_rank_device_state = any_rank_cuda[self._device.index]
            else:
                any_rank_device_state = any_rank_cpu

            from torch.distributed.tensor._random import _PhiloxState

            any_rank_philox = _PhiloxState(any_rank_device_state)
            state.seed = int(any_rank_philox.seed.item())
            state.offset = int(any_rank_philox.offset.item())


        old_offset = state.offset
        self._set_pre_op_offset(state, spec)
        state.apply_to_local_tensor_mode(self._device_handle)

        try:
            yield
        finally:
            self._set_post_op_offset(state, spec, old_offset)
            state.apply_to_local_tensor_mode(self._device_handle)
    else:
        yield

    # maybe reset generator to rank 0's state
    if generator is not None:
        rank_0_state = state._per_rank_states[0]
        generator.set_state(rank_0_state)
