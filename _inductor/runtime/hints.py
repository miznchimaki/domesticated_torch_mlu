import typing
from typing import Optional

import torch

TRITON_MAX_BLOCK = {
    "X": 1024 * 64,
    "Y": 1024 * 64,
    "Z": 1024 * 64,
    "R0_": 4096 * 16,  # * 16 is multi-kernel only
    "R1_": 2048 * 16,  # * 16 is multi-kernel only
}

torch._inductor.runtime.hints.TRITON_MAX_BLOCK.update(TRITON_MAX_BLOCK)


class DeviceProperties(typing.NamedTuple):
    """Copy device properties into a data structure not requiring torch to be imported"""

    type: str  # type: ignore[assignment]
    index: int  # type: ignore[assignment]
    cc: int
    supports_linear_memory: Optional[bool] = False
    major: Optional[int] = None
    regs_per_multiprocessor: Optional[int] = None
    max_threads_per_multi_processor: Optional[int] = None
    max_threads_per_block: int | None = None
    multi_processor_count: Optional[int] = None
    warp_size: Optional[int] = None
    onchip_mem_size: Optional[int] = None

    @classmethod
    def create(cls, device):
        import torch
        from torch._dynamo.device_interface import get_interface_for_device

        device_type = device.type
        device_interface = get_interface_for_device(device)
        if device_type == "mlu":
            props = device_interface.get_device_properties(device)
            cc = device_interface.get_compute_capability(device)
            return cls(
                type=device_type,
                index=device.index,
                cc=cc,
                supports_linear_memory=props.supports_linear_memory,
                major=props.major,
                regs_per_multiprocessor=props.regs_per_multiprocessor
                if hasattr(props, "regs_per_multiprocessor")
                else None,
                max_threads_per_multi_processor=props.max_threads_per_multi_processor
                if hasattr(props, "max_threads_per_multi_processor")
                else None,
                multi_processor_count=props.multi_processor_count
                if hasattr(props, "multi_processor_count")
                else None,
                max_threads_per_block=getattr(props, "max_threads_per_block", 1024),
                warp_size=props.warp_size if hasattr(props, "warp_size") else 32,
                onchip_mem_size=props.nram_size + props.wram_size
                if cc >= 600
                else props.nram_size,
            )
        return cls(
            type=device_type,
            index=device.index,
            cc=device_interface.get_compute_capability(device),
        )


torch._inductor.runtime.hints.DeviceProperties = DeviceProperties
