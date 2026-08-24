# mypy: allow-untyped-defs
# pylint: disable=useless-parent-delegation
from __future__ import annotations

import ctypes
import torch
import torch_mlu
from torch._utils import _dummy_type

if not hasattr(torch_mlu._MLUC, "_MLUStreamBase"):
    # Define dummy base classes
    torch_mlu._MLUC.__dict__["_MLUStreamBase"] = _dummy_type("_MLUStreamBase")
    torch_mlu._MLUC.__dict__["_MLUEventBase"] = _dummy_type("_MLUEventBase")


# define Stream
class Stream(torch_mlu._MLUC._MLUStreamBase):
    r"""Wrapper around a MLU stream.

    A MLU stream is a linear sequence of execution that belongs to a specific
    device, independent from other streams. It supports with statement as a
    context manager to ensure the operators within the with block are running
    on the corresponding stream.

    Args:
        device(torch.device or int, optional): a device on which to allocate
            the stream. If :attr:`device` is ``None`` (default) or a negative
            integer, this will use the current device.
        priority(int, optional): priority of the stream, which can be positive, 0, or negative.
            A lower number indicates a higher priority. Values outside the allowed range will
            be automatically mapped to the nearest valid priority. By default, the priority is set to 0.

    """

    def __new__(cls, device=None, priority=0, **kwargs):
        # setting device manager is expensive, so we avoid it unless necessary
        if device is None or ("stream_id" in kwargs and "device_index" in kwargs):
            return super().__new__(cls, priority=priority, **kwargs)
        else:
            with torch.mlu.device(device):
                return super().__new__(cls, priority=priority, **kwargs)

    def wait_event(self, event: Event | torch.Event) -> None:
        r"""Make all future work submitted to the stream wait for an event.

        Args:
            event (Event, torch.Event): an event to wait for.

        .. note:: This is a wrapper around ``cnrtQueueWaitNotifier()``: see
           `CNRT documentation`_ for more info.

           This function returns without waiting for :attr:`event`: only future
           operations are affected.
        """
        event.wait(self)

    def wait_stream(self, stream: Stream | torch.Stream) -> None:
        r"""Synchronize with another stream.

        All future work submitted to this stream will wait until all kernels
        submitted to a given stream at the time of call complete.

        Args:
            stream (Stream, torch.Stream): a stream to synchronize.

        .. note:: This function returns without waiting for currently enqueued
           kernels in :attr:`stream`: only future operations are affected.
        """
        self.wait_event(stream.record_event())

    def record_event(self, event: Event | torch.Event | None = None):
        r"""Record an event.

        Args:
            event (Event, torch.Event, optional): event to record. If not given, a new one
                will be allocated.

        Returns:
            Recorded event.
        """
        if event is None:
            event = Event()
        event.record(self)
        return event

    def query(self) -> bool:
        r"""Check if all the work submitted has been completed.

        Returns:
            A boolean indicating if all kernels in this stream are completed.
        """
        return super().query()

    def synchronize(self) -> None:
        r"""Wait for all the kernels in this stream to complete.

        .. note:: This is a wrapper around ``cnrtQueueSync()``: see
           `CNRT documentation`_ for more info.
        """
        super().synchronize()

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.mlu_stream)

    def __eq__(self, o) -> bool:
        if isinstance(o, Stream):
            return super().__eq__(o)
        return False

    def __hash__(self):
        return hash((self.mlu_stream, self.device))

    def __repr__(self):
        return (
            f"<torch.mlu.Stream device={self.device} mlu_stream={self.mlu_stream:#x}>"
        )

    def __mlu_stream__(self):
        """Implements the MLU Stream Protocol.

        Returns:
            tuple: A 2-tuple of (version, handle) where version is the protocol version
                   and handle is the address of cnrtQueue_t as a Python int.
        """
        return (0, self.mlu_stream)


class ExternalStream(Stream):
    r"""Wrapper around an externally allocated MLU stream.

    This class is used to wrap streams allocated in other libraries in order
    to facilitate data exchange and multi-library interactions.

    .. note:: This class doesn't manage the stream life-cycle, it is the user
       responsibility to keep the referenced stream alive while this class is
       being used.

    Args:
        stream_ptr(int): Integer representation of the `cnrtQueue_t` value.
            allocated externally.
        device(torch.device or int, optional): the device where the stream
            was originally allocated. If device is specified incorrectly,
            subsequent launches using this stream may fail.
    """

    def __new__(cls, stream_ptr, device=None, **kwargs):
        with torch.mlu.device(device):
            return super().__new__(cls, stream_ptr=stream_ptr, **kwargs)


### Event
class Event(torch_mlu._MLUC._MLUEventBase):
    r"""Wrapper around a MLU event.

    MLU events are synchronization markers that can be used to monitor the
    device's progress, to accurately measure timing, and to synchronize MLU
    streams.

    The underlying MLU events are lazily initialized when the event is first
    recorded or exported to another process. After creation, only streams on the
    same device may record the event. However, streams on any device can wait on
    the event.

    Args:
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
        blocking (bool, optional): if ``True``, :meth:`wait` will be blocking (default: ``False``)
        interprocess (bool): if ``True``, the event can be shared between processes
            (default: ``False``)
        external (bool, optional): indicates whether this event should create event record and event wait nodes, or create an internal cross-stream dependency, when captured in a mlu graph.

    """

    def __new__(
        cls, enable_timing=False, blocking=False, interprocess=False, external=False
    ):
        return super().__new__(
            cls,
            enable_timing=enable_timing,
            blocking=blocking,
            interprocess=interprocess,
            external=external,
        )

    @classmethod
    def from_ipc_handle(cls, device, handle):
        r"""Reconstruct an event from an IPC handle on the given device."""
        return super().from_ipc_handle(device, handle)

    def record(self, stream: Stream | torch.Stream | None = None):
        r"""Record the event in a given stream.

        Args:
            stream (Stream, torch.Stream, optional): Uses ``torch.mlu.current_stream()`` if no stream is specified.
                The stream's device must match the event's device.
        """
        if stream is None:
            stream = torch.mlu.current_stream()
        super().record(stream)

    def wait(self, stream: Stream | torch.Stream | None = None) -> None:
        r"""Make all future work submitted to the given stream wait for this event.

        Args:
            stream (Stream, torch.Stream, optional): Uses ``torch.mlu.current_stream()`` if no stream is specified.

        .. note:: This is a wrapper around ``cnrtQueueWaitNotifier()``: see
            `CNRT documentation`_ for more info.
        """
        if stream is None:
            stream = torch.mlu.current_stream()
        super().wait(stream)

    def query(self):
        r"""Check if all work currently captured by event has completed.

        Returns:
            A boolean indicating if all work currently captured by event has
            completed.
        """
        return super().query()

    def elapsed_time(self, end_event: Event):
        r"""Return the time elapsed.

        Time reported in milliseconds after the event was recorded and
        before the end_event was recorded.

        Args:
            end_event (Event): the end event.
        """
        return super().elapsed_time(end_event)

    def hardware_time(self, end_event: Event):
        r"""Return the hardware time elapsed.

        Time reported in microseconds.

        Args:
            end_event (Event): the end event.
        """
        time = super().hardware_time(end_event)
        return time

    def synchronize(self) -> None:
        r"""Wait for the event to complete.

        Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.

         .. note:: This is a wrapper around ``cnrtWaitNotifier``: see
            `CNRT documentation`_ for more info.
        """
        super().synchronize()

    def ipc_handle(self):
        r"""Return an IPC handle of this event.

        If not recorded yet, the event will use the current device.
        """
        return super().ipc_handle()

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.mlu_event)

    def __repr__(self) -> str:
        if self.mlu_event:
            return f"<torch.mlu.Event {self._as_parameter_.value:#x}>"
        else:
            return "<torch.mlu.Event uninitialized>"
