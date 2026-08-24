import sys
from typing import Union
import torch
import torch_mlu
from torch.nn.parallel.scatter_gather import (  # type: ignore[attr-defined]
    _is_namedtuple,
)
from torch_mlu._MLUC import ProcessGroupGlooMLU

ProcessGroupGlooMLU.Device = torch.distributed.distributed_c10d.ProcessGroupGloo.Device
ProcessGroupGlooMLU._Options = (
    torch.distributed.distributed_c10d.ProcessGroupGloo._Options
)
torch._C._distributed_c10d.ProcessGroupGloo = ProcessGroupGlooMLU
torch.distributed.distributed_c10d.ProcessGroupGloo = ProcessGroupGlooMLU
torch.distributed.ProcessGroupGloo = ProcessGroupGlooMLU
from torch._C._distributed_c10d import ProcessGroup, ProcessGroupGloo

from torch.distributed.distributed_c10d import (
    _get_default_group,
    _rank_not_in_group,
    _GLOO_AVAILABLE,
    _check_p2p_op_list,
    _coalescing_manager,
    GroupMember,
    _update_default_pg,
    default_pg_timeout,
    Backend,
    logger,
    is_gloo_available,
    _check_valid_timeout,
    is_initialized,
    _find_pg_by_ranks_and_tag,
    _get_split_source,
    BackendConfig,
    is_mpi_available,
    is_nccl_available,
    is_ucc_available,
    is_xccl_available,
    _create_process_group_wrapper,
    _process_group_color,
)

from torch._C._distributed_c10d import (
    _ProcessGroupWrapper,
    _unregister_process_group,
    _unregister_process_group,
    _unregister_all_process_groups,
    PrefixStore,
    _DistributedBackendOptions,
    get_debug_level,
    DebugLevel,
    _register_process_group,
)

from typing import Optional
import warnings
from datetime import timedelta
import os
import warnings

from torch_mlu._MLUC import _DEFAULT_PG_CNCL_TIMEOUT

__all__ = ["default_pg_cncl_timeout"]

# Get default cncl pg timeout
default_pg_cncl_timeout: Optional[timedelta] = _DEFAULT_PG_CNCL_TIMEOUT

_CNCL_AVAILABLE = True


def version():
    major, minor, patch = torch_mlu._MLUC._cncl_version()
    return (major, minor, patch)


def is_cncl_available():
    return _CNCL_AVAILABLE


torch.distributed.__setattr__("is_cncl_available", is_cncl_available)
torch.distributed.distributed_c10d.__setattr__("is_cncl_available", is_cncl_available)


def batch_isend_irecv(p2p_op_list):
    """
    Send or Receive a batch of tensors asynchronously and return a list of requests.

    Process each of the operations in ``p2p_op_list`` and return the corresponding
    requests. CNCL, Gloo, and UCC backend are currently supported.

    Args:
        p2p_op_list: A list of point-to-point operations(type of each operator is
            ``torch.distributed.P2POp``). The order of the isend/irecv in the list
            matters and it needs to match with corresponding isend/irecv on the
            remote end.

    Returns:
        A list of distributed request objects returned by calling the corresponding
        op in the op_list.

    Examples:
        >>> # xdoctest: +SKIP("no rank")
        >>> send_tensor = torch.arange(2) + 2 * rank
        >>> recv_tensor = torch.randn(2)
        >>> send_op = dist.P2POp(dist.isend, send_tensor, (rank + 1)%world_size)
        >>> recv_op = dist.P2POp(dist.irecv, recv_tensor, (rank - 1 + world_size)%world_size)
        >>> reqs = batch_isend_irecv([send_op, recv_op])
        >>> for req in reqs:
        >>>     req.wait()
        >>> recv_tensor
        tensor([2, 3])     # Rank 0
        tensor([0, 1])     # Rank 1

    .. note:: Note that when this API is used with the CNCL PG backend, users must set
        the current GPU device with `torch.cuda.set_device`, otherwise it will
        lead to unexpected hang issues.

        In addition, if this API is the first collective call in the ``group``
        passed to ``dist.P2POp``, all ranks of the ``group`` must participate in
        this API call; otherwise, the behavior is undefined. If this API call is
        not the first collective call in the ``group``, batched P2P operations
        involving only a subset of ranks of the ``group`` are allowed.
    """
    _check_p2p_op_list(p2p_op_list)
    group = p2p_op_list[0].group
    device = p2p_op_list[0].tensor.device
    if device.type == "mlu":
        # CNCL style coalescing
        with _coalescing_manager(group, device, async_ops=True) as cm:
            for p2p_op in p2p_op_list:
                p2p_op.op(p2p_op.tensor, p2p_op.peer, p2p_op.group, p2p_op.tag)
        return cm.works
    else:
        # Backward support for Gloo
        reqs = []
        for p2p_op in p2p_op_list:
            work = p2p_op.op(p2p_op.tensor, p2p_op.peer, p2p_op.group, p2p_op.tag)
            if work:
                reqs.append(work)
        return reqs


def _abort_process_group(group: Optional[ProcessGroup] = None):
    """
    Abort a given process group. If group.WORLD (i.e. `None`) is given, all
    process groups including the default one will be aborted.

    Args:
        group (ProcessGroup, optional): The process group to be aborted.

    .. note:: this API is experimental and currently only works with the NCCL
        backend.

    .. note:: this API should be used with `TORCH_NCCL_ASYNC_ERROR_HANDLING`
        turned off (i.e. set to 0). Otherwise, ProcessGroupNCCL's watchdog may
        automatically handle errors or timeouts for you including aborting the
        ProcessGroup.
    """

    if group == GroupMember.NON_GROUP_MEMBER:
        return

    pg = group or GroupMember.WORLD

    assert pg is not None
    if torch.distributed.distributed_c10d._world.pg_map.get(pg, None) is None:
        raise ValueError("Invalid process group specified or has been destroyed.")

    try:
        backend = pg._get_backend(torch.device("mlu"))
    except RuntimeError:
        backend = None

    if not isinstance(backend, ProcessGroupCNCL):
        logger.warning(
            "`abort_process_group` currently only has implementation for ProcessGroupCNCL; "
            "however, no CNCL backend is found. This call will be a no-op."
        )
        return

    if group == GroupMember.WORLD:
        # This is copied from distributed_c10d, and CNCL may have no limitations described below(NCCL abort will incur a device-size sync):
        # Abort all backends within a ncclGroupStart|End semantic.
        # This ensures that different NCCL communicators' abort calls won't
        # deadlock each other.
        # For details, please see: https://github.com/pytorch/pytorch/issues/119797
        backend._group_start()
        for pg_to_abort in sorted(
            torch.distributed.distributed_c10d._world.pg_names,
            key=lambda x: torch.distributed.distributed_c10d._world.pg_names[x],
            reverse=True,
        ):
            _abort_backend(pg_to_abort)
        backend._group_end()

        _update_default_pg(None)
        torch.distributed.distributed_c10d._world.pg_map.clear()
        torch.distributed.distributed_c10d._world.pg_names.clear()
        torch.distributed.distributed_c10d._world.pg_group_ranks.clear()
        torch.distributed.distributed_c10d._world.pg_backend_config.clear()
        torch.distributed.distributed_c10d._world.pg_to_tag.clear()
        torch.distributed.distributed_c10d._world.tags_to_pg.clear()
        torch.distributed.distributed_c10d._world.pg_coalesce_state.clear()
        _unregister_all_process_groups()

        # when process group doesn't have an explicit name (only WORLD (default)
        # process group can have an explicit name), we use global torch.distributed.distributed_c10d._world.group_count
        # to generate the name. We need to reset the counter on destruction to
        # allow consistent value to be generated when we re-create process
        # groups after some trainers recover from failure
        #
        # We only reset this when WORLD is being destroyed because if this
        # process group is in good state, we aren't dealing with failures.
        torch.distributed.distributed_c10d._world.group_count = 0
    else:
        _abort_backend(pg)
        del torch.distributed.distributed_c10d._world.pg_map[pg]
        del torch.distributed.distributed_c10d._world.pg_names[pg]
        del torch.distributed.distributed_c10d._world.pg_group_ranks[pg]
        del torch.distributed.distributed_c10d._world.pg_backend_config[pg]
        if pg in torch.distributed.distributed_c10d._world.pg_coalesce_state.keys():
            warnings.warn(
                "Some coalesced collectives haven't been launched when "
                "ProcessGroup is aborted. They will be cleaned."
            )
            del torch.distributed.distributed_c10d._world.pg_coalesce_state[pg]

        tag = torch.distributed.distributed_c10d._world.pg_to_tag.get(pg)
        del torch.distributed.distributed_c10d._world.pg_to_tag[pg]
        if tag is not None:
            try:
                torch.distributed.distributed_c10d._world.tags_to_pg[tag].remove(pg)
                if tag.startswith("ptd:"):
                    torch.distributed.distributed_c10d._world.tags_to_pg[""].remove(pg)
            except Exception:
                pass
        _unregister_process_group(pg.group_name)


def _shutdown_backend(pg):
    """
    Try to shut down the backend of a process group.
    """
    backend = None
    try:
        backend = pg._get_backend(torch.device("mlu"))
    except RuntimeError:
        pass
    if is_cncl_available() and isinstance(backend, ProcessGroupCNCL):
        # explicitly call shutdown to ensure that CNCL resources are released
        backend._shutdown()


def destroy_process_group(group: Optional[ProcessGroup] = None):
    """
    Destroy a given process group, and deinitialize the distributed package.

    Args:
        group (ProcessGroup, optional): The process group to be destroyed, if
                                        group.WORLD is given, all process
                                        groups including the default one will
                                        be destroyed.
    """

    if group == GroupMember.NON_GROUP_MEMBER:
        return

    if group is None:
        pg = GroupMember.WORLD
    else:
        pg = group

    assert pg is not None
    if torch.distributed.distributed_c10d._world.pg_map.get(pg, None) is None:
        raise ValueError("Invalid process group specified")

    # When users register Python onCompletion hooks, those hooks will run on a
    # different thread than the main thread. Today, the ProcessGroup dtor does
    # wait for that thread. However, the dtor might finish after the Python
    # Interpreter exits. After that grabbing the GIL for the Python hook will crash.
    # We can either revive the interpreter when running hooks or keep the main one
    # alive until all works and hooks are done. The current implementation does the
    # latter. Therefore, we explicitly call _wait_for_pending_works() here to wait
    # for the pending hooks to finish.
    if pg.name().lower() == "custom" and pg._has_hooks():
        pg._wait_for_pending_works()

    if group is None or group == GroupMember.WORLD:
        for pg_to_shutdown in sorted(
            torch.distributed.distributed_c10d._world.pg_names,
            key=lambda x: torch.distributed.distributed_c10d._world.pg_names[x],
            reverse=True,
        ):
            _shutdown_backend(pg_to_shutdown)

        _update_default_pg(None)
        torch.distributed.distributed_c10d._world.pg_map.clear()
        torch.distributed.distributed_c10d._world.pg_names.clear()
        torch.distributed.distributed_c10d._world.pg_group_ranks.clear()
        torch.distributed.distributed_c10d._world.pg_backend_config.clear()
        torch.distributed.distributed_c10d._world.pg_to_tag.clear()
        torch.distributed.distributed_c10d._world.tags_to_pg.clear()
        torch.distributed.distributed_c10d._world.pg_coalesce_state.clear()
        _unregister_all_process_groups()

        # when process group doesn't have an explicit name (only WORLD (default)
        # process group can have an explicit name), we use global torch.distributed.distributed_c10d._world.group_count
        # to generate the name. We need to reset the counter on destruction to
        # allow consistent value to be generated when we re-create process
        # groups after some trainers recover from failure
        #
        # We only reset this when WORLD is being destroyed because if this
        # process group is in good state, we aren't dealing with failures.
        torch.distributed.distributed_c10d._world.group_count = 0
    else:
        _shutdown_backend(pg)
        del torch.distributed.distributed_c10d._world.pg_map[pg]
        del torch.distributed.distributed_c10d._world.pg_names[pg]
        del torch.distributed.distributed_c10d._world.pg_group_ranks[pg]
        del torch.distributed.distributed_c10d._world.pg_backend_config[pg]
        if pg in torch.distributed.distributed_c10d._world.pg_coalesce_state.keys():
            warnings.warn(
                "Some coalesced collectives haven't been launched when "
                "ProcessGroup is destroyed. They will be cleaned."
            )
            del torch.distributed.distributed_c10d._world.pg_coalesce_state[pg]

        tag = torch.distributed.distributed_c10d._world.pg_to_tag.get(pg)
        del torch.distributed.distributed_c10d._world.pg_to_tag[pg]
        if tag is not None:
            try:
                torch.distributed.distributed_c10d._world.tags_to_pg[tag].remove(pg)
                if tag.startswith("ptd:"):
                    torch.distributed.distributed_c10d._world.tags_to_pg[""].remove(pg)
            except Exception:
                pass
        _unregister_process_group(pg.group_name)


def _set_pg_timeout(timeout: timedelta, group: Optional[ProcessGroup] = None) -> None:
    if group is None:
        group = _get_default_group()
    if _rank_not_in_group(group):
        raise ValueError("Invalid process group specified")
    assert isinstance(group, ProcessGroup)
    devices = group._device_types
    backends = set()
    if torch.device("cpu") in devices and is_gloo_available():
        backend = group._get_backend(torch.device("cpu"))
        if isinstance(backend, ProcessGroupGloo):
            backends.add(backend)
    if torch.device("mlu") in devices:
        backend = group._get_backend(torch.device("mlu"))
        if is_cncl_available() and isinstance(backend, ProcessGroupCNCL):
            backends.add(backend)  # type: ignore[arg-type]
        elif is_gloo_available() and isinstance(backend, ProcessGroupGloo):
            backends.add(backend)  # type: ignore[arg-type]
    if len(backends) == 0:
        warnings.warn("Set timeout is now only supported for either cncl or gloo.")
    for backend in backends:
        backend._set_default_timeout(timeout)


def _get_default_timeout(backend: Backend) -> timedelta:
    if backend == Backend.CNCL:
        if not isinstance(torch.mlu.default_pg_cncl_timeout, timedelta):
            warnings.warn(
                "Attempted to get default timeout for cncl backend, but CNCL support is not compiled"
            )
            return default_pg_timeout
        return torch.mlu.default_pg_cncl_timeout
    else:
        return default_pg_timeout


def _add_ephemeral_timeout_for_all_pgs(timeout: timedelta) -> None:
    for pg in torch.distributed.distributed_c10d._world.pg_map.keys():
        devices = pg._device_types
        if torch.device("mlu") in devices:
            backend = pg._get_backend(torch.device("mlu"))
            if is_cncl_available() and isinstance(backend, ProcessGroupCNCL):
                backend._add_ephemeral_timeout(timeout)


def _set_sequence_number_for_group(self):
    device = torch.device(self._device_types[0])
    pg = self._get_backend(device)
    pg._set_sequence_number_for_group()


def _get_sequence_number_for_group(self):
    device = torch.device(self._device_types[0])
    pg = self._get_backend(device)
    return pg._get_sequence_number_for_group()


torch.distributed.distributed_c10d._get_default_timeout.__code__ = (
    _get_default_timeout.__code__
)

torch.distributed.__setattr__("destroy_process_group", destroy_process_group)
torch.distributed.__setattr__("batch_isend_irecv", batch_isend_irecv)
torch.distributed.__setattr__("_shutdown_backend", _shutdown_backend)

torch.distributed.distributed_c10d.batch_isend_irecv = batch_isend_irecv
torch.distributed.distributed_c10d.destroy_process_group = destroy_process_group
torch.distributed.distributed_c10d._shutdown_backend = _shutdown_backend
torch.distributed.distributed_c10d._add_ephemeral_timeout_for_all_pgs = (
    _add_ephemeral_timeout_for_all_pgs
)
torch.distributed.distributed_c10d._set_pg_timeout = _set_pg_timeout

# this monkey patch will be remove until the PR(https://github.com/pytorch/pytorch/pull/124138) is merged.
setattr(
    torch.distributed.ProcessGroup,
    "_set_sequence_number_for_group",
    _set_sequence_number_for_group,
)
setattr(
    torch.distributed.ProcessGroup,
    "_get_sequence_number_for_group",
    _get_sequence_number_for_group,
)


def can_convert_to_int(s: str) -> bool:
    try:
        int(s)
        return True
    except Exception:
        return False


def create_pg(dist_backend_opts, pg_opt):
    pg_options = ProcessGroupCNCL.Options()
    pg_options.group_name = dist_backend_opts.group_id
    pg_options._timeout = dist_backend_opts.timeout
    reset_mlulink_timeout = os.getenv("TORCH_RESET_MLULINK_TIMEOUT", "1")
    mlulink_timeout = os.getenv("CNCL_MLULINK_TIMEOUT_SECS")
    if reset_mlulink_timeout == "1":
        if mlulink_timeout == None:
            # CNCL_MLULINK_TIMEOUT_SECS is default 3600s
            mlulink_timeout = "3600"

        if mlulink_timeout != "-1":
            if can_convert_to_int(mlulink_timeout):
                mlulink_timedelta = timedelta(seconds=int(mlulink_timeout))
                if mlulink_timedelta < pg_options._timeout:
                    warnings.warn(
                        "ProcessGroupCNCL detected CNCL_MLULINK_TIMEOUT_SECS less than watchdog timeout setting, force CNCL_MLULINK_TIMEOUT_SECS 2 seconds more than watchdog timeout!"
                    )
                    os.environ["CNCL_MLULINK_TIMEOUT_SECS"] = str(
                        int(pg_options._timeout.total_seconds() + 2)
                    )
            else:
                warnings.warn(
                    "ProcessGroupCNCL detected CNCL_MLULINK_TIMEOUT_SECS an invalid value, set CNCL_MLULINK_TIMEOUT_SECS 2 seconds more than pg timeout!"
                )
                os.environ["CNCL_MLULINK_TIMEOUT_SECS"] = str(
                    int(pg_options._timeout.total_seconds() + 2)
                )

    pg_options.global_ranks_in_group = dist_backend_opts.global_ranks_in_group
    if pg_opt is not None:
        pg_options.is_high_priority_stream = pg_opt.is_high_priority_stream
    return torch_mlu._MLUC.ProcessGroupCNCL(
        dist_backend_opts.store,
        dist_backend_opts.group_rank,
        dist_backend_opts.group_size,
        pg_options,
    )


def _abort_backend(pg: ProcessGroup):
    try:
        backend = pg._get_backend(torch.device("mlu"))
    except RuntimeError:
        backend = None
    if isinstance(backend, ProcessGroupCNCL):
        backend.abort()


torch.distributed.distributed_c10d._abort_backend = _abort_backend
torch.distributed.distributed_c10d._abort_process_group = _abort_process_group


def _new_process_group_helper(
    group_size,
    group_rank,
    global_ranks_in_group,
    backend,
    store,
    group_name,
    backend_options=None,
    timeout=None,
    pg_tag=None,
    device_id=None,
    group_desc=None,
):
    """
    Create a new distributed process group.

    This function must be called by ALL processes in the global group, even if
    the calling process is not part of the newly created group. In that case,
    this function returns GroupMember.NON_GROUP_MEMBER.

    This function is called with ``global_ranks_in_group == []`` for the default group.
    """
    # modify by Cambricon: we need to access global var _world from module directly
    #                      avoiding user change this object
    # global _world
    _world = torch.distributed.distributed_c10d._world
    # end Cambricon

    if group_name in _world.pg_names.values():
        raise ValueError(
            "The specified group name has already been "
            "created, please use a different group name"
        )

    if device_id is not None and (device_id.index is None or device_id.type == "cpu"):
        raise ValueError(
            "init_process_group device_id parameter must be an accelerator with an index"
        )

    # Note: _new_process_group_helper is only called from init_process_group, which always provides a timeout value
    _check_valid_timeout(timeout)

    if pg_tag not in [None, ""]:
        # creating with the same tag and rank set results in the same underlying PG
        existing_group = _find_pg_by_ranks_and_tag(pg_tag, global_ranks_in_group)
        if existing_group:
            _, prefix_store = _world.pg_map[existing_group]
            return existing_group, prefix_store

    group_desc = "undefined" if group_desc is None else group_desc

    # The list of group ranks is empty if we're creating the default group.
    is_default_group = len(global_ranks_in_group) == 0

    # nccl and potentially other backends allow creation of
    # communicators based on pre-existing ones, which can save
    # initialization time.  Due to lazy initialization of
    # communicators in some backends, we have to be careful and only
    # split when we *know* the default PG has already started communicator initialization.
    # We know this if we have bound a device id to the default pg (eager initialized).
    # Modify by CAMBRICON
    # if is_initialized() and _get_default_group().bound_device_id:
    #    split_from = _get_split_source(_get_default_group())
    # else:
    #    split_from = None
    split_from = None
    # End Modify by CAMBRICON

    # If this is a subgroup (which means group_ranks is specified),
    # we check if the current process is a member of the new group.
    if not is_default_group:
        global_rank = _get_default_group().rank()
        if global_rank not in global_ranks_in_group:
            # If we are using `ncclCommSplit` (or similar split from
            # other APIs) to create the communicator, we will need to
            # call `ncclCommSplit` on *all* ranks in this new group's
            # parent group, even those not in the new group.  This is
            # a requirement of the NCCL API as otherwise we would get
            # out of sync.
            if split_from:
                split_from.perform_nocolor_split(_get_default_group().bound_device_id)
            return GroupMember.NON_GROUP_MEMBER, None

    prefix_store = PrefixStore(f"{group_name}/", store)
    # The backend for PG will be set later based on what's inside BackendConfig
    # and timeout are set in each backend's option.
    pg: ProcessGroup = ProcessGroup(
        prefix_store,
        group_rank,
        group_size,
    )
    backend_config = BackendConfig(backend)
    # Set the default backend when single backend is passed in.
    if "," not in str(backend) and ":" not in str(backend):
        assert backend in Backend.backend_type_map, f"Unknown backend type {backend}"
        if backend == Backend.UNDEFINED:
            # Currently when backend is UNDEFINED, both ``gloo`` and ``nccl`` backends
            # will be created, we use nccl(if cuda is available) or gloo as default
            # backend so we can correctly call getDefaultBackend which in ProcessGroup.

            # modify by Cambricon: use customm backendtype when backen is cncl
            backend_map = backend_config.get_device_backend_map()
            if Backend.CNCL in backend_map.values():
                pg._set_default_backend(ProcessGroup.BackendType.CUSTOM)
            # if Backend.NCCL in backend_config.get_device_backend_map().values():
            elif Backend.NCCL in backend_map.values():
                # end Cambricon
                pg._set_default_backend(ProcessGroup.BackendType.NCCL)
            else:
                pg._set_default_backend(ProcessGroup.BackendType.GLOO)
        else:
            pg._set_default_backend(Backend.backend_type_map[backend])
    # In order to correctly call pg._has_hooks(), we should set the default backend
    # when multi backend is passed in
    else:
        if Backend.NCCL in backend_config.device_backend_map.values():
            pg._set_default_backend(ProcessGroup.BackendType.NCCL)
        elif Backend._plugins.keys():
            custom_backend = next(iter(Backend._plugins.keys()))
            if custom_backend in backend_config.device_backend_map.values():
                pg._set_default_backend(ProcessGroup.BackendType.CUSTOM)
        else:
            pg._set_default_backend(ProcessGroup.BackendType.GLOO)

    if device_id:
        pg.bound_device_id = device_id
    backend_class: torch._C._distributed_c10d.Backend
    for device, backend_str in backend_config.get_device_backend_map().items():
        # Use the group name as prefix in the default store, such that
        # a single store can be reused by multiple groups.
        backend_prefix_store = PrefixStore(f"{device}/", prefix_store)

        if backend_str == Backend.MPI:
            if not is_mpi_available():
                raise RuntimeError(
                    "Distributed package doesn't have MPI built in."
                    " MPI is only included if you build PyTorch from"
                    " source on a host that has MPI installed."
                )
            # modify by Cambricon: only mpi available then can import processgroupMpi
            from torch.distributed.distributed_c10d import ProcessGroupMPI

            # end Cambricon

            backend_class = ProcessGroupMPI.create(global_ranks_in_group)
            backend_type = ProcessGroup.BackendType.MPI
            if not backend_class:
                return GroupMember.NON_GROUP_MEMBER, None
            # create new process group with accurate rank and size
            if pg.rank() == -1 and pg.size() == -1:
                pg = ProcessGroup(
                    backend_prefix_store,
                    backend_class.rank(),
                    backend_class.size(),
                )
                pg._set_default_backend(backend_type)
        elif backend_str == Backend.GLOO:
            # TODO: remove this check after lazy initialization is supported
            # if pg_options is not None:
            #     raise RuntimeError("GLOO options not supported")
            if not is_gloo_available():
                raise RuntimeError("Distributed package doesn't have Gloo built in")
            backend_class = ProcessGroupGloo(
                backend_prefix_store, group_rank, group_size, timeout=timeout
            )
            backend_class.options.global_ranks_in_group = global_ranks_in_group
            backend_class.options.group_name = group_name
            backend_type = ProcessGroup.BackendType.GLOO
        elif backend_str == Backend.NCCL:
            if not is_nccl_available():
                raise RuntimeError("Distributed package doesn't have NCCL built in")
            # modify by Cambricon: only nccl available then can import ProcessGroupNCCL
            from torch._C._distributed_c10d import ProcessGroupNCCL

            # end Cambricon
            if backend_options is not None:
                assert isinstance(
                    backend_options, ProcessGroupNCCL.Options
                ), "Expected backend_options argument to be of type ProcessGroupNCCL.Options"
                if backend_options._timeout != timeout:
                    warnings.warn(
                        "backend_options._timeout was specified, "
                        "but timeout kwarg has a default value that will always override it. "
                    )
            else:
                # default backend_options for NCCL
                backend_options = ProcessGroupNCCL.Options()
                backend_options.is_high_priority_stream = False
            backend_options._timeout = timeout

            if split_from:
                backend_options.split_from = split_from
                backend_options.split_color = _process_group_color(
                    global_ranks_in_group
                )
            backend_options.global_ranks_in_group = global_ranks_in_group
            backend_options.group_name = group_name
            backend_class = ProcessGroupNCCL(
                backend_prefix_store, group_rank, group_size, backend_options
            )
            backend_type = ProcessGroup.BackendType.NCCL
        elif backend_str == Backend.UCC and is_ucc_available():
            # modify by Cambricon: only nccl available then can import ProcessGroupUCC
            from torch._C._distributed_c10d import ProcessGroupUCC

            # end Cambricon

            # TODO: once UCC plugin is fully deprecated, remove
            # is_ucc_available() from above elif-condition and raise
            # RuntimeError if is_ucc_available() returns false.

            backend_class = ProcessGroupUCC(
                backend_prefix_store, group_rank, group_size, timeout=timeout
            )
            backend_type = ProcessGroup.BackendType.UCC
        elif backend_str == Backend.XCCL:
            if not is_xccl_available():
                raise RuntimeError("Distributed package doesn't have XCCL built in")

            # modify by Cambricon: only nccl available then can import ProcessGroupUCC
            from torch._C._distributed_c10d import ProcessGroupXCCL

            backend_class = ProcessGroupXCCL(
                backend_prefix_store, group_rank, group_size
            )
            backend_type = ProcessGroup.BackendType.XCCL
        else:
            assert (
                backend_str.upper() in Backend._plugins
            ), f"Unknown c10d backend type {backend_str.upper()}"

            backend_plugin = Backend._plugins[backend_str.upper()]
            creator_fn = backend_plugin.creator_fn
            extended_api = backend_plugin.extended_api
            backend_type = ProcessGroup.BackendType.CUSTOM

            if not extended_api:
                backend_class = creator_fn(
                    backend_prefix_store, group_rank, group_size, timeout
                )
            else:
                dist_backend_opts = _DistributedBackendOptions()
                dist_backend_opts.store = backend_prefix_store
                dist_backend_opts.group_rank = group_rank
                dist_backend_opts.group_size = group_size
                dist_backend_opts.timeout = timeout
                dist_backend_opts.group_id = group_name
                dist_backend_opts.global_ranks_in_group = global_ranks_in_group

                backend_class = creator_fn(dist_backend_opts, backend_options)

        # Set sequence numbers for gloo and nccl backends.
        if backend_str == Backend.GLOO:
            assert isinstance(backend_class, ProcessGroupGloo)
            backend_class._set_sequence_number_for_group()
        elif backend_str == Backend.NCCL:
            assert isinstance(backend_class, ProcessGroupNCCL)
            backend_class._set_sequence_number_for_group()

        # If the type is a subclass of ProcessGroup then return this process group immediately
        # TODO: This defaults to the old behavior for PythonProcessGroups which overwrites the
        # ProcessGroup instance
        if issubclass(type(backend_class), ProcessGroup):
            pg = backend_class  # type: ignore[assignment]
            break

        # Process group wrapper initialization for supported PGs when TORCH_DISTRIBUTED_DEBUG is set
        if (
            backend_str in [Backend.GLOO, Backend.NCCL, Backend.UCC]
            or backend_str.upper() in Backend._plugins
        ):
            # In debug mode and if GLOO is available, wrap in a wrapper PG that
            # enables enhanced collective checking for debuggability.
            if get_debug_level() == DebugLevel.DETAIL:
                if not _GLOO_AVAILABLE:
                    logger.info(
                        """TORCH_DISTRIBUTED_DEBUG was set to DETAIL, but
                                GLOO is not available. Build with Gloo to
                                create a wrapper process group in debug mode
                                to aid collective desynchronization debugging."""
                    )
                else:
                    backend_class = _create_process_group_wrapper(
                        wrapped_pg=backend_class,
                        store_prefix=group_name,
                        store=backend_prefix_store,
                        rank=group_rank,
                        world_size=group_size,
                        timeout=timeout,
                    )

        # register only a single backend when all get_device_backend_map values are the same
        if len(set(backend_config.get_device_backend_map().values())) == 1:
            for device in backend_config.get_device_backend_map().keys():
                pg._register_backend(torch.device(device), backend_type, backend_class)

            # break out of outer loop to not create any more backends
            break

        pg._register_backend(torch.device(device), backend_type, backend_class)

    # set group_name and group_dsec to backend
    assert group_name is not None
    assert group_desc is not None
    pg._set_group_name(group_name)
    pg._set_group_desc(group_desc)

    if device_id and pg._get_backend(device_id).supports_splitting:
        eager_backend = pg._get_backend(device_id)
        eager_backend.eager_connect_single_device(device_id)

    # update global state
    _world.pg_map[pg] = (backend, prefix_store)
    _world.pg_names[pg] = group_name
    _register_process_group(group_name, pg)

    _world.pg_backend_config[pg] = str(backend_config)
    # "" is the default tag for user PGs
    if pg_tag in [None, ""]:
        pg_tag = f"ptd:{group_name}"
        _world.tags_to_pg.setdefault("", []).append(pg)
    else:
        pg_tag = f"user:{pg_tag}"

    _world.tags_to_pg.setdefault(pg_tag, []).append(pg)
    _world.pg_to_tag[pg] = pg_tag
    return pg, prefix_store


torch.distributed.distributed_c10d._new_process_group_helper = _new_process_group_helper

if not hasattr(torch.distributed.Backend, "CNCL"):
    torch.distributed.Backend.register_backend(
        "cncl", create_pg, extended_api=True, devices=["mlu"]
    )
torch.distributed.Backend.backend_capability["gloo"] = ["cpu", "mlu"]


def _cncl_factory(
    store,
    rank,
    world_size,
    timeout,
    device,
    **kwargs,
) -> ProcessGroup:
    from torch_mlu._MLUC import ProcessGroupCNCL

    opts = ProcessGroupCNCL.Options()
    opts._timeout = timeout
    for k, v in kwargs.items():
        if not hasattr(opts, k):
            raise KeyError(f"Unknown option {k}")
        setattr(opts, k, v)

    backend_class = ProcessGroupCNCL(store, rank, world_size, opts)
    backend_class._set_sequence_number_for_group()
    backend_class.eager_connect_single_device(device)

    pg = ProcessGroup(store, rank, world_size)
    pg._set_default_backend(ProcessGroup.BackendType.CUSTOM)
    pg._register_backend(device, ProcessGroup.BackendType.CUSTOM, backend_class)

    return pg


import torch.distributed._dist2 as dist2

if "cncl" not in dist2._BACKENDS:
    dist2.register_backend("cncl", _cncl_factory)

if hasattr(torch_mlu._MLUC, "ProcessGroupCNCL"):
    from torch_mlu._MLUC import ProcessGroupCNCL

    torch.distributed.__setattr__("ProcessGroupCNCL", ProcessGroupCNCL)
    ProcessGroupCNCL.__module__ = "torch.distributed.distributed_c10d"
