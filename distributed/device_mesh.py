# Patch for DeviceMesh._init_one_process_group to skip the split_group path.
# ProcessGroupCNCL.Options cannot be pickled (copy.deepcopy fails with TypeError),
# and torch_mlu does not support the split_group mechanism.
# Always use the new_group path instead for MLU devices.

from ..utils import gorilla
import torch
from torch._C._distributed_c10d import Backend as C10dBackend
from torch.distributed._mesh_layout import _MeshLayout
from torch.distributed import config as dist_config
from torch.distributed.distributed_c10d import (
    _get_default_group,
    get_backend,
    get_rank,
    get_world_size,
    GroupName,
    new_group,
    split_group,
)

BackendConfig = tuple[str | None, C10dBackend.Options | None]


@gorilla.patch(torch.distributed.DeviceMesh)
def _init_one_process_group(
    sub_layout: _MeshLayout,
    rank_map: torch.Tensor,
    dim_name: str,
    backend_override: BackendConfig,
) -> GroupName | None:
    # Generate a 2D global mesh tensor for the current dim for PG creation.
    pg_ranks_by_dim = sub_layout.nest().remap_to_tensor(rank_map)
    backend, pg_options = backend_override
    # We need to explicitly pass in timeout when specified in option, otherwise
    # the default timeout will be used to override the timeout set in option.
    # TODO: remove this once we have fixed inside c10d level.
    timeout = pg_options._timeout if pg_options else None

    # If we have a 2D mesh with mesh_dim_names ("dp", "tp"), the group description
    # of the subgroups would be `mesh_dim_dp` and `mesh_name_tp`.
    # If the mesh doesn't have a mesh_dim_names, then the group description of the
    # subgroup would be `mesh_dim_0` and `mesh_dim_1`.
    group_desc = f"mesh_{dim_name}"

    dim_group = None
    default_group = _get_default_group()

    # Early return if there is only one sub_layout in the mesh layout.
    if sub_layout.numel() == get_world_size() and backend_override == (
        None,
        None,
    ):
        # Append the default pg to the first dim groups only if the default pg is compatible with `self._device_type`.
        # Otherwise, create new pg.
        ranks = list(range(get_world_size()))
        dim_group = (
            new_group(
                backend=backend,
                ranks=ranks,
                group_desc="mesh_default",
            )
            if torch.cuda.is_available() and get_backend(default_group) == "gloo"
            else default_group
        )
        return dim_group.group_name  # type: ignore[union-attr]

    # Modify by CAMBRICON
    # If bound_device_id exists, it means the nccl communicator has been eagerly initialized
    # so that we can use `split_group` to create subgroups through `ncclCommSplit`.
    # In this case, we only need to make one API call (`split_group``) for the subgroup creation
    # for each mesh dimension. In a 2 * 4 mesh, we only need to make two API calls per ranks to create
    # all the subgroups.
    # Otherwise, we need to make more than one API call (`new_group`) for subgroup creations. The
    # numbers of API calls are equal to the number of subgroups for each mesh dimension. In a 2 * 4
    # mesh, we need to make two API calls per ranks to create all the subgroups.
    # if (
    #    (
    #        getattr(default_group, "bound_device_id", None) is not None
    #        or dist_config.use_torchcomms
    #    )
    #    and torch.cuda.is_available()
    #    and (
    #        backend is None
    #        or default_group._get_backend(torch.device("cuda")).name() == backend
    #    )
    # ):
    #    dim_group = split_group(
    #        parent_pg=default_group,
    #        timeout=timeout,
    #        pg_options=pg_options,
    #        split_ranks=pg_ranks_by_dim.tolist(),
    #        group_desc=group_desc,
    #    )
    #    if dim_group is None:
    #        return None
    #    return dim_group.group_name
    # End Modify by CAMBRICON

    # If the subgroup has been already created through `split_group`, we simply loop over `pg_ranks_by_dim`
    # and append the `group_name` to the `dim_group_names` list when the current rank is in the subgroup.
    # Otherwise, we use `new_group` instead of `split_group` to create subgroups by looping over `pg_ranks_by_dim`
    # along with appending information to the `dim_group_names` list whenever necessary.
    pg_name = None
    for dim_mesh in pg_ranks_by_dim:
        subgroup_ranks = dim_mesh.tolist()
        dim_group = new_group(
            ranks=subgroup_ranks,
            timeout=timeout,
            backend=backend,
            pg_options=pg_options,
            group_desc=group_desc,
        )

        # only add to dim_groups if the current rank in the subgroup
        if get_rank() in subgroup_ranks:
            if pg_name is not None:
                raise RuntimeError(
                    f"Each device mesh dimension should get only one process group, but got {get_rank()} "
                    f"in {subgroup_ranks}!"
                )
            pg_name = dim_group.group_name
    return pg_name
