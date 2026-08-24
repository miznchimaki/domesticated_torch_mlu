from typing import List
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpStrategy,
    PlacementList,
    RuntimeSchemaInfo,
)
from torch.distributed.tensor._ops.utils import (
    expand_to_full_mesh_op_strategy,
    register_op_strategy,
)
from torch.distributed.tensor.placement_types import Placement, Replicate, Shard

aten = torch.ops.aten


# - func: _scaled_dot_product_fused_attention_overrideable
# (Tensor query, Tensor key, Tensor value, Tensor? attn_bias=None, float dropout_p=0.0, bool is_causal=False, bool return_debug_mask=False, *, float? scale=None)
# -> (Tensor output, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, SymInt max_q, SymInt max_k, Tensor philox_seed, Tensor philox_offset, Tensor debug_attn_mask)
@register_op_strategy(
    aten._scaled_dot_product_fused_attention_overrideable.default,
    schema_info=RuntimeSchemaInfo(6),
)
def scaled_dot_product_fused_attention_overrideable_strategy(
    op_schema: OpSchema,
) -> OpStrategy:
    # NOTE: currently we only support some simple strategies to support tensor parallelism
    mesh = op_schema.get_mesh_from_args()
    return_debug_mask = len(op_schema.args_schema) >= 7 and op_schema.args_schema[6]
    q_input_strategy = op_schema.args_schema[0]
    assert isinstance(q_input_strategy, OpStrategy)
    # assuming q/k/v have the same shape

    has_attn_bias = (
        len(op_schema.args_schema) >= 4 and op_schema.args_schema[3] is not None
    )

    single_mesh_dim_strategies: List[PlacementList] = []

    # placement list stores placements of [outputs, inputs]
    # in the spda case, we have 3 valid tensor outputs and 3 or 4 tensor inputs
    # first we can always accept full replication for both inputs and outputs
    all_replicate: PlacementList = [
        # outputs
        Replicate(),
        Replicate(),
        None,  # cum_seq_q
        None,  # cum_seq_k
        None,  # max_q
        None,  # max_k
        None,  # philox_seed
        None,  # philox_offset
        None,  # debug_attn_mask
        # inputs
        Replicate(),
        Replicate(),
        Replicate(),
    ]
    if has_attn_bias:
        all_replicate.append(Replicate())  # attn_bias
    single_mesh_dim_strategies.append(all_replicate)

    # second we can accept the sharding pattern of tensor parallelism, which
    # shard on the num of head dim
    qkv_sharding = Shard(1)  # num head dim
    output_sharding = Shard(1)  # num head dim
    logsumexp_sharding = Shard(1)  # num head dim

    num_heads_dim_sharding: PlacementList = [
        output_sharding,
        logsumexp_sharding,
        None,  # cum_seq_q
        None,  # cum_seq_k
        None,  # max_q
        None,  # max_k
        None,  # philox_seed
        None,  # philox_offset
        None,  # debug_attn_mask
        qkv_sharding,
        qkv_sharding,
        qkv_sharding,
    ]
    if has_attn_bias:
        num_heads_dim_sharding.append(Shard(1))
    single_mesh_dim_strategies.append(num_heads_dim_sharding)

    # Context Parallelism: shards on the sequence dim
    single_mesh_dim_strategies.append(
        [
            Shard(2),  # output
            Shard(2),  # logsumexp
            None,  # cum_seq_q
            None,  # cum_seq_k
            None,  # max_q
            None,  # max_k
            None,  # philox_seed
            None,  # philox_offset
            None,  # debug_attn_mask
            Shard(2),  # q
            Shard(2),  # k
            Shard(2),  # v
        ]
    )

    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=9
    )


# - func: _scaled_dot_product_fused_attention_overrideable_backward
# (Tensor grad_out, Tensor query, Tensor key, Tensor value, Tensor attn_bias, bool[4] grad_input_mask, Tensor out, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, SymInt max_q, SymInt max_k, float dropout_p, bool is_causal, Tensor philox_seed, Tensor philox_offset, *, float? scale=None)
# -> (Tensor grad_query, Tensor grad_key, Tensor grad_value, Tensor grad_attn_bias)
@register_op_strategy(
    aten._scaled_dot_product_fused_attention_overrideable_backward.default
)
def scaled_dot_product_fused_attention_overrideable_backward_strategy(
    op_schema: OpSchema,
) -> OpStrategy:
    mesh = op_schema.get_mesh_from_args(validate=False)

    q_input_strategy = op_schema.args_schema[1]
    assert isinstance(q_input_strategy, OpStrategy)
    # assuming q/k/v have the same shape
    has_attn_bias = op_schema.args_schema[4] is not None

    tensor_input_indices = [
        i
        for i, arg_spec in enumerate(op_schema.args_schema)
        if isinstance(arg_spec, OpStrategy)
    ]
    num_tensor_inputs = len(tensor_input_indices)

    single_mesh_dim_strategies = []

    # placement list stores placements of [outputs, inputs]
    # in the spda backward case, we have 4 tensor outputs and 6 to 11 tensor inputs
    # first we can always accept full replication for both inputs and outputs
    all_replicate: PlacementList = [Replicate()] * (4 + num_tensor_inputs)

    single_mesh_dim_strategies.append(all_replicate)

    # second we can accept the sharding pattern of tensor parallelism, which
    # shard on the num of head dim
    grad_output_sharding = Shard(1)  # num head dim
    qkv_sharding = Shard(1)  # num head dim
    output_sharding = Shard(1)  # num head dim
    logsumexp_sharding = Shard(1)  # num head dim
    grad_qkv_sharding = Shard(1)  # num head dim
    grad_bias_sharding = Shard(1) if has_attn_bias else None

    num_heads_dim_sharding: PlacementList = [
        grad_qkv_sharding,  # grad_q
        grad_qkv_sharding,  # grad_k
        grad_qkv_sharding,  # grad_v
        grad_bias_sharding,  # grad_bias
        grad_output_sharding,  # grad_output
        qkv_sharding,  # q
        qkv_sharding,  # k
        qkv_sharding,  # v
        # the place for optional input attn_bias
        output_sharding,  # output
        logsumexp_sharding,  # logsumexp
    ]
    # accept replicate on the rest tensor inputs, potentially
    # cum_seq_q, cum_seq_k, philox_seed, philox_offset
    # at indices 8, 9, 15, 16, respectively
    num_heads_dim_sharding.extend(
        [Replicate()] * (num_tensor_inputs - (7 if has_attn_bias else 6))
    )
    # input sharding of attn_bias on heads dim if present
    if has_attn_bias:
        num_heads_dim_sharding.insert(8, Shard(1))
    single_mesh_dim_strategies.append(num_heads_dim_sharding)

    # Context Parallelism: shards on the sequence dim
    seq_dim_sharding: PlacementList = [
        Shard(2),  # grad_q
        Shard(2),  # grad_k
        Shard(2),  # grad_v
        Shard(1) if has_attn_bias else None,  # grad_bias
        Shard(2),  # grad_output
        Shard(2),  # q
        Shard(2),  # k
        Shard(2),  # v
        # the place for optional input attn_bias
        Shard(2),  # output
        Shard(2),  # logsumexp
    ]
    # accept replicate on the rest tensor inputs, potentially
    # cum_seq_q, cum_seq_k, philox_seed, philox_offset
    # at indices 8, 9, 15, 16, respectively
    seq_dim_sharding.extend(
        [Replicate()] * (num_tensor_inputs - (7 if has_attn_bias else 6))
    )
    if has_attn_bias:
        num_heads_dim_sharding.insert(8, Shard(1))
    single_mesh_dim_strategies.append(seq_dim_sharding)

    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=4
    )
