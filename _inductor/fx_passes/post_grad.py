import torch
from torch._inductor.fx_passes import post_grad

from ...utils import gorilla


def should_prefer_unfused_addmm(match):
    # Modify by CAMBRICON
    # inp = match.kwargs["inp"]
    # if not is_gpu(inp.meta["val"].device.type):
    #     return False

    # output = match.output_node()
    # return all(is_pointwise_use(use) for use in output.users)
    return False
    # end Modify by CAMBRICON


patch = gorilla.Patch(
    post_grad,
    "should_prefer_unfused_addmm",
    should_prefer_unfused_addmm,
)
gorilla.apply(patch)


def post_grad_passes(gm: torch.fx.GraphModule, is_inference: bool):
    """
    Passes that run on after grad.  This is called once on the forwards
    graph and once on the backwards graph.

    The IR here has been normalized and functionalized.
    """
    GraphTransformObserver = functools.partial(
        torch.fx.passes.graph_transform_observer.GraphTransformObserver,
        subsystem="post_grad_passes",
    )

    if not torch._dynamo.config.skip_fsdp_hooks:
        remove_fsdp2_unsharded_param_graph_input_usage(gm.graph)

    if config.dce:
        # has some issues with mutation in inference mode
        gm.graph.eliminate_dead_code()

    if is_inference and config.reorder_for_locality:
        GraphTransformObserver(gm, "reorder_for_locality").apply_graph_pass(
            reorder_for_locality
        )

    fake_tensor_updater = FakeTensorUpdater(gm.graph)

    if post_grad_custom_pre_pass := config.post_grad_custom_pre_pass:
        GraphTransformObserver(gm, "post_grad_custom_pre_pass").apply_graph_pass(
            post_grad_custom_pre_pass
        )

    if torch._C._has_mkldnn:
        if (
            config.cpp.enable_grouped_gemm_template
            and config.max_autotune
            and "CPP" in config.max_autotune_gemm_backends
        ):
            from .mkldnn_fusion import grouped_gemm_pass

            grouped_gemm_pass(gm.graph)

        if config.cpp.enable_concat_linear:
            from .quantization import concat_linear_woq_int4

            # Concat linear optimization for WOQ int4
            concat_linear_woq_int4(gm)

    # Remove profiler ops (record_function) to prevent them blocking fusion
    GraphTransformObserver(gm, "remove_profiler_ops").apply_graph_pass(
        _remove_profiler_ops
    )

    if config.pattern_matcher:
        lazy_init()
        GraphTransformObserver(gm, "post_grad_custom_pre_pass").apply_graph_pass(
            functools.partial(group_batch_fusion_passes, pre_grad=False)
        )
        GraphTransformObserver(gm, "remove_noop_ops").apply_graph_pass(remove_noop_ops)
        GraphTransformObserver(gm, "remove_assert_ops").apply_graph_pass(
            remove_assert_ops
        )
        for i, patterns in enumerate(pass_patterns):
            GraphTransformObserver(gm, f"pass_pattern_{i}").apply_graph_pass(
                patterns.apply
            )
        if config.partitioned_scatter_enabled:
            GraphTransformObserver(
                gm, "partitioned_scatter_optimization"
            ).apply_graph_pass(partitioned_scatter_optimization_pass)
        for pass_name in config.post_grad_fusion_options:
            # skip all patterns for group batch fusions or quantization patterns
            if pass_name in POST_GRAD_FUSIONS or pass_name in OPTIMUS_EXCLUDE_POST_GRAD:
                continue
            pattern_matcher_pass = POST_GRAD_PATTERNS[pass_name]
            inductor_before_change = save_inductor_dict(
                [pattern_matcher_pass.pass_name]
            )
            GraphTransformObserver(gm, pass_name).apply_graph_pass(
                pattern_matcher_pass.apply
            )
            if not is_same_dict(counters["inductor"], inductor_before_change):
                trace_structured(
                    "artifact",
                    metadata_fn=lambda: {
                        "name": f"{pattern_matcher_pass.pass_name}_post_grad",
                        "encoding": "string",
                    },
                    payload_fn=lambda: gm.print_readable(
                        print_output=False, include_stride=True, include_device=True
                    ),
                )
        if config.b2b_gemm_pass:
            B2B_GEMM_PASS.apply(gm.graph)  # type: ignore[arg-type]

    if config._micro_pipeline_tp:
        micro_pipeline_tp_pass(gm.graph)

    if config._fuse_ddp_communication:
        GraphTransformObserver(gm, "fuse_ddp_communication").apply_graph_pass(
            lambda graph: fuse_ddp_communication(
                graph,
                config._fuse_ddp_communication_passes,
                config._fuse_ddp_bucket_size,
            )
        )

    if post_grad_custom_post_pass := config.post_grad_custom_post_pass:
        # Modify by CAMBRICON pass throuph "is_inference" arg to mlu
        from torch._inductor.custom_graph_pass import CustomGraphPass

        if (
            isinstance(post_grad_custom_post_pass, CustomGraphPass)
            and "MLU" in type(post_grad_custom_post_pass).__name__
        ):
            GraphTransformObserver(gm, "post_grad_custom_post_pass").apply_graph_pass(
                functools.partial(post_grad_custom_post_pass, is_inference=is_inference)
            )
        else:
            GraphTransformObserver(gm, "post_grad_custom_post_pass").apply_graph_pass(
                post_grad_custom_post_pass
            )
        # end Modify by CAMBRICON

    GraphTransformObserver(gm, "stable_sort").apply_graph_pass(stable_topological_sort)

    GraphTransformObserver(gm, "move_constructors_to_cuda").apply_graph_pass(
        move_constructors_to_gpu
    )

    fake_tensor_updater.incremental_update()

    for device, custom_backend_pass in custom_backend_passes.items():
        if custom_backend_pass is not None:
            gm_devices = [d.type for d in get_all_devices(gm)]
            if device in gm_devices:
                pass_name = "custom_backend_passes_" + device
                GraphTransformObserver(gm, pass_name).apply_gm_pass(custom_backend_pass)

    collectives_bucketing: bool = False

    if config.bucket_reduce_scatters_fx != "none":
        from torch._inductor.fx_passes.bucketing import bucket_reduce_scatter
        from torch._inductor.fx_passes.fsdp import bucket_fsdp_reduce_scatter

        p = (
            bucket_fsdp_reduce_scatter
            if "fsdp" in config.bucket_reduce_scatters_fx
            else bucket_reduce_scatter
        )
        GraphTransformObserver(gm, "bucket_reduce_scatters").apply_graph_pass(
            lambda graph: p(
                graph.owning_module,
                config.bucket_reduce_scatters_fx_bucket_size_determinator,
                config.bucket_reduce_scatters_fx,  # type: ignore[arg-type]
            )
        )
        collectives_bucketing = True

    if config.bucket_all_reduces_fx != "none":
        from torch._inductor.fx_passes.bucketing import bucket_all_reduce

        GraphTransformObserver(gm, "bucket_all_reduce").apply_graph_pass(
            lambda graph: bucket_all_reduce(
                graph.owning_module,
                config.bucket_all_reduces_fx_bucket_size_determinator,
                config.bucket_all_reduces_fx,  # type: ignore[arg-type]
            )
        )
        collectives_bucketing = True

    # Fx all_gather bucketing introduces mutation op
    # Keeping it in the end to keep invariant of functional graph for previous passes.
    if config.bucket_all_gathers_fx != "none":
        from torch._inductor.fx_passes.bucketing import bucket_all_gather
        from torch._inductor.fx_passes.fsdp import bucket_fsdp_all_gather

        p = (
            bucket_fsdp_all_gather  # type: ignore[assignment]
            if "fsdp" in config.bucket_all_gathers_fx
            else bucket_all_gather
        )
        GraphTransformObserver(gm, "bucket_all_gathers").apply_graph_pass(
            lambda graph: p(
                graph.owning_module,
                config.bucket_all_gathers_fx_bucket_size_determinator,
                config.bucket_all_gathers_fx,  # type: ignore[arg-type]
            )
        )
        collectives_bucketing = True

    if collectives_bucketing:
        # Fx collectives bucketing passes require topological sort for the cases:
        # when bucketed collectives have users before the last collective in the bucket
        # AND when inputs of bucketed collective have ancestors after the first collective in the bucket.
        #
        # In this case we can not manually pick the place for bucketed collective insertion.
        # But we are guaranteed by the bucketing (independent collectives in the bucket),
        # that it is possible to reorder nodes to satisfy all ordering requirements.
        #
        # --- before bucketing ---
        # in0 = ...
        # wait_ag0 = ag(in0)
        # user0(wait_ag0)
        # ...
        # pre_in1 = ...
        # in1 = transform(pre_in1)
        # wait_ag1 = ag(in1)
        # user1(wait_ag1)
        #
        # --- after bucketing ---
        #
        # in0 = ...
        # user(wait_ag0) <--- wait_ag0 is defined only after bucketed collective.
        #
        # pre_in1 = ...
        # in1 = transform(pre_in1)
        # ag_bucket(in0+in1)
        # wait_bucket
        # wait_ag0 = wait_bucket[0]
        # wait_ag1 = wait_bucket[1]
        # user1(wait_ag1)
        stable_topological_sort(gm.graph)

    # Apply overlap scheduling if enabled
    if config.aten_distributed_optimizations.enable_overlap_scheduling:
        from torch._inductor.fx_passes.overlap_scheduling import (
            schedule_overlap_bucketing_from_inductor_configs,
        )

        overlap_deps = config.aten_distributed_optimizations.insert_overlap_deps
        fusion_regions = config.aten_distributed_optimizations.enable_fusion_regions

        # by default, insert overlap deps and enable fusion regions within inductor
        with config.patch(
            {
                "aten_distributed_optimizations.insert_overlap_deps": (
                    True if overlap_deps is None else overlap_deps
                ),
                "aten_distributed_optimizations.enable_fusion_regions": (
                    True if fusion_regions is None else fusion_regions
                ),
            }
        ):
            GraphTransformObserver(gm, "overlap_scheduling").apply_graph_pass(
                lambda graph: schedule_overlap_bucketing_from_inductor_configs(
                    graph.owning_module,
                )
            )

    # Keep these last, since they introduce mutation. Look at
    # ./fx_passes/README.md for a discussion of mutation invariants.
    GraphTransformObserver(gm, "reinplace_inplaceable_ops").apply_graph_pass(
        functools.partial(reinplace_inplaceable_ops, fake_tensor_updater),
    )
    GraphTransformObserver(
        gm, "decompose_triton_kernel_wrapper_functional"
    ).apply_graph_pass(decompose_triton_kernel_wrapper_functional)
    GraphTransformObserver(gm, "decompose_auto_functionalized").apply_graph_pass(
        decompose_auto_functionalized
    )
    if not torch._dynamo.config.skip_fsdp_hooks:
        GraphTransformObserver(gm, "reinplace_fsdp_all_gather").apply_graph_pass(
            comms.reinplace_fsdp_all_gather
        )
    GraphTransformObserver(gm, "decompose_scan_to_while_loop").apply_gm_pass(
        decompose_scan_to_while_loop
    )
    GraphTransformObserver(gm, "decompose_map_to_while_loop").apply_gm_pass(
        decompose_map_to_while_loop
    )

    gm.recompile()
    gm.graph.lint()


patch = gorilla.Patch(
    post_grad,
    "post_grad_passes",
    post_grad_passes,
)
gorilla.apply(patch)
