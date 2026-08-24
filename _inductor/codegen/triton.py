import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
from packaging import version
import textwrap
from typing import (
    Any,
    cast,
    Dict,
    Iterable,
    Optional,
    Set,
    Sequence,
    Union,
    Callable,
    Counter,
)

import os
import sympy
from sympy.printing.precedence import PRECEDENCE


import torch
import torch.utils._pytree as pytree
from functools import lru_cache
from torch._dynamo.utils import preserve_rng_state
from torch._inductor import config, ir, scheduler, utils
from torch._inductor.codecache import PyCodeCache
from torch.utils._ordered_set import OrderedSet
from torch._dynamo.utils import identity
from torch._inductor.codegen.block_analysis import BlockPatternMatcher
from torch._dynamo.device_interface import get_interface_for_device
from torch._inductor.codegen.common import (
    ArgName,
    CSEVariable,
    DeferredLine,
    IndentedBuffer,
    SizeArg,
    WorkspaceArg,
    RemovedArg,
    InplacedBuffer,
    WorkspaceZeroMode,
    ConstexprArg,
)
from torch._inductor.shape_propagation import BlockShapeType

from torch._inductor.codegen.simd import (
    CandidateTiling,
    constant_repr,
    IterationRanges,
    IterationRangesEntry,
    IterationRangesRoot,
    SIMDScheduling,
    SIMDKernel,
    EnableReduction,
    perf_hint_log,
    prefix_is_reduction,
)
from torch._inductor.codegen.triton import (
    BlockParameters,
    BlockPtrOptions,
    BlockDescriptorOptions,
    TensorDescriptorOptions,
    gen_attr_descriptor_import,
    get_triton_reduction_function,
    IndexingOptions,
    is_sympy_integer_like,
    maybe_upcast_float32,
    triton_acc_type,
    triton_compute_type,
    triton_reshape,
    triton_store_type,
    TritonCSEVariable,
    TritonKernel,
    TritonKernelOverrides,
    TritonScheduling,
    TritonOverrides,
    TritonSymbols,
    upcast_acc_dtype,
    TMACompatibilityChecker,
    TritonPrinter,
)

from torch._inductor.codegen.triton_utils import (
    config_of,
    equal_1_arg_indices,
    non_constexpr_signature,
    signature_of,
    signature_to_meta,
    should_unwrap_unspec_arg,
)
from torch._inductor.dependencies import MemoryDep, StarDep
from torch._inductor.runtime.hints import DeviceProperties, ReductionHint
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.runtime.runtime_utils import next_power_of_2

from torch._inductor.utils import (
    cache_on_self,
    DelayReplaceLine,
    is_welford_reduction,
    Placeholder,
    sympy_dot,
    sympy_product,
    sympy_subs,
    triton_version_uses_attrs_dict,
)
from torch._inductor.virtualized import ReductionType, V
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import CeilDiv, FloorDiv, ModularIndexing
from torch.utils._sympy.symbol import prefix_str, symbol_is_type, SymT
from torch.utils._sympy.value_ranges import ValueRanges
from ...utils import gorilla

log = logging.getLogger(__name__)


# Threshold for detecting inner reductions based on tiling score ratio.
# If r0_tiling_score / x_tiling_score >= this value, upgrade DEFAULT hint to INNER.
INNER_REDUCTION_RATIO_THRESHOLD = 8


def __init__(
    self,
    name: str,
    bounds: ValueRanges[Any],
    dtype: torch.dtype,
    shape: BlockShapeType = None,
) -> None:
    # Modify by CAMBRICON
    # super().__init__(name, bounds, dtype, shape=shape)
    super(TritonCSEVariable, self).__init__(name, bounds, dtype, shape=shape)
    # end Modify by CAMBRICON
    # We'll use this to track which masks the variable needs when used for indirect indexing
    self.mask_vars: OrderedSet[str] = OrderedSet()
    assert dtype is not None, "TritonCSEVariable must have dtype"
    # Modify by CAMBRICON
    # TODO: uncomment this and fix the mlu custom triton kernels
    # assert shape is not None, "TritonCSEVariable must have shape"
    # end Modify by CAMBRICON


patch = gorilla.Patch(
    torch._inductor.codegen.triton.TritonCSEVariable,
    "__init__",
    __init__,
    settings=gorilla.Settings(use_replace_references=True),
)
gorilla.apply(patch)


# add one line: import torch_mlu
@classmethod
@lru_cache(None)
def gen_common_triton_imports(cls) -> str:
    imports = IndentedBuffer()
    imports.splice(
        """
        import triton
        import triton.language as tl
        """
    )
    try:
        import triton.language.extra.tlx  # noqa: F401

        imports.splice(
            """
           import triton.language.extra.tlx as tlx  # noqa: F401
           """
        )
    except ImportError:
        pass
    if attr_desc := gen_attr_descriptor_import():
        imports.writeline(attr_desc)

    imports.splice(
        """
        # Modify by CAMBRICON
        import torch
        import torch_mlu
        # end Modify by CAMBRICON
        from torch._inductor.runtime import triton_helpers, triton_heuristics
        from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
        from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
        """
    )
    if config.triton.proton_profiling:
        imports.splice(
            """
            import triton.profiler as proton
            import triton.profiler.language as pl
            pl.enable_semantic('triton')
            """
        )

    return imports.getvalue()


patch = gorilla.Patch(
    torch._inductor.codegen.triton.TritonKernel,
    "gen_common_triton_imports",
    gen_common_triton_imports,
)
gorilla.apply(patch)


# Add libdevice functions that are not natively supported by the community.
class MLUTritonOverrides(TritonOverrides):
    @staticmethod
    def gelu(x):
        return f"triton.language.extra.mlu.libdevice.ultra_gelu({x})"

    @staticmethod
    @maybe_upcast_float32()
    def sigmoid(x):
        from torch_mlu._inductor import config as inductor_config

        if inductor_config.use_ultra_sigmoid and inductor_config.use_ultra_math:
            return f"triton.language.extra.mlu.libdevice.ultra_sigmoid({x})"
        else:
            return f"triton.language.extra.mlu.libdevice.fast_sigmoid({x})"

    @staticmethod
    def minimum(a, b):
        # Modify by CAMBRICON
        # return f"triton_helpers.minimum({a}, {b})"
        return f"triton.language.minimum({a}, {b}, propagate_nan = triton.language.PropagateNan.ALL)"

    @staticmethod
    @maybe_upcast_float32()
    def tanh(x):
        from torch_mlu._inductor import config as inductor_config

        if inductor_config.use_ultra_tanh and inductor_config.use_ultra_math:
            return f"triton.language.extra.mlu.libdevice.ultra_tanh({x})"
        else:
            return f"triton.language.extra.mlu.libdevice.fast_tanh({x})"

    @staticmethod
    def maximum(a, b):
        return f"triton.language.maximum({a}, {b}, propagate_nan = triton.language.PropagateNan.ALL)"

    @staticmethod
    @maybe_upcast_float32()
    def erf(x):
        from torch_mlu._inductor import config as inductor_config

        # libdevice.erf is used by default to align with CNNL's precision.
        # For inference, it is recommended to switch to libdevice.fast_erf.
        # To control the codegen of all op via use_ultra_math, reduce configuration complexity,
        # the config here utilizes the use_ultra_erf field.
        if inductor_config.use_ultra_erf and inductor_config.use_ultra_math:
            return f"triton.language.extra.mlu.libdevice.fast_erf({x})"
        else:
            return f"triton.language.extra.mlu.libdevice.erf({x})"

    @staticmethod
    @maybe_upcast_float32()
    def log(x):
        return f"libdevice.log({x})"

    @staticmethod
    @maybe_upcast_float32()
    def silu(x):
        return f"triton.language.extra.mlu.libdevice.ultra_silu({x})"

    @staticmethod
    def div(a, b):
        return f"triton.language.extra.mlu.libdevice.fast_dividef({a}, {b})"

    @staticmethod
    @maybe_upcast_float32()
    def exp(x):
        return f"libdevice.exp({x})"

    # @staticmethod
    # @maybe_upcast_float32()
    # def sqrt(x):
    #     import triton
    #     TRITON_VERSION = triton.__version__
    #     if version.parse(TRITON_VERSION) < version.parse("3.4.0"):
    #         return f"triton.language.extra.mlu.libdevice.accurate_sqrt({x})"
    #     else:
    #         return f"triton.language.extra.mlu.libdevice.sqrt({x})"

    @staticmethod
    @maybe_upcast_float32()
    def pow(a, b):
        return f"libdevice.fast_powf({a}, {b})"


MLUTritonOverrides._initialize_pointwise_overrides("triton")
for name, attr in MLUTritonOverrides.__dict__.items():
    if not name.startswith("_"):
        setattr(TritonOverrides, name, attr)


class MluTritonKernel(TritonKernel):
    # For mlu, we always keep the mask.
    def filter_masks(self, mask_vars):
        pass

    def needs_yz_grid_overflow(self, entry: IterationRangesRoot) -> bool:
        return False

    def codegen_block_ptr_store_line(self, name, indexing, block_ptr, value, other=""):
        # Stores require an explicit broadcast. We do this in two phases:
        #  1. Broadcast the operand to the final shape of the range trees, e.g. [ZBLOCK,
        #     YBLOCK, XBLOCK]. This protects against implicit broadcasting from loads.
        #  2. In case the block pointer / tma descriptor has different dimensionality, broadcast/reshape the
        #     result to the shape of the pointer.
        value = f"tl.broadcast_to({value}, {indexing.final_shape})"

        # These dims no longer need broadcasting.
        for idx, (dim, broadcast_dim) in enumerate(
            zip(indexing.final_shape, indexing.broadcast_shape)
        ):
            if V.graph.sizevars.statically_known_equals(dim, broadcast_dim):
                indexing.broadcasting_dims[idx] = False

        value = indexing.codegen_broadcast_and_reshape(
            value,
            indexing.final_shape,
            indexing.block_shape,
            allow_implicit=False,
            for_store=True,
        )

        # workaround https://github.com/triton-lang/triton/issues/2814
        value = f"{value}.to({triton_store_type(V.graph.get_dtype(name))})"
        if isinstance(indexing, BlockPtrOptions):
            return f"tl.store({block_ptr}, {value}{other})"
        return f"{block_ptr}.store({V.kernel.index_to_str(indexing.offsets)}, {value})"

    def _get_persistent_RBLOCK(self, rnumel):
        rnumel = V.graph.sizevars.simplify(rnumel)
        if isinstance(rnumel, (sympy.Integer, int)):
            val = int(rnumel)
            # Modified by cambricon: set RBlock = rnumel
            # val = next_power_of_2(val)
        else:
            val = 128
            while not V.graph.sizevars.statically_known_leq(rnumel, val):
                assert val <= 16 * 1024, f"Failed to find static RBLOCK for {rnumel}"
                val *= 2
        return val

    # Add a new for loop structure.
    def codegen_range_tree(self):
        super().codegen_range_tree()
        self.body.writelines(
            [
                # pid_offset for the combo kernel and the general kernel are different, specifically generated by codegen_pid_range (for the combo kernel) and codegen_kernel (for the general kernel) respectively.
                "block_start = pid_offset",
            ]
        )

        def count_rindex_in_range_trees():
            return sum(1 for i in self.range_trees if i.prefix[0] == "r")

        if len(self.range_trees) - count_rindex_in_range_trees() == 3:
            self.body.writeline(
                "for block_id in range(block_start, zoffset_num * yoffset_num * xoffset_num, block_step):"
            )
        elif len(self.range_trees) - count_rindex_in_range_trees() == 2:
            self.body.writeline(
                "for block_id in range(block_start, yoffset_num * xoffset_num, block_step):"
            )
        elif len(self.range_trees) - count_rindex_in_range_trees() == 1:
            self.body.writeline(
                "for block_id in range(block_start, xoffset_num, block_step):"
            )
        else:
            log.error(
                """Internal error, the length of range_tree should be
                         less than or equal to 3!"""
            )

        with self.body.indent():
            for idx in range(len(self.range_trees) - count_rindex_in_range_trees()):
                self.iteration_ranges_codegen_body(self.range_trees[idx], self.body)

    # In the original logic, loop splitting was only performed for reduction
    # operators. Based on the original structure, we have added another layer
    # of for loop structure.
    def codegen_body(self):
        with self.body.indent():
            super().codegen_body()

    def codegen_static_numels(self, code):
        for tree in self.range_trees:
            # Match upstream Triton codegen: keep static numels in the kernel
            # signature/call args, and overwrite them with plain constants in
            # the kernel body when the values are compile-time known.
            if not tree.is_reduction or self.inside_reduction:
                simplified_tree_numel = V.graph.sizevars.simplify(tree.numel)
                if isinstance(simplified_tree_numel, (sympy.Integer, int)):
                    code.writeline(f"{tree.prefix}numel = {int(simplified_tree_numel)}")

            if tree.is_reduction and self.persistent_reduction:
                if self.cooperative_reduction:
                    numel = self.kexpr(self.rename_indexing(tree.numel))
                    val = f"triton_helpers.constexpr_next_power_of_2(({numel} + RSPLIT - 1) // RSPLIT)"
                else:
                    val = self._get_persistent_RBLOCK(tree.numel)
                code.writeline(f"{tree.prefix.upper()}BLOCK: tl.constexpr = {val}")

            if tree.prefix == "x" and self.no_x_dim:
                code.writeline("XBLOCK: tl.constexpr = 1")

    # There are two modification points:
    # First, modify the calculation logic of size_hint
    # from next_power_of_2(int(numel_hint)) to int(numel_hint).
    # The next_power_of_2 processing will no longer be performed.
    def codegen_kernel(self, name=None) -> str:
        """
        Convert the TritonKernel from Inductor SIMD IR to triton code, including inductor triton heuristics, imports,
        metadata, and benchmarking infra.
        """
        is_dynamic_shape = any(
            hasattr(x, "free_symbols") and bool(x.free_symbols)
            for x in self.numels.values()
        )

        code = IndentedBuffer()

        size_hints = {}
        for prefix, numel in self.numels.items():
            if prefix_is_reduction(prefix) and not self.inside_reduction:
                continue

            numel_hint = V.graph.sizevars.symbolic_hint(numel)
            if not isinstance(numel_hint, (int, sympy.Integer)):
                # This default heuristic hint was picked carefully: it is
                # large, to ensure that we don't shrink the block size (since
                # if you don't have many elements, it'd be wasteful to pick a
                # large block size).  Since we don't know how many elements we
                # might have, we should be OK with some inefficiency to make
                # sure we handle the large case well.  8192 is the largest
                # block size we support, so we pick that.
                #
                # If we have a better hint for unbacked SymInts (e.g., because
                # a user told us, or we are tracking upper bounds) we could
                # use that here.
                # Modified by Cambricon: use sizevars.size_hint(numel) replace 8192
                size_hint = V.graph.sizevars.size_hint(numel, fallback=8192)
            else:
                # For dynamic-shape kernels, use next_power_of_2 to normalize the size_hints avoid kernel recompilation caused by shape variations.
                if is_dynamic_shape:
                    size_hint = next_power_of_2(int(numel_hint))
                else:
                    size_hint = int(numel_hint)
            size_hints[prefix] = size_hint

        if name is None:
            code.splice(self.gen_common_triton_imports())
            device_type = V.graph.get_current_device_or_throw().type
            if device_type == "cpu":
                code.splice("triton_helpers.set_driver_to_cpu()")
            else:
                code.splice("triton_helpers.set_driver_to_gpu()")

            if config.benchmark_kernel:
                code.splice(self.imports_for_benchmark_kernel())

        argdefs, _, signature, _ = self.args.python_argdefs()
        # maps actual expression to SizeArg if it is in sizevars replacements
        for i, arg in enumerate(signature):
            if isinstance(arg, SizeArg):
                # mypy is unhappy about the sympy.Expr
                # type for the key of the dict below
                symbol = cast(sympy.Symbol, arg.expr)
                if symbol in V.graph.sizevars.inv_precomputed_replacements:
                    signature[i] = SizeArg(
                        arg.name, V.graph.sizevars.inv_precomputed_replacements[symbol]
                    )

        mutated_args: OrderedSet[str] = OrderedSet()
        for mutation in self.mutations:
            if mutation in self.args.input_buffers:
                mutated_args.add(self.args.input_buffers[mutation])
            if (
                mutation in self.args.inplace_buffers
                and mutation not in V.graph.removed_buffers
                and mutation not in self.removed_buffers
            ):
                mutated_args.add(
                    cast(InplacedBuffer, self.args.inplace_buffers[mutation]).inner_name
                )
            if mutation in self.args.output_buffers:
                mutation_arg = self.args.output_buffers[mutation]
                assert not isinstance(mutation_arg, RemovedArg)
                mutated_args.add(mutation_arg)

        # Note: [Workspace Mutation]
        # workspace arguments are mutated, but are not marked as mutations in self.mutations
        # because their buffers are added during codegen, and aren't tracked during
        # lowering/scheduling. So we add them as mutated_args explicitly below.
        #
        # In the logic below, we only mark the workspaces a mutated if they are marked with
        # zero_fill: that's because, if we don't expect the buffer to be pre-filled with
        # zeros, then, although we still mutate the data, we don't care about those
        # mutations because we don't make any assumptions about the contents of the
        # workspace buffer.  Similarly, ZERO_PER_GRAPH requires the kernel to return
        # the buffer back to its original state.
        for argname, arg in zip(argdefs, signature):
            if (
                isinstance(arg, WorkspaceArg)
                and arg.zero_mode == WorkspaceZeroMode.ZERO_ON_CALL
            ):
                mutated_args.add(argname.name)

        # pyrefly: ignore [bad-assignment]
        mutated_args = sorted(mutated_args)

        for tree in self.active_range_trees():
            sizearg = SizeArg(f"{tree.prefix}numel", tree.numel)
            signature.append(sizearg)
            argdefs.append(ArgName(sizearg.name))
            # constexpr version causes issues, see
            # https://github.com/pytorch/torchdynamo/pull/1362
            # triton_meta["constants"][len(argdefs)] = V.graph.sizevars.size_hint(
            #     tree.numel
            # )
            # argdefs.append(f"{tree.prefix}numel: tl.constexpr")

        def add_constexpr_arg(arg_name):
            # new versions (but not old versions) of Triton need constexprs included in the signature
            if triton_version_uses_attrs_dict():
                signature.append(ConstexprArg(arg_name))
            argdefs.append(ArgName(arg_name, is_constexpr=True))

        for tree in self.range_trees:
            if tree.is_reduction and self.persistent_reduction:
                # Rn_BLOCK for persistent_reduction is defined in codegen_static_numels
                continue
            if tree.tensor_dim is None:
                continue

            add_constexpr_arg(f"{tree.prefix.upper()}BLOCK")

        if self.cooperative_reduction:
            add_constexpr_arg("RSPLIT")

        if self.mix_order_reduction:
            add_constexpr_arg("RSPLIT_SIZE")
            add_constexpr_arg("NUM_STAGES")

        triton_meta_signature = signature_to_meta(
            signature, size_dtype=self.index_dtype, argdefs=argdefs
        )
        triton_meta: dict[str, Any] = {
            "signature": triton_meta_signature,
            "device": DeviceProperties.create(V.graph.get_current_device_or_throw()),
            "constants": {},
            "native_matmul": (
                torch._inductor.config.triton.native_matmul
                and ("tl.dot" in str(self.body) or "tl.dot" in str(self.compute))
            ),
            **self.triton_meta_common(),
        }

        if self.cooperative_reduction:
            # Cooperative reductions rely on multi-block synchronization that
            # requires cooperative-grid launches to avoid hanging.
            triton_meta["launch_cooperative_grid"] = True

        # Skip memory optimization for forward of the training loop where we expect
        # every new node will increase the peak memory and our greedy approach would
        # introduce a lot of unnecessary cpu copies.
        optimize_mem = V.graph.is_inference or V.graph.is_backward

        inductor_meta = {
            "grid_type": self._get_grid_type().__name__,
            # Triton will not accept an OrderedSet for autotune_hints
            "autotune_hints": set(self.autotune_hints),  # noqa: set_linter
            "kernel_name": str(Placeholder.DESCRIPTIVE_NAME),
            "mutated_arg_names": mutated_args,
            "optimize_mem": optimize_mem,
            "no_x_dim": self.no_x_dim,
            "atomic_add_found": self.atomic_add_found,
            "num_load": self.num_load,
            "num_store": self.num_store,
            "num_reduction": self.num_reduction,
            **self.inductor_meta_common(),
        }

        if self.mix_order_reduction:
            inductor_meta["RSPLIT_SIZE"] = self.rsplit_size

        if config.deterministic or config.test_configs.force_filter_reduction_configs:
            inductor_meta["has_loadstore_with_contiguous_rdim"] = (
                self.has_load_with_contiguous_rdim
                or self.has_store_with_contiguous_rdim
            )

        # Bail on 3d tiling, which has more complicated coalesce patterns
        looped_red = V.kernel.features.is_reduction() and not self.persistent_reduction
        tiling_scores = self.tiling_scores
        two_d_red = len(self.tiling) == 2
        if looped_red and two_d_red:
            memory_stats = self.features.memory_stats(self.tiling)
            dim_stats = memory_stats.persistent.memory.dim[0]
            mem_ops_per_thread = dim_stats.count_per_thread

            if (
                tiling_scores is not None
                and "x" in tiling_scores
                and "r0_" in tiling_scores
            ):
                # large rblock inhibits xblock size, dont attempt if there is a decent amount of
                # reads coalesced by xblock
                r_coalesce_ratio = tiling_scores["r0_"] / max(tiling_scores["x"], 1)
                contiguous_red = r_coalesce_ratio >= INNER_REDUCTION_RATIO_THRESHOLD
            else:
                contiguous_red = (
                    self.features.get_reduction_hint(tiling_scores)
                    == ReductionHint.INNER
                )

            looped_mem = memory_stats.looped.memory.bytes
            persistent_mem = memory_stats.persistent.memory.bytes
            # check that we save significant memory by doing persistent
            saved_bytes_ratio = V.graph.sizevars.optimization_hint(looped_mem) / max(
                V.graph.sizevars.optimization_hint(persistent_mem),
                1,
            )

            # TODO - rnumel should be reasonably close to power of 2
            if (
                # significant memory bandwidth savings
                saved_bytes_ratio >= 1.3
                and contiguous_red
                # TODO - need more detailed register analysis
                and V.graph.sizevars.statically_known_leq(
                    self.features.reduction_numel, 32768
                )
                # We will already generate a persistent config in this case
                and V.graph.sizevars.statically_known_gt(
                    self.features.reduction_numel, 2048
                )
                and mem_ops_per_thread <= 10
            ):
                inductor_meta["add_persistent_rblock"] = True

        if self.tiling_scores:
            inductor_meta["tiling_scores"] = self.tiling_scores

        if self.tma_min_block_sizes:
            inductor_meta["tma_min_block_sizes"] = self.tma_min_block_sizes

        if self.cooperative_reduction:
            inductor_meta["persistent_reduction"] = self.persistent_reduction

        num_gb = None
        if config.benchmark_kernel or config.profile_bandwidth:
            num_gb = self.estimate_kernel_num_bytes() / 1e9
            if num_gb is not None:
                inductor_meta["kernel_num_gb"] = num_gb
        if config.benchmark_kernel:
            flops = self.estimate_flops()
            if flops is not None:
                inductor_meta["kernel_flop"] = flops

        triton_meta["configs"] = [config_of(signature)]

        # Triton compiler includes equal_to_1 args into constants even
        # when they are not constexpr. otherwise there may be a segfault
        # during launching the Inductor-compiled Triton kernel.
        # https://github.com/pytorch/pytorch/issues/120478#issuecomment-1962822307
        # https://github.com/triton-lang/triton/blob/231efe9ed2d200be0f69a07c298e4342b08efe3d/python/triton/runtime/jit.py#L384
        for arg_num in equal_1_arg_indices(signature):  # type: ignore[index]
            triton_meta["constants"][signature[arg_num].name] = 1  # type: ignore[index,union-attr]

        self.triton_meta = triton_meta
        self.inductor_meta = inductor_meta

        self.codegen_prologue(self.body)
        self.codegen_body()
        self._filter_pdl(self.body)

        for helper in self.helper_functions:
            code.writeline("")
            code.splice(helper)

        if self.fixed_config:
            heuristics_line = f"""
                @triton_heuristics.{self._get_heuristic()}(
                    config={self.fixed_config.config!r},
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r}
                )
                @triton.jit
            """
        elif self.inside_reduction:
            reduction_hint = self.features.get_reduction_hint(self.tiling_scores)
            heuristics_line = f"""
                @triton_heuristics.{self._get_heuristic()}(
                    size_hints={size_hints!r},
                    reduction_hint={reduction_hint},
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r}
                )
                @triton.jit
            """
        else:
            tile_hint = ""
            if len(size_hints) == 2:
                if (
                    len(non_constexpr_signature(signature)) == 4
                ):  # input, output and 2 args
                    tile_hint = "tile_hint=TileHint.SQUARE,"
                else:
                    tile_hint = "tile_hint=TileHint.DEFAULT,"
            heuristics_line = f"""
                @triton_heuristics.{self._get_heuristic()}(
                    size_hints={size_hints!r}, {tile_hint}
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r},
                    min_elem_per_thread={self.min_elem_per_thread}
                )
                @triton.jit
            """
        code.splice(heuristics_line)
        kernel_name = name or str(Placeholder.KERNEL_NAME)
        code.writeline(
            f"def {kernel_name}({', '.join(x.full_name() for x in argdefs)}):"
        )
        with code.indent():
            if config.triton.proton_profiling:
                code.writeline(f'pl.enter_scope("{kernel_name}")')
            # Modify by CAMBRICON:
            # pid_offset was used as block_start, different from triton_combo_kernel
            code.writeline("pid = tl.program_id(0)")
            code.writeline("pid_offset = pid")
            code.writeline("block_step = tl.num_programs(0)")
            self.codegen_static_numels(code)
            for old, new in self.args.aliases():
                code.writeline(f"{old} = {new}")
            code.splice(self.body)
            if config.triton.proton_profiling:
                code.writeline(f'pl.exit_scope("{kernel_name}")')

        if config.benchmark_kernel:
            code.splice(self.codegen_kernel_benchmark(num_gb))

        return code.getvalue()

    # Because we added an external for loop, it needs to be integrated
    # into the indexing_code.
    def codegen_iteration_ranges_entry(self, entry: IterationRangesEntry):
        line = f"{entry.name} = {self.kexpr(self.rename_indexing(entry.expr))}"
        if entry.root.is_loop:
            self.indexing_code.writeline(line)
        else:
            # lift non-reduction stores outside loop
            with self.body.indent():
                # add indent for block_ptr
                self.body.writeline(line)

    # We have added a new for loop structure, so additional calculation logic
    # related to the loop variables needs to be included.
    def iteration_ranges_codegen_header(self, entry, code):
        x = entry.prefix
        if entry.is_loop:
            code.writeline(f"{entry.name} = {x}offset + {x}base")
        elif entry.grid_dim is None:
            if entry.tensor_dim is not None:
                # no need to "{x}offset = "
                code.writeline(
                    f"{entry.name} = {self.iteration_ranges_ranges_code(entry)}"
                )
            else:
                line = self.iteration_ranges_scalar_code(entry, f"{x}offset")
                code.writeline(f"{entry.name} = {line}")
            code.writeline(f"{x}offset = 0")
        else:
            if entry.tensor_dim is not None:
                line = f"{x}offset + {self.iteration_ranges_ranges_code(entry)}"
            else:
                line = self.iteration_ranges_scalar_code(entry, f"{x}offset")
            code.writeline(f"{x}offset_num = tl.cdiv({x}numel, {x.upper()}BLOCK)")

        # Add by CAMBRICON
        if entry.prefix[0] != "r":
            return

        if self._has_constant_mask(entry):
            sizes = self.dense_size_str()
            code.writeline(f"{x}mask = tl.full({sizes}, True, tl.int1)")
        else:
            code.writeline(f"{x}mask = {entry.name} < {x}numel")

    # This is our newly added member function, used to calculate the index of
    # each data block within the loop body.
    def iteration_ranges_codegen_body(self, entry, code):
        x = entry.prefix

        if x not in ["x", "y", "z"]:
            return
        if x == "z":
            code.writeline(
                f"{x}offset = block_id // (yoffset_num * xoffset_num) * {x.upper()}BLOCK"
            )
        elif x == "y":
            code.writeline(
                f"{x}offset = block_id // xoffset_num % yoffset_num * {x.upper()}BLOCK"
            )
        elif x == "x":
            code.writeline(f"{x}offset = block_id % xoffset_num * {x.upper()}BLOCK")

        if entry.tensor_dim is not None:
            # line = f"{x}offset + {self.iteration_ranges_ranges_code(entry, True)}"
            line = f"{x}offset + {self.iteration_ranges_ranges_code(entry)}"
        else:
            line = self.iteration_ranges_scalar_code(entry, f"{x}offset")
        code.writeline(f"{entry.name} = {line}")
        code.writeline(f"{x}mask = {entry.name} < {x}numel")

    # The original implementation of this function is quite long. We only
    # modified one place within it and added comments at the modification
    # points, marking them with "cambricon".
    def indexing(
        self,
        index: sympy.Expr,
        *,
        copy_shape=None,
        dense_indexing=False,
        override_mask=None,
        block_ptr=False,
        tma_compatibility_checker: Optional[TMACompatibilityChecker] = None,
    ):
        """
        Compute the index and mask to pass to tl.load() or tl.store()
        """
        index = self.prepare_indexing(index)
        index_vars = index.free_symbols
        has_rindex = False

        mask_vars = OrderedSet[str]()
        for var in index_vars:
            assert isinstance(var, sympy.Symbol)
            has_rindex = has_rindex or symbol_is_type(
                var, TritonSymbols.reduction_types
            )
            if override_mask:
                pass
            elif symbol_is_type(var, SymT.TMP):
                # indirect indexing
                cse_var = self.cse.varname_map[var.name]
                mask_vars.update(cse_var.mask_vars)
            elif symbol_is_type(
                var,
                (
                    SymT.UNBACKED_INT,
                    SymT.SIZE,
                    SymT.PRECOMPUTED_SIZE,
                    SymT.INDEX,
                    SymT.FLOAT,
                    SymT.UNBACKED_FLOAT,
                ),
            ):
                pass
            else:
                # var is one of xN, yN, r0_N or r1_N
                prefix_matches = [
                    prefix_str[symt]
                    for symt in TritonSymbols.block_types
                    if symbol_is_type(var, symt)
                ]
                assert len(prefix_matches) == 1, f"Ambiguous type: {var.name}"
                mask_vars.add(f"{prefix_matches[0]}mask")

        need_dense = (
            config.triton.dense_indexing
            or dense_indexing
            or self._load_mask is not None
        ) and index != 0

        have_dense = True
        have_loop_vars = False
        dense_mask_vars = OrderedSet[str]()

        for tree in self.active_range_trees():
            if index_vars.intersection(tree.var_list):
                have_loop_vars = True
            else:
                have_dense = False
            dense_mask_vars.add(f"{tree.prefix}mask")

        if (
            (
                (block_ptr and self.allow_block_ptr and config.triton.use_block_ptr)
                or (
                    tma_compatibility_checker
                    and tma_compatibility_checker.can_use_tma()
                )
            )
            and not override_mask
            and not self._load_mask
            and len(mask_vars - dense_mask_vars) == 0
            and not self.is_indirect_indexing(index)
            and have_loop_vars
            # workaround https://github.com/triton-lang/triton/issues/2821
            and self.index_dtype == "tl.int32"
        ):

            def match_affine_block(
                index: sympy.Expr, range_tree: IterationRangesRoot
            ) -> Optional[BlockParameters]:
                """
                Matches expressions of the form:
                    idx = s * xindex

                This implies stride (s,), and shape (XBLOCK,).
                """
                stride = BlockPatternMatcher.match_affine_block_expr(
                    index, range_tree.symbol()
                )
                if stride is None:
                    return None

                return BlockParameters(
                    shape=[range_tree.numel],
                    block_shape=[TritonSymbols.get_block_size(range_tree)],
                    strides=[stride],
                    offsets=[TritonSymbols.get_block_offset(range_tree)],
                )

            def match_mod_div_block(
                index: sympy.Expr, range_tree: IterationRangesRoot
            ) -> Optional[BlockParameters]:
                """
                Matches higher-dimensional blocks coming from FloorDiv and ModularIndexing.

                Example expression to match:
                   sN * ((rindex//(d1 * ... * d(N-1))))
                       + s1 * ModularIndexing(rindex, 1, d1)
                       + ...
                       + s(N-1) * ModularIndexing(rindex, d1 * ... * d(N-2), d(N-1))

                This iterates over a block of shape (dN, ..., d1) and stride
                (sN, ..., s1). (d1,...,d(N-1)) and (s1,...,sN) are
                wildcards that we match.

                Note that dN does not appear in the expression, but we solve for it
                using range tree numels and the other dims.
                """

                index_var = range_tree.symbol()

                # Bound the possible number of dims. We use the following heuristics:
                # - At least one dim for each range tree node.
                # - At least one dim for every FloorDiv or ModularIndexing op.
                # - At least 2 dims to pattern match.
                denom, modulo = sympy.symbols(
                    "denom modulo",
                    cls=functools.partial(sympy.Wild, exclude=[index_var]),
                )

                # Modified by Cambricon: workaround for [x0, x1, r2, x3], we only need [x0, x1, r2]
                filtered_range_tree_nodes = self.range_tree_nodes
                if self.inside_reduction:
                    keys = list(self.range_tree_nodes.keys())
                    stop_index = next(
                        (i for i, k in enumerate(keys) if str(k).startswith("r")),
                        len(keys),
                    )
                    filtered_range_tree_nodes = {
                        k: self.range_tree_nodes[k] for k in keys[: stop_index + 1]
                    }

                num_dims = max(
                    2,
                    len(filtered_range_tree_nodes),
                    (
                        index.count(FloorDiv(index_var, denom))
                        + index.count(ModularIndexing(index_var, denom, modulo))
                    ),
                )

                match_result = BlockPatternMatcher.match_mod_div_block_expr(
                    index, index_var, range_tree.numel, num_dims
                )
                if match_result is None:
                    return None

                (
                    dims,
                    strides,
                    block_index_exprs,
                ) = match_result
                slice_numels = BlockPatternMatcher.get_slice_numels(dims)

                # Check for applicable iteration range sizes.
                # When mapping a 1D block into an ND one, we need to know that
                # the number of elements is not changed. This means the slice numels of
                # the ND iteration range must evenly divide the length of the 1D block.
                # There are two cases where we can guarantee this:
                #  1. Numels are powers of 2. If numel == 2 ** n, and we know XBLOCK == 2 ** m,
                #     with n and m integers, then either numel is a multiple of XBLOCK, or numel
                #     is less than XBLOCK. (If numel is less than XBLOCK, we round up to 1 below.)
                #  2. Numels are multiples of the maximum possible block size.
                sizevars = V.graph.sizevars
                max_block = self.max_block(range_tree.prefix)
                if any(
                    not sizevars.statically_known_multiple_of(numel, max_block)
                    and not sizevars.statically_known_power_of_2(numel)
                    for numel in slice_numels
                ):
                    return None

                # Compute the ND block shape from the linear block size.
                # Use CielDiv to round leading dimensions up to 1.
                # Non-leading dimensions are clamped to the size of the iteration range,
                # while the leading dimension can exceed this to accommodate a larger
                # block size.
                linear_block_size = TritonSymbols.get_block_size(range_tree)
                block_shape: list[sympy.Expr] = [
                    CeilDiv(linear_block_size, slice_numels[0])
                ] + [
                    sympy.Min(CeilDiv(linear_block_size, numel), dim)
                    for numel, dim in zip(slice_numels[1:], dims[1:])
                ]

                # Compute block offsets from {xyzr}offset and the matched expressions.
                block_offsets: list[sympy.Expr] = [
                    sympy_subs(
                        expr, {index_var: TritonSymbols.get_block_offset(range_tree)}
                    )
                    for expr in block_index_exprs
                ]

                # Modified by cambricon: add two checks.
                # If any of block_shape and bock_offset is not symbol, match failed.
                for _block_shape in block_shape:
                    if not _block_shape.is_symbol:
                        return None

                for block_offset in block_offsets:
                    if not block_offset.is_symbol:
                        return None

                return BlockParameters(
                    shape=dims,
                    block_shape=block_shape,
                    strides=strides,
                    offsets=block_offsets,
                )

            def match_block_subexpr(
                expr: sympy.Expr, range_tree: IterationRangesRoot
            ) -> Optional[BlockParameters]:
                """
                Match a block indexing subexpression involving a single range tree.
                """
                for match_func in (
                    match_affine_block,
                    match_mod_div_block,
                ):
                    match = match_func(expr, range_tree)
                    if match is not None:
                        return match

                return None

            def match_block_expr() -> Optional[BlockDescriptorOptions]:
                # Add by CAMBRICON
                from torch_mlu._inductor import config as inductor_config

                index_relative_to_xyr_index = sympy_subs(
                    index, {v: t.expr for v, t in self.range_tree_nodes.items()}
                )
                range_trees = self.active_range_trees()

                # Partition the index into subexpressions pertaining to each range tree.
                # For example xindex * 5 + rindex * 3 is partitioned to
                # (xindex * 5, rindex * 3).
                index_subexprs = [
                    BlockPatternMatcher.get_subexpr_involving_symbol(
                        index_relative_to_xyr_index, tree.symbol()
                    )
                    for tree in range_trees
                ]

                # Match each range tree separately.
                range_symbols = OrderedSet(tree.symbol() for tree in range_trees)
                block_params = BlockParameters()
                for tree, subexpr in zip(range_trees, index_subexprs):
                    # Reject mixed terms, e.g. xindex * r0_index.
                    # NB: the zero expression is allowed, for broadcasting.
                    if len(range_symbols.intersection(subexpr.free_symbols)) > 1:
                        return None

                    # Match the subexpression for this range tree.
                    params = match_block_subexpr(subexpr, tree)
                    if params is None:
                        return None
                    block_params += params

                # Collect leftover terms as a constant offset.
                offset = index_relative_to_xyr_index - sum(index_subexprs)

                # Form the block pointer or TMA descriptor.
                self.filter_masks(mask_vars)

                options_class = (
                    BlockPtrOptions
                    if config.triton.use_block_ptr
                    else TensorDescriptorOptions
                )
                nonlocal tma_compatibility_checker
                stride_sorter_cls: type[BlockParameters.StrideSorter]
                if config.triton.use_block_ptr:
                    can_lift = False
                    stride_sorter_cls = BlockParameters.IdentityStrideSorter
                else:
                    tma_compatibility_checker = cast(
                        TMACompatibilityChecker, tma_compatibility_checker
                    )
                    can_lift = tma_compatibility_checker.can_lift()

                    if (
                        self.transpose_discontiguous_tensor_descriptors_override
                        is not None
                    ):
                        transpose_contiguous = (
                            self.transpose_discontiguous_tensor_descriptors_override
                        )
                    else:
                        transpose_contiguous = (
                            config.triton.transpose_discontiguous_tensor_descriptor
                        )

                    # For templates:
                    # Only try transpose if we know the output shape
                    # in case we need to transpose the data.
                    if hasattr(self, "template_out_shape"):
                        transpose_contiguous &= copy_shape is not None

                    stride_sorter_cls = (
                        BlockParameters.TensorDecriptorStrideSorter
                        if transpose_contiguous
                        else BlockParameters.IdentityStrideSorter
                    )

                options = options_class.create(
                    params=block_params,
                    constant_offset=offset,
                    range_trees=range_trees,
                    mask_vars=mask_vars,
                    get_max_block=self.max_block,
                    can_lift=can_lift,
                    stride_sorter_cls=stride_sorter_cls,
                )
                if options_class == TensorDescriptorOptions:
                    tma_compatibility_checker = cast(
                        TMACompatibilityChecker, tma_compatibility_checker
                    )
                    if not tma_compatibility_checker.are_block_parameters_compatible(
                        options.params
                    ):
                        return None

                return options

            # Return a block pointer, if indexing matches the pattern.
            options = match_block_expr()
            if options is not None:
                return options
        expand_str = None
        expand_shape: BlockShapeType = None
        index_str = self.index_to_str(index)
        if isinstance(index, sympy.Integer):
            expand_str = f"{copy_shape}.shape" if copy_shape else self.dense_size_str()
            expand_shape = None if copy_shape else tuple(self.dense_size_list())
            index_str = f"tl.full({expand_str}, {index_str}, tl.int32)"
            if self.fixed_config and not self._has_constant_xmask():
                mask_vars = OrderedSet(["xmask"])
            else:
                mask_vars = OrderedSet()
            if self._load_mask:
                mask_vars.add(self._load_mask)
            return IndexingOptions(
                index_str,
                mask_vars,
                expand_str,
                has_rindex,
                index,
                expand_shape=expand_shape,
            )
        if need_dense and not have_dense:
            expand_str = f"{copy_shape}.shape" if copy_shape else self.dense_size_str()
            expand_shape = None if copy_shape else tuple(self.dense_size_list())
            index_str = f"tl.broadcast_to({index_str}, {expand_str})"
            mask_vars = dense_mask_vars
        elif not have_loop_vars and copy_shape:
            index_str = f"tl.broadcast_to({index_str}, {copy_shape}.shape)"
            mask_vars = dense_mask_vars

        if expand_shape is None:
            if need_dense or have_dense:
                expand_shape = None if copy_shape else tuple(self.dense_size_list())
            else:
                expand_shape = ()

        if override_mask:
            mask_vars = OrderedSet([override_mask])

        if self._load_mask:
            mask_vars.add(self._load_mask)

        self.filter_masks(mask_vars)

        return IndexingOptions(
            index_str,
            mask_vars,
            expand_str,
            has_rindex,
            index,
            expand_shape=expand_shape,
        )

    def welford_reduce(
        self, result_var, reduction_type, value, where_cond, acc_type, dtype
    ):
        """Helper to codegen a welford reduction"""
        dim = self.triton_tensor_ndim() - self.num_reduction_dims

        accumulator = TritonCSEVariable(
            f"{result_var}_mean",
            shape=tuple(self.dense_size_list()),
            dtype=acc_type,
            bounds=ValueRanges.unknown(),
        )
        accumulator_m2 = TritonCSEVariable(
            f"{result_var}_m2",
            shape=tuple(self.dense_size_list()),
            dtype=acc_type,
            bounds=ValueRanges.unknown(),
        )
        accumulator_weight = TritonCSEVariable(
            f"{result_var}_weight",
            shape=tuple(self.dense_size_list()),
            dtype=acc_type,
            bounds=ValueRanges.unknown(),
        )
        # Modify by CAMBRICON: add an indent context.
        with self.body.indent():
            self.body.writeline(
                f"{accumulator} = tl.zeros({self.dense_size_str()}, {acc_type})"
            )
            self.body.writeline(
                f"{accumulator_m2} = tl.zeros({self.dense_size_str()}, {acc_type})"
            )
            self.body.writeline(
                f"{accumulator_weight} = tl.zeros({self.dense_size_str()}, {acc_type})"
            )
        if reduction_type == "welford_combine":
            mean, m2, weight = value
            self.compute.splice(
                f"""\
                {accumulator}_next, {accumulator_m2}_next, {accumulator_weight}_next = triton_helpers.welford_combine(
                    {accumulator}, {accumulator_m2}, {accumulator_weight},
                    {mean}, {m2}, {weight}
                )
                """
            )
        else:
            assert reduction_type == "welford_reduce"
            self.compute.splice(
                f"""\
                {accumulator}_next, {accumulator_m2}_next, {accumulator_weight}_next = triton_helpers.welford_reduce(
                    {value}, {accumulator}, {accumulator_m2}, {accumulator_weight}, roffset == 0
                )
                """
            )
        self.compute.splice(
            f"""\
            {accumulator} = {where_cond(f"{accumulator}_next", accumulator)}
            {accumulator_m2} = {where_cond(f"{accumulator_m2}_next", accumulator_m2)}
            {accumulator_weight} = {where_cond(f"{accumulator_weight}_next", accumulator_weight)}
            """
        )
        result_mean = result_var
        return self.welford_reduce_final_reduction(
            self.post_loop_combine,
            result_mean,
            None,
            None,
            accumulator,
            accumulator_m2,
            accumulator_weight,
            dim,
            dtype,
        )

    # In the original logic, the body code generated by reduction was within
    # one layer of for loop, while the code related to indexing and CSE was
    # outside the loop. After our modification, the indexing and CSE related
    # code generated by reduction should also be inside the outermost loop.
    # Therefore, indentation needs to be added to this code. The main
    # modification in this function is adding indentation.
    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Union[CSEVariable, tuple[CSEVariable, ...]],
    ) -> Union[CSEVariable, tuple[CSEVariable, ...]]:
        """
        codegen reduction of value to Triton according the reduction_type
        """

        def maybe_upcast(value: CSEVariable) -> CSEVariable:
            # Math reductions in FP16/BF16 are less accurate because the Triton compiler does not
            # automatically promote to FP32 for accumulation. Additionally, max/min reductions
            # do not support FP16/BF16. We manually promote to FP32 here.
            return (
                ops.to_dtype(value, torch.float32)
                if value.dtype
                in [
                    torch.float16,
                    torch.bfloat16,
                ]
                else value
            )

        original_dtypes = [val.dtype for val in pytree.tree_leaves(value)]
        value = pytree.tree_map(maybe_upcast, value)
        if any(x in [torch.float16, torch.bfloat16] for x in original_dtypes):
            # Only promote FB16/BF16; do not promote other integer/boolean dtypes
            src_dtype = torch.promote_types(src_dtype, torch.float32)
            dtype = torch.promote_types(dtype, torch.float32)

        assert self.inside_reduction
        masks = OrderedSet(f"{tree.prefix}mask" for tree in self.range_trees)
        self.filter_masks(masks)
        masks = sorted(masks)
        if self._load_mask:
            masks.append(self._load_mask)
        reduction_range_prefix = self.range_trees[-1].prefix

        # When we do native matmtul codegen,
        # we don't want to keep the R0_BLOCK/R1_BLOCK in the accumulator.
        # so instead of naively calling dense_size_str(), we filter out
        # reduction block from accumulator and only keep (Y,X).
        # In bmm (Z,Y,R)x(Z,R,X) case, we also remove z dimension from accumulator
        # because 3d (Z,Y,X) tl.dot is somehow slower than 2d tl.dot.
        # Instead, we force ZBLOCK to be always 1 during autotune.
        dense_size_str: str
        if self.is_native_matmul:
            dense_sizes = self.dense_size_list()
            assert len(dense_sizes) >= 3
            xy_sizes_only = [size for size in dense_sizes if "X" in size or "Y" in size]
            dense_size_str = f"[{', '.join(xy_sizes_only)}]"
            value_shape = tuple(xy_sizes_only)
        else:
            dense_size_str = self.dense_size_str()
            value_shape = tuple(self.dense_size_list())

        # Say we have
        #     tmp0 = ops.constant(1, torch.int64)
        #     tmp1 = ops.reduction(torch.int64, torch.int64, "sum", tmp0)
        # tmp0 in the triton code is either a scalar, or single-element tensor
        # so if we emit tl.sum directly, it will only give 1 instead of RBLOCK * 1
        # To avoid this, we broadcast to the expected shape first.
        value = self._map_tuple_or_scalar(
            lambda v: self.cse.generate(
                self.compute,
                f"tl.broadcast_to({v}, {dense_size_str})",
                dtype=v.dtype,
                shape=value_shape,
            ),
            value,
        )

        logical_index = None
        if reduction_type in ("argmin", "argmax"):
            if isinstance(value, tuple):
                value, logical_index = value

        dim = self.triton_tensor_ndim() - self.num_reduction_dims
        root_op: str

        def final_reduction(
            buffer,
            value: CSEVariable,
            result_type: Optional[torch.dtype],
        ) -> tuple[str, Optional[torch.dtype], BlockShapeType]:
            """
            Helper to generate a reduction call, e.g. tl.sum.
            """
            triton_reduction_fn = get_triton_reduction_function(reduction_type)

            value = self.reduction_collapse_dims(buffer, value, dtype)
            if reduction_type == "dot":
                # Native matmul is a special case because accumulator shape is fixed to (Y,X)
                is_bmm = len(self.dense_size_list()) == 4
                assert value.shape is not None
                if is_bmm:
                    result = f"{value}[None,:,:,None]"  # (Y,X) to (Z=1,Y,X,R=1)
                    shape = [1, *value.shape, 1]
                else:
                    result = f"{value}[:,:,None]"  # (Y,X) to (Y,X,R=1)
                    shape = [*value.shape, 1]
            else:
                result, shape = self.reduction_resize_and_shape(  # type: ignore[assignment]
                    f"{triton_reduction_fn}({value}, {dim})", value.shape
                )

            if result_type is not None:
                result = f"{result}.to({self.dtype_to_str(result_type)})"
            else:
                result_type = value.dtype

            return result, result_type, shape

        def final_reduction_define(
            buffer,
            result_var: CSEVariable,
            value: CSEVariable,
            result_type: Optional[torch.dtype],
        ) -> None:
            """
            Generate a reduction and assign it to an existing variable.
            """
            value, _, _ = final_reduction(buffer, value, result_type)
            buffer.splice(f"{result_var} = {value}")

        def final_argreduce(buffer, result_var, value, index):
            value = self.reduction_collapse_dims(buffer, value, dtype)
            index = self.reduction_collapse_dims(buffer, index, dtype)
            buffer.splice(
                f"""\
                {result_var}_val, {result_var}_idx = triton_helpers.{root_op}_with_index({value}, {index}, {dim})
                {result_var} = {self.reduction_resize(f"{result_var}_idx")}
                """
            )

        cache_key = (src_dtype, reduction_type, value)
        if cache_key in self.cse.reduction_cache:
            return self.cse.reduction_cache[cache_key]

        acc_type = triton_acc_type(src_dtype)
        torch_acc_type = upcast_acc_dtype(src_dtype)
        result_shape = list(self.dense_size_list())
        result_shape[dim] = "1"
        result_var: Any = self.cse.newvar(
            dtype=torch_acc_type, shape=tuple(result_shape)
        )
        result_var.mask_vars = OrderedSet(
            var for var in masks if not prefix_is_reduction(var[0])
        )
        cond = " & ".join(masks)

        def where_cond(tval, fval):
            if not cond:
                return tval
            return TritonKernelOverrides.where(cond, tval, fval)

        if self.persistent_reduction:
            default = ir.Reduction.default_value(reduction_type, src_dtype)

            def update_constant_dtype(constant, src_dtype, dst_dtype):
                "update reduction constant mask value to match dst_dtype"

                # int is the only mask which may not fit within lower bitwidth,
                # because float uses inf/-inf
                if src_dtype.is_floating_point or src_dtype == torch.bool:
                    return constant

                if src_dtype == dst_dtype or constant == 0:
                    return constant

                if constant == torch.iinfo(src_dtype).max:
                    return torch.iinfo(dst_dtype).max
                elif constant == torch.iinfo(src_dtype).min:
                    return torch.iinfo(dst_dtype).min
                else:
                    return constant

            def _mask_value(value, default):
                default = update_constant_dtype(default, src_dtype, value.dtype)
                default_str = self._map_tuple_or_scalar(constant_repr, default)

                return self.cse.generate(
                    self.compute,
                    where_cond(value, default_str),
                    dtype=value.dtype,
                    shape=value.shape,
                )

            masked_value: Union[CSEVariable, Sequence[CSEVariable]]
            if reduction_type == "online_softmax_reduce":
                # Don't generate mask value for online_softmax since we
                # will fallback below
                pass
            elif isinstance(value, tuple):
                masked_value = [_mask_value(v, d) for v, d in zip(value, default)]  # type: ignore[arg-type]
            elif reduction_type == "dot":
                # Here, we don't perform the masking.
                # Masking w/ where condition in native matmul is handled in ops.dot codegen.
                # Since tl.dot performs reduction within the triton block,
                # masking should happen before the tl.dot is called.
                masked_value = self.cse.generate(self.compute, value, dtype=value.dtype)
            else:
                masked_value = _mask_value(value, default)

            if reduction_type in ("argmax", "argmin"):
                assert isinstance(masked_value, CSEVariable)
                accumulator_dtype = V.kernel.get_index_dtype_as_torch_dtype()
                if logical_index:
                    accumulator_index = f"({str(logical_index)}).to({self.dtype_to_str(accumulator_dtype)})"
                else:
                    accumulator_index = str(
                        self.cse.generate(
                            self.compute,
                            f"tl.broadcast_to({reduction_range_prefix}index, {masked_value}.shape)",
                            dtype=accumulator_dtype,
                            shape=masked_value.shape,
                        )
                    )
                root_op = {"argmax": "max", "argmin": "min"}[reduction_type]
                final_argreduce(
                    self.compute, result_var, masked_value, accumulator_index
                )
                result_var.dtype = accumulator_dtype
            elif reduction_type == "welford_reduce":
                if self.cooperative_reduction:
                    # cooperative reductions require full welford for correctness
                    result_var = self.welford_reduce(
                        result_var, reduction_type, value, where_cond, acc_type, dtype
                    )
                else:
                    # For persistent reductions, don't bother with
                    # welford's algorithm since it uses more registers, and
                    # taking two reductions doesn't increase memory usage.
                    result_var = self.welford_reduce_fallback(dtype, value)
            elif reduction_type == "welford_combine":
                assert isinstance(masked_value, Sequence)
                (mean, m2, weight) = masked_value
                result_var = tuple(
                    self.cse.generate(self.compute, value, dtype=dtype, shape=shape)
                    for value, shape in self._welford(
                        self.compute, mean, m2, weight, dim, dtype
                    )
                )
            elif reduction_type == "online_softmax_reduce":
                # All data is loaded to register anyway, no need to do
                # online softmax
                result_var = self.prepare_softmax_twopass_fallback(dtype, value)
            else:
                assert isinstance(masked_value, CSEVariable)
                _result, _dtype, _shape = final_reduction(
                    self.compute, masked_value, masked_value.dtype
                )
                result_var = self.cse.generate(
                    self.compute, _result, dtype=_dtype, shape=_shape
                )
        else:
            accumulator = self.cse.namedvar(
                f"_{result_var}",
                dtype=torch_acc_type,
                shape=tuple(self.dense_size_list()),
            )
            default = ir.Reduction.default_accumulator(reduction_type, src_dtype)
            default = self._map_tuple_or_scalar(constant_repr, default)
            if not isinstance(default, tuple):
                # Modify by CAMBRICON: add an indent context.
                with self.body.indent():
                    if reduction_type == "dot":
                        dense_sizes = self.dense_size_list()
                        assert len(dense_sizes) >= 3
                        xy_sizes_only = [
                            size for size in dense_sizes if "X" in size or "Y" in size
                        ]
                        accumulator.shape = tuple(xy_sizes_only)
                        dense_size_str = f"[{', '.join(xy_sizes_only)}]"
                        self.body.writeline(
                            f"{accumulator} = tl.full({dense_size_str}, {default}, {acc_type})"
                        )
                    else:
                        self.body.writeline(
                            f"{accumulator} = tl.full({self.dense_size_str()}, {default}, {acc_type})"
                        )

            if reduction_type in {"argmax", "argmin"}:
                accumulator_index = f"_{result_var}_index"
                index_dtype = self.features.select_index_dtype()
                # Modify by CAMBRICON: add an indent context.
                with self.body.indent():
                    self.body.writeline(
                        f"{accumulator_index} = tl.full({self.dense_size_str()}, "
                        f"{torch.iinfo(index_dtype).max}, {self.dtype_to_str(index_dtype)})"
                    )
                root_op = {"argmax": "max", "argmin": "min"}[reduction_type]
                # Use logical_index if it was unpacked, otherwise fall back to physical index
                index_var = (
                    f"({str(logical_index)}).to({self.dtype_to_str(index_dtype)})"
                    if logical_index is not None
                    else f"{reduction_range_prefix}index"
                )
                self.compute.splice(
                    f"""\
                {accumulator}_next, {accumulator_index}_next = triton_helpers.{root_op}imum_with_index(
                    {accumulator}, {accumulator_index}, {value}, {index_var}
                )
                {accumulator} = {where_cond(f"{accumulator}_next", accumulator)}
                {accumulator_index} = {where_cond(f"{accumulator_index}_next", accumulator_index)}
                """
                )
                final_argreduce(
                    self.post_loop_combine, result_var, accumulator, accumulator_index
                )
            elif is_welford_reduction(reduction_type):
                result_var = self.welford_reduce(
                    result_var, reduction_type, value, where_cond, acc_type, dtype
                )
            elif reduction_type == "online_softmax_reduce":
                accumulator_max = f"_{result_var}_max"
                accumulator_sum = f"_{result_var}_sum"

                # setup accumulator
                self.body.writeline(
                    f"{accumulator_max} = tl.full({self.dense_size_str()}, float('-inf'), {acc_type})"
                )
                self.body.writeline(
                    f"{accumulator_sum} = tl.zeros({self.dense_size_str()}, {acc_type})"
                )

                # combine
                # Note, we pass config.use_fast_math to the JITFunction
                # since a triton kernel can not access a config.
                self.compute.splice(
                    f"""
                    {accumulator_max}_next, {accumulator_sum}_next = triton_helpers.online_softmax_combine(
                        {accumulator_max}, {accumulator_sum}, {value}, {config.use_fast_math}
                    )
                    """
                )

                # mask
                self.compute.splice(
                    f"""
                    {accumulator_max} = {where_cond(f"{accumulator_max}_next", accumulator_max)}
                    {accumulator_sum} = {where_cond(f"{accumulator_sum}_next", accumulator_sum)}
                    """
                )

                # reduce. Similar to the final reduction for coopereative
                # reduction
                result_max = result_var
                result_sum = self.cse.newvar(dtype=dtype, shape=result_max.shape)

                result_var = self.online_softmax_reduce_final_reduction(
                    self.post_loop_combine,
                    result_max,
                    result_sum,
                    accumulator_max,
                    accumulator_sum,
                    dim,
                    dtype,
                )
            else:
                combine_fn = ir.get_reduction_combine_fn(reduction_type, src_dtype)
                updated = combine_fn(accumulator, value)
                # Modify by CAMBRICON: add an indent context.
                with self.body.indent():
                    if reduction_type == "dot":
                        self.compute.writeline(f"{accumulator} = {updated}")
                    else:
                        self.compute.writeline(
                            f"{accumulator} = {where_cond(updated, accumulator)}"
                        )

                if src_dtype == torch.bool:
                    # This is only really used for aten.any. It changes the
                    # final reduction of a non-persistent reduction from
                    #     tmp5 = triton_helpers.max(_tmp5, 1)[:, None]
                    # to
                    #     tmp5 = triton_helpers.max(_tmp5.to(tl.int8), 1)[:, None].to(tl.int1)
                    # which is needed because tl.reduce doesn't support tl.int1
                    accumulator = self.cse.generate(
                        self.post_loop_combine,
                        f"{accumulator}.to(tl.int8)",
                        dtype=torch.int8,
                        shape=accumulator.shape,
                    )

                final_reduction_define(
                    self.post_loop_combine, result_var, accumulator, None
                )

        if self.cooperative_reduction:
            default = ir.Reduction.default_accumulator(reduction_type, src_dtype)
            exit_stack = contextlib.ExitStack()
            for buf in (self.post_loop_combine, self.post_loop_store):
                # only do cooperative reduction combines if we have more than one thread block
                buf.writeline("if HAS_RSPLIT:")
                exit_stack.enter_context(buf.indent())

            if reduction_type in ("argmax", "argmin"):
                self.post_loop_combine.writeline(
                    f"{result_var}_bval = {self.reduction_resize(f'{result_var}_val')}"
                )
                peer_val = self.codegen_cooperative_reduction_peer_combine(
                    f"{result_var}_bval", src_dtype, default
                )
                index_dtype = self.features.select_index_dtype()
                peer_idx = self.codegen_cooperative_reduction_peer_combine(
                    result_var, index_dtype, torch.iinfo(index_dtype).max
                )
                final_argreduce(self.post_loop_store, result_var, peer_val, peer_idx)
            elif is_welford_reduction(reduction_type):
                assert reduction_type == "welford_reduce"
                result_mean, result_m2, result_weight = result_var
                peer_mean = self.codegen_cooperative_reduction_peer_combine(
                    result_mean,
                    upcast_acc_dtype(src_dtype),
                    default[0],  # type: ignore[index]
                )
                peer_m2 = self.codegen_cooperative_reduction_peer_combine(
                    result_m2,
                    upcast_acc_dtype(src_dtype),
                    default[1],  # type: ignore[index]
                )
                peer_weight = self.codegen_cooperative_reduction_peer_combine(
                    result_weight,
                    upcast_acc_dtype(src_dtype),
                    default[2],  # type: ignore[index]
                )
                self.welford_reduce_final_reduction(
                    self.post_loop_store,
                    result_mean,
                    result_m2,
                    result_weight,
                    peer_mean,
                    peer_m2,
                    peer_weight,
                    dim,
                    dtype,
                )
            elif reduction_type == "online_softmax_reduce":
                result_max, result_sum = result_var
                assert isinstance(default, Sequence)
                peer_max = self.codegen_cooperative_reduction_peer_combine(
                    result_max, upcast_acc_dtype(src_dtype), default[0]
                )
                peer_sum = self.codegen_cooperative_reduction_peer_combine(
                    result_sum, upcast_acc_dtype(src_dtype), default[1]
                )
                self.online_softmax_reduce_final_reduction(
                    self.post_loop_store,
                    result_max,
                    result_sum,
                    peer_max,
                    peer_sum,
                    dim,
                    dtype,
                )
            else:
                peers = self.codegen_cooperative_reduction_peer_combine(
                    result_var, upcast_acc_dtype(src_dtype), default
                )
                final_reduction_define(self.post_loop_store, result_var, peers, None)
            exit_stack.close()

        self.cse.reduction_cache[cache_key] = result_var

        if isinstance(result_var, tuple):
            assert all(isinstance(x, TritonCSEVariable) for x in result_var)
            self.outside_loop_vars.update(result_var)

            # Match output dtype with input dtype
            if reduction_type in ("welford_reduce", "online_softmax_reduce"):
                assert len(original_dtypes) == 1
                original_dtypes = len(result_var) * original_dtypes

            assert len(result_var) == len(original_dtypes)
            for var, orig_dtype in zip(result_var, original_dtypes):
                assert orig_dtype is not None
                if var.dtype != orig_dtype:
                    self.post_loop_combine.writeline(
                        f"{var} = {var}.to({triton_compute_type(orig_dtype)})"
                    )
        else:
            assert isinstance(result_var, TritonCSEVariable)
            self.outside_loop_vars.add(result_var)

            # Match output dtype with input dtype
            if result_var.dtype != original_dtypes[0]:
                assert original_dtypes[0] is not None
                self.post_loop_combine.writeline(
                    f"{result_var} = {result_var}.to({triton_compute_type(original_dtypes[0])})"
                )

        return result_var

    def scan(
        self,
        dtypes: tuple[torch.dtype, ...],
        combine_fn: Callable[
            [tuple[CSEVariable, ...], tuple[CSEVariable, ...]], tuple[CSEVariable, ...]
        ],
        values: tuple[CSEVariable, ...],
    ) -> tuple[CSEVariable, ...]:
        """
        Perform an associative scan on 'values'.
        """
        assert self.inside_reduction
        assert not self.cooperative_reduction, "TODO"
        masks = OrderedSet(f"{tree.prefix}mask" for tree in self.range_trees)
        self.filter_masks(masks)
        masks = sorted(masks)
        assert not self._load_mask, "ops.scan not supported inside ops.masked"

        broadcasted_values = []
        accumulators = []

        dtypes = tuple(upcast_compute_type(dtype) for dtype in dtypes)
        cse_compute = functools.partial(self.cse.generate, self.compute)
        combine_helper_fn = self._lift_helper(combine_fn, values, dtypes)
        dim = self.triton_tensor_ndim() - self.num_reduction_dims

        for value, dtype in zip(values, dtypes):
            value_dtype = self.cse.generate(
                self.compute,
                f"{value}.to({triton_compute_type(dtype)})",
                dtype=dtype,
                shape=value.shape,
            )
            value = self.cse.generate(
                self.compute,
                f"tl.broadcast_to({value_dtype}, {self.dense_size_str()})",
                dtype=dtype,
                shape=tuple(self.dense_size_list()),
            )
            broadcasted_values.append(value)

            acc_type = triton_acc_type(dtype)

            if not self.persistent_reduction:
                reduced_size = self.dense_size_list()
                reduced_size[-1] = "1"
                accumulator = self.cse.newvar(dtype=dtype, shape=reduced_size)
                reduced_size_str = f"[{', '.join(reduced_size)}]"

                default = "float('nan')" if dtype.is_floating_point else "-1"
                # Modify by CAMBRICON
                with self.body.indent():
                    self.body.writeline(
                        f"{accumulator} = tl.full({reduced_size_str}, {default}, {acc_type})"
                    )

                accumulators.append(accumulator)

        def csv(values):
            return " ".join(f"{value}," for value in values)

        def cse_multiple(line, values, masks, dtypes):
            n = len(values)
            cache_keys = [f"{line}, {i}, {masks}" for i in range(n)]
            if all(self.cse.contains(cache_key) for cache_key in cache_keys):
                return [self.cse.get(cache_key) for cache_key in cache_keys]
            result_vars = [
                self.cse.newvar(dtype=dtype, shape=value.shape)
                for (dtype, value) in zip(dtypes, values)
            ]
            self.compute.writeline(
                f"{csv(result_vars)} = {line}",
            )
            for result_var, cache_key in zip(result_vars, cache_keys):
                if masks:
                    result_var.mask_vars = masks  # type: ignore[attr-defined]
                self.cse.put(cache_key, result_var)
            return tuple(result_vars)

        partial_scan_vars = cse_multiple(
            f"tl.associative_scan(({csv(broadcasted_values)}), {dim}, {combine_helper_fn})",
            broadcasted_values,
            masks,
            dtypes,
        )

        if not self.persistent_reduction:
            # tl.reduce doesn't work for non-commutative operators, so instead
            # of repeating the scan op as a reduction, we use sum to select the
            # last scan value
            def _partial_scan_shape(var):
                if var.shape is None:
                    return None
                else:
                    shape = list(var.shape)
                    shape[-1] = "1"
                    return shape

            partial_reduce_vars = [
                cse_compute(
                    f"triton_helpers.select_one(({partial_scan_var}), rbase == (RBLOCK - 1), dim=-1, keep_dims=True)",
                    dtype=upcast_compute_type(partial_scan_var.dtype),
                    shape=_partial_scan_shape(partial_scan_var),
                )
                for partial_scan_var in partial_scan_vars
            ]
            accs_next = combine_fn(tuple(accumulators), tuple(partial_reduce_vars))
            full_scan_vars = combine_fn(tuple(accumulators), partial_scan_vars)
            result_vars = [
                cse_compute(
                    f"tl.where(roffset > 0, {full_scan}, {partial_scan})",
                    dtype=partial_scan.dtype,
                    shape=partial_scan.shape,
                )
                for full_scan, partial_scan in zip(full_scan_vars, partial_scan_vars)
            ]
            for acc_next, accumulator, partial_reduce in zip(
                accs_next, accumulators, partial_reduce_vars
            ):
                self.compute.writeline(
                    f"{accumulator} = tl.where(roffset > 0, {acc_next}, {partial_reduce})"
                )
        else:
            result_vars = partial_scan_vars

        for result_var in result_vars:
            assert isinstance(result_var, TritonCSEVariable)
            result_var.mask_vars = OrderedSet(masks)

        return tuple(result_vars)

    # Since we have added a new layer of for loop structure, indentation
    # needs to be added to some parts of the code.
    def codegen_block_ptr(
        self,
        name: str,
        var: str,
        indexing: Union[BlockPtrOptions, TensorDescriptorOptions],
        other="",
    ) -> tuple[str, str]:
        # Modified by cambricon: add an indent context.
        with self.body.indent():
            return super().codegen_block_ptr(name, var, indexing, other)

    # Mainly add indentation to the generated CSE code.
    def load(self, name: str, index: sympy.Expr):
        """
        Load from the memory location 'name', offset by some indexing expression 'index'.
        """
        var = self.args.input(name)
        load_counts = self._load_counts
        load_counts[name] += 1
        make_line: Callable[[str], Union[str, DelayReplaceLine]] = identity
        indirect_indexing = self.is_indirect_indexing(index)
        original_index = index
        dtype = V.graph.get_dtype(name)
        indexing = self.indexing(
            index,
            block_ptr=True,
            tma_compatibility_checker=self.tma_compatibility_checker_cls(
                self,
                dtype,
                for_store=False,
                force=False,
            ),
        )

        if isinstance(indexing, IndexingOptions) and self._has_stride1_on_rdim(
            indexing.index
        ):
            self.has_load_with_contiguous_rdim = True

        has_rindex = indexing.has_rindex()
        has_tmpmask = indexing.has_tmpmask()

        # Keep the variable in cache if were going to reuse it. Equiv., if any of the following hold
        #  1) We are doing broadcasting
        #  2) It is a non-coalesced load. The intuition is that if it's
        #  non-coalesced, we will likely load each element multiple times in
        #  practice.
        #  3) It will be used later and it won't be CSE'd. Equiv., if all the following hold
        #   3.1) We are in a reduction loop
        #   3.2) Its not its last use
        #   3.3) This load will not be lifted to the body
        #
        is_coalesced = any(
            i == 1 for i in self.get_strides_of_load(original_index).values()
        )
        if self.is_broadcasted(original_index):
            ep = ", eviction_policy='evict_last'"
        elif not is_coalesced:
            ep = ", eviction_policy='evict_last'"
        # Modified by cambricon.
        # elif self.inside_reduction and self.range_trees[-1].is_loop:

        #    def decide_later():
        #        if load_counts[name] > expected_count and (
        #            has_rindex or indirect_indexing
        #        ):
        #            return "evict_last"
        #        return "evict_first"

        #    expected_count = load_counts[name]
        #    ep = ", eviction_policy='<EP>'"
        #    make_line = functools.partial(DelayReplaceLine, "<EP>", decide_later)
        # else:
        #    ep = ""
        else:
            # before modified is below:
            # elif self.inside_reduction and self.range_trees[-1].is_loop:
            def decide_later():
                if load_counts[name] > expected_count and (
                    has_rindex or indirect_indexing
                ):
                    return "evict_last"
                return "evict_first"

            expected_count = load_counts[name]
            ep = ", eviction_policy='<EP>'"
            make_line = functools.partial(DelayReplaceLine, "<EP>", decide_later)
        # end Modify by CAMBRICON

        if (has_tmpmask or has_rindex) and indexing.has_mask():
            if self._load_other:
                other = f", other={constant_repr(self._load_other)}"
            else:
                other = ", other=0.0"
        else:
            other = ""

        """Check if the buffer we're about to load, has
        more than one read dependency
        NOTE: enabled with env variable TORCHINDUCTOR_SKIP_L1
        """
        has_read_deps = True
        if config.triton.skip_l1_cache:
            buffer_read_counts = self.features.buffer_read_counts()
            has_read_deps = buffer_read_counts[name] > 1
        """Skip L1 cache if we're (pretty?) sure the data is used only once
        """
        skip_l1_cache = (
            not self.is_broadcasted(original_index)
            and not self.inside_reduction
            and not has_read_deps
            and is_coalesced  # for indirect loads is_coalesced is False?
        )
        cachemod = ""
        if skip_l1_cache:
            cachemod = ", cache_modifier='.cg'"

        append_broadcast = None
        shape: BlockShapeType = None

        if should_unwrap_unspec_arg(name):
            line = var
            # unwrapped bf16/fp16 0d tensors are passed in as float32 scalars
            # see triton_utils.py:signature_of
            if dtype in (torch.float16, torch.bfloat16):
                if config.triton.codegen_upcast_to_fp32:
                    dtype = torch.float32
                else:
                    line += f".to({triton_type(dtype)})"
            shape = ()

        else:
            if isinstance(indexing, (BlockPtrOptions, TensorDescriptorOptions)):
                block_descriptor, other = self.codegen_block_ptr(
                    name, var, indexing, other
                )
                if isinstance(indexing, BlockPtrOptions):
                    # Modify by CAMBRICON
                    # line = f"tl.load({block_descriptor}{other}{ep}{cachemod})"
                    line = f"tl.load({block_descriptor}{other}{ep})"
                else:
                    line = f"{block_descriptor}.load({V.kernel.index_to_str(indexing.offsets)})"
                line = indexing.codegen_broadcast_and_reshape(
                    line,
                    indexing.block_shape,
                    indexing.final_shape,
                    allow_implicit=True,
                    for_store=False,
                )
                shape = indexing.final_shape
            elif is_sympy_integer_like(original_index):
                line = f"tl.load({var} + ({original_index}))"
                append_broadcast = indexing.expand_str
                shape = ()
            else:
                # Modify by CAMBRICON
                # line = f"tl.load({var} + ({indexing.index_str}), {indexing.mask_str}{ep}{other}{cachemod})"
                line = f"tl.load({var} + ({indexing.index_str}), {indexing.mask_str}{ep}{other})"

                # The block shape of tl.load depends on the indexing expression.
                # Inferring shape solely from the mask may miss cases where the mask is constant.
                # Inferring from indexing.expand_shape alone may also fail when dense indexing is absent.
                # so, iterate over variables in the indexexpr to accurately infer the block shape.
                if indexing.expand_shape:
                    shape = indexing.expand_shape
                else:
                    shape = TritonSymbols.get_block_shape(indexing.index)

            if (
                dtype in (torch.float16, torch.bfloat16)
                and config.triton.codegen_upcast_to_fp32
            ):
                line += ".to(tl.float32)"
                dtype = torch.float32
            if dtype == torch.bool and torch.version.hip is None:
                # Workaround for https://github.com/triton-lang/triton/issues/2151
                # tl.load returns int8 when loading from pointer to int1
                # NOTE: Currently causes hangs on bool UTs for ROCm
                line += ".to(tl.int1)"
                dtype = torch.bool

        load_buffer = self.get_load_buffer(indexing)
        # Modify by CAMBRICON: Indentation is determined by load_buffer.
        # Indentation is performed only when load_buffer is body,
        # and no indentation is performed when load_buffer is compute or load.
        if (
            not indexing.has_indirect()
            and not indexing.has_tmpmask()
            and self.inside_reduction
            and not self.persistent_reduction
            and self.range_trees[-1].is_loop
            and not has_rindex
        ):
            load_buffer.do_indent(1)
            result_var = self.cse.generate(
                load_buffer, make_line(line), dtype=dtype, shape=shape
            )
            load_buffer.do_unindent(1)
        else:
            result_var = self.cse.generate(
                load_buffer, make_line(line), dtype=dtype, shape=shape
            )

        if result_var.use_count > 1:
            load_counts[name] -= 1  # don't double count cache hit
        assert isinstance(result_var, TritonCSEVariable)
        result_var.mask_vars = indexing.mask_vars  # type: ignore[assignment]

        if append_broadcast:
            line = f"tl.broadcast_to({result_var}, {append_broadcast})"
            # Modify by CAMBRICON: Indentation is determined by load_buffer.
            # Indentation is performed only when load_buffer is body,
            # and no indentation is performed when load_buffer is compute or load.
            if (
                not indexing.has_indirect()
                and not indexing.has_tmpmask()
                and self.inside_reduction
                and not self.persistent_reduction
                and self.range_trees[-1].is_loop
                and not has_rindex
            ):
                load_buffer.do_indent(1)
                result_var = self.cse.generate(
                    load_buffer, line, dtype=dtype, shape=shape
                )
                load_buffer.do_unindent(1)
            else:
                result_var = self.cse.generate(
                    load_buffer, line, dtype=dtype, shape=shape
                )

            if indexing.mask_vars:
                if dtype.is_floating_point:
                    zero = "0.0"
                elif dtype == torch.bool:
                    zero = "True"
                else:
                    zero = "0"
                other_val = (
                    constant_repr(self._load_other) if self._load_other else zero
                )
                line = f"tl.where({indexing.mask_str}, {result_var}, {other_val})"
                result_var = self.cse.generate(
                    load_buffer, line, dtype=dtype, shape=shape
                )

        if not self.inside_reduction or (not indexing.has_rmask() and not has_rindex):
            self.outside_loop_vars.add(result_var)

        return result_var


class MluTritonScheduling(TritonScheduling):
    kernel_type = MluTritonKernel

    # In addition to generating the original candidate tiling list, we have
    # added a method to generate more tiling options based on strides, allowing
    # for the creation of multidimensional indices.
    @classmethod
    @functools.lru_cache(32)
    def candidate_tilings(cls, node, numel, reduction_numel) -> list[CandidateTiling]:
        # tilings = SIMDScheduling.candidate_tilings(node, numel, reduction_numel)
        is_pointwise = reduction_numel == 1

        def tile_ranges(is_pointwise: bool, ranges, rw) -> list[CandidateTiling]:
            """
            Compute tiling candidates by dividing up the iteration ranges.
            """
            assert len(rw.range_vars) == len(ranges), f"{rw.range_vars=} {ranges=}"

            # isinstance(dep, MemoryDep): this filters out StarDeps. StarDeps refer to reads
            # that need to access the entire tensor; they don't contribute read indexing
            # information (and practically, they don't have dep.index so they can't be used
            # for stride_hints below
            dep_sources = [rw.reads, rw.writes]
            assert all(
                isinstance(dep, (MemoryDep, StarDep))
                for dep in itertools.chain.from_iterable(dep_sources)
            )
            deps = [
                dep
                for dep in itertools.chain.from_iterable(dep_sources)
                if dep.name not in V.graph.removed_buffers
                and isinstance(dep, MemoryDep)
            ]
            write_names = OrderedSet([dep.name for dep in rw.writes])

            def collapse_ranges(ranges: Sequence[sympy.Expr]) -> sympy.Expr:
                return V.graph.sizevars.simplify(sympy_product(ranges))

            # Default to no tiling.
            tilings = [
                CandidateTiling(
                    tiling=cls.create_partial_tiling([collapse_ranges(ranges)], 1),
                    name="none",
                    score=0,
                )
            ]

            def is_regular_strides(shape, strides):
                if len(shape) != len(strides):
                    return False
                expected_stride = 1
                for dim, stride in zip(reversed(shape), reversed(strides)):
                    if stride != expected_stride:
                        return False
                    expected_stride *= dim
                return True

            # Find non-trivial tiling candidates.
            for dep in deps:
                strides = V.graph.sizevars.stride_hints(dep.index, rw.range_vars)
                assert len(strides) == len(ranges)
                skip = is_regular_strides(ranges, strides)
                splits = [0]
                previous_is_zero = len(strides) > 0 and strides[0] == 0
                for i in range(1, len(strides)):
                    current_is_zero = strides[i] == 0
                    current_is_one = strides[i] == 1
                    if previous_is_zero != current_is_zero:
                        splits.append(i)
                        previous_is_zero = current_is_zero
                    if i == len(strides) - 1 and current_is_one and i not in splits:
                        if skip:
                            continue
                        splits.append(i)

                if len(splits) == 1:
                    continue
                splits.append(len(strides))
                if len(splits) > 4:
                    splits = splits[:3] + splits[-1:]

                tiled_groups = tuple(
                    V.graph.sizevars.simplify(
                        sympy_product(ranges[splits[i] : splits[i + 1]])
                    )
                    for i in range(len(splits) - 1)
                )
                score = V.graph.sizevars.size_hint(sympy_product(ranges))
                score *= len(splits)

                if (
                    V.graph.sizevars.size_hint(
                        score - sympy_product(itertools.chain(ranges, reduction_ranges))
                    )
                    >= 0
                ):
                    tilings.append(
                        CandidateTiling(
                            tiling=cls.create_partial_tiling(
                                tiled_groups,
                                reduction_numel,
                            ),
                            score=score,
                            name=dep.name,
                        )
                    )

            return tilings

        pointwise_ranges, reduction_ranges = node.get_ranges()
        if len(pointwise_ranges) <= 1 and len(reduction_ranges) <= 1:
            return []

        # Tile either pointwise or reduction dims.
        pointwise_ranges, reduction_ranges = node.get_ranges()
        partial_tilings = tile_ranges(
            is_pointwise,
            pointwise_ranges,
            # pointwise_ranges if is_pointwise else reduction_ranges,
            node.pointwise_or_reduction_read_writes(is_pointwise),
        )

        # Fill in the missing ranges.
        full_tilings = [
            CandidateTiling(
                tiling=cls.complete_partial_tiling(
                    tiling.tiling, numel, reduction_numel
                ),
                score=tiling.score,
                name=tiling.name,
            )
            for tiling in partial_tilings
        ]

        return full_tilings

    # Just remove the lines in "if config.triton.max_tiles >= 3:" scope.
    @classmethod
    def select_tiling(
        cls, node_schedule, numel, reduction_numel=sympy.S.One
    ) -> Dict[str, sympy.Expr]:
        """
        Heuristics to decide how to tile kernels.
        Currently, we tile based on stride-1 dimensions.

        Returns:
            `(tile1, tile2, reduction_numel)` s.t. `tile1 * tile2 == numel`

        """
        # If this is a reduction, only tile reduction dims.
        is_pointwise = reduction_numel == 1

        # Tiled reductions are gated by a config flag.
        default_tiling = cls.create_tiling([numel], [reduction_numel])

        seen_names = OrderedSet[str]()
        candidate_tiles: Counter[CandidateTiling] = collections.Counter()
        for node in EnableReduction.filter(node_schedule):
            for candidate_tiling in cls.candidate_tilings(node, numel, reduction_numel):
                if candidate_tiling.name in seen_names:
                    continue
                elif candidate_tiling.name is not None:
                    seen_names.add(candidate_tiling.name)
                candidate_tiles[candidate_tiling] += candidate_tiling.score

        ranked_tilings: list[dict[str, sympy.Expr]] = [
            candidate_tiling.tiling
            for candidate_tiling, score in candidate_tiles.most_common()
        ]

        if len(ranked_tilings) > 1:
            perf_hint_log.info("possibly bad tiling: %s", ranked_tilings)

        # Optionally, prefer tiling into as many dimensions as possible.
        if config.triton.prefer_nd_tiling:
            ranked_tilings = (
                cls.get_nd_tilings(node_schedule, numel, reduction_numel)
                + ranked_tilings
            )

        if not is_pointwise:
            ranked_tilings = [r for r in ranked_tilings if r["r0_"] != 1]

        for tiling in ranked_tilings:
            assert isinstance(tiling, dict)
            if all(
                SIMDKernel.is_compatible(
                    tiling.values(), node.get_ranges(), reduction_numel=reduction_numel
                )
                for node in node_schedule
                if isinstance(node, scheduler.SchedulerNode)
            ):
                return tiling

        return default_tiling

    @classmethod
    def get_nd_tilings(
        cls,
        node_schedule,
        pointwise_numel,
        reduction_numel,
    ) -> list[dict[str, tuple[sympy.Expr]]]:
        """
        Creates N-dimensional tiling candidates, attempting to simplify loads/stores
        by tiling the kernel into higher dimensions.

        Returns a list of tilings ranked by dimensionality.
        """
        tilings = OrderedSet[dict[str, sympy.Expr]]()
        for node in EnableReduction.filter(node_schedule):
            if not isinstance(node, scheduler.SchedulerNode):
                continue

            # If this is a reduction schedule, skip nodes which are missing their
            # reduction ranges.
            node_ranges = node.get_ranges()

            # Use the node ranges as the default tiling candidate.
            ranges_to_tile = node_ranges[0]
            node_tilings = [ranges_to_tile]

            # Search the indexing expressions for more candidates.
            # If we see modular indexing, try to subdivide ranges into their implied
            # block shape.
            memory_deps = [
                dep
                for dep in node.read_writes.reads_and_writes()
                if isinstance(dep, MemoryDep) and len(dep.ranges) > 0
            ]
            for dep in memory_deps:
                # Attempt to partition variable ranges into pointwise and reduction groups.
                # To achieve this, merge the leading ranges until we reach the pointwise numel.
                all_var_ranges = [*dep.ranges.items()]
                pointwise_vars_numel = sympy.S.One
                sizevars = V.graph.sizevars
                for pointwise_end_idx, (var, numel) in enumerate(all_var_ranges):
                    pointwise_vars_numel *= numel
                    if sizevars.statically_known_geq(
                        pointwise_vars_numel, pointwise_numel
                    ):
                        break

                # Reject the split if it does not match the total pointwise numel.
                if not sizevars.statically_known_equals(
                    pointwise_vars_numel, pointwise_numel
                ):
                    continue

                # Partition var ranges into pointwise and reduction splits.
                reduction_start_idx = pointwise_end_idx + 1
                var_ranges = (
                    all_var_ranges[:reduction_start_idx]
                    # TODO(miaochen): support reduce tiling
                    # if is_pointwise
                    # else all_var_ranges[reduction_start_idx:]
                )

                # Pattern match the subexpression pertaining to each index variable.
                index_tiling = []
                for var, numel in var_ranges:
                    index = BlockPatternMatcher.get_subexpr_involving_symbol(
                        dep.index, var
                    )

                    # Heuristic to bound the maximum dimensionality of the block.
                    num_dims = max(
                        2,
                        index.count(FloorDiv) + index.count(ModularIndexing),
                        len(ranges_to_tile),
                    )

                    # Attempt to pattern match the index expr.
                    # Failed matches default to the full range.
                    match_result = BlockPatternMatcher.match_mod_div_block_expr(
                        index, var, numel, num_dims
                    )
                    dims = match_result[0] if match_result is not None else [numel]
                    index_tiling.extend(dims)

                # Modify by CAMBRICON
                # cp from torch 2.8
                index_tiling = [
                    dim
                    for dim in index_tiling
                    if not V.graph.sizevars.statically_known_equals(dim, sympy.S.One)
                ]
                if len(index_tiling) > 0:
                    node_tilings.append(index_tiling)

            # Flatten leading dimensions, assigning labels to each dim.
            for node_tiling in node_tilings:
                num_leading_dims = max(0, len(node_tiling) - config.triton.max_tiles)
                first_trailing_dim = num_leading_dims + 1
                collapsed_leading_dim = sympy_product(node_tiling[:first_trailing_dim])
                collapsed_splits = (collapsed_leading_dim,) + tuple(
                    node_tiling[first_trailing_dim:]
                )
                tilings.add(
                    cls.complete_partial_tiling(
                        cls.create_partial_tiling(collapsed_splits, 1),
                        pointwise_numel,
                        reduction_numel,
                    )
                )

        # Rank tilings by the number of dimensions. E.g., prefer 2D to 1D.
        # Since this is a stable sort, ties are broken by schedule order.
        ranked_tilings = sorted(
            tilings,
            key=len,
            reverse=True,
        )

        return ranked_tilings


def want_no_x_dim(self):
    # Modify by CAMBRICON
    # return (
    #     self.persistent_reduction
    #     and len(self.numels) == self.num_reduction_dims + 1
    #     and self.fixed_config
    #     and self.fixed_config["XBLOCK"] == 1
    # )
    # end Modify by CAMBRICON
    return False


patch = gorilla.Patch(TritonKernel, "want_no_x_dim", want_no_x_dim)
gorilla.apply(patch)


@classmethod
def create(
    cls,
    *,
    params: BlockParameters,
    constant_offset: sympy.Expr,
    range_trees: list[IterationRangesEntry],
    mask_vars: OrderedSet[str],
    get_max_block: Callable[[str], int],
    stride_sorter_cls: type[BlockParameters.StrideSorter],
    can_lift: bool = False,
) -> BlockDescriptorOptions:
    """Helper to create a  BlockDescriptorOptions instance"""

    sizevars = V.graph.sizevars

    def lookup_size(exprs: Iterable[sympy.Expr]) -> list[sympy.Expr]:
        return [sizevars.lookup_precomputed_size(expr) for expr in exprs]

    # Look up precomputed sizes
    params.shape = lookup_size(params.shape)
    params.strides = lookup_size(params.strides)

    # Strip out dimensions of size 1.
    # Size 1 dimensions are redundant since the triton kernel shape
    # will be e.g. [YBLOCK, XBLOCK], so tl.reshape would just remove these
    # dimensions anyway
    singleton_dims = [
        sizevars.statically_known_equals(dim, 1) for dim in params.block_shape
    ]
    if all(singleton_dims):
        # Handle a pure singletons, e.g. [1, 1]
        singleton_dims[-1] = False

    # Drop singleton dimensions from the block descriptor.
    params = params.remove_dims(singleton_dims)

    # Maybe reorder dimensions based on strides
    # with tl.trans applied at load / store time
    params, stride_sorter = params.maybe_sort_with_stride_order(
        stride_sorter_cls=stride_sorter_cls, shape_env=V.graph._shape_env
    )

    # Strip out dimensions of stride 0.
    # These will be restored with tl.broadcast_to.
    broadcasting_dims = [
        sizevars.statically_known_equals(stride, 0) for stride in params.strides
    ]

    # Record the post-broadcast shape before broadcasting dims are removed.
    # The pre-broadcast shape is identical to this, except broadcasting dims are
    # replaced with 1.
    broadcast_shape = params.block_shape

    # Drop broadcasting dims from the block descriptor.
    params = params.remove_dims(broadcasting_dims)

    # Compute the final shape, adjusting for special kernel types.
    final_shape = [TritonSymbols.get_block_size(tree) for tree in range_trees]
    if V.kernel.no_x_dim:
        assert range_trees[0].prefix == "x"
        final_shape.pop(0)

    reduction_ndim = V.kernel.num_reduction_dims
    if (
        not V.kernel.inside_reduction
        # Modified by Cambricon: set [XBLOCK] to [XBLOCK, 1] in reduction for broadcast
        # and len(params.strides) == len(V.kernel.numels) - reduction_ndim
        and V.kernel.features.is_reduction()
    ):
        # Need to expand rank to match the rank used inside the reduction loop
        final_shape += [sympy.S.One] * reduction_ndim

    try:
        # Get permutation to sort strides in ascending order.
        # This is used as the order argument in tl.make_block_ptr
        order = utils.argsort_sym(V.graph._shape_env, params.strides)
    except AssertionError:
        # Symbolic shapes, failed to evaluate comparison expression
        order = list(reversed(range(len(params.strides))))

    def argsort(stride_list):
        if any(s.is_symbol for s in stride_list):
            return list(reversed(range(len(stride_list))))
        return sorted(range(len(stride_list)), key=lambda i: stride_list[i])

    result = BlockPtrOptions(
        params=params,
        constant_offset=V.graph.sizevars.lookup_precomputed_size(constant_offset),
        # Modified by cambricon: generate order by strides value.
        # order=list(reversed(range(len(params.shape)))),
        order=argsort(params.strides),
        mask_vars=mask_vars,
        final_shape=final_shape,
        broadcast_shape=broadcast_shape,
        broadcasting_dims=broadcasting_dims,
        stride_sorter=stride_sorter,
        can_lift=can_lift,
    )
    result.compute_boundary_check(get_max_block, range_trees)
    return result


patch = gorilla.Patch(torch._inductor.codegen.triton.BlockPtrOptions, "create", create)
gorilla.apply(patch)


def compute_boundary_check(
    self,
    get_max_block: Callable[[str], int],
    range_trees: list[IterationRangesRoot],
) -> None:
    """List of indices to pass to tl.load(boundary_check=...)"""
    sizevars = V.graph.sizevars

    # Substitute maximum block sizes in shape expressions.
    # This works in multiple_of checks because block sizes are powers of 2.
    # Modify by CAMBRICON
    # block_to_max: dict[sympy.Expr, Any] = {
    #     TritonSymbols.block_sizes[t.symt]: get_max_block(prefix_str[t.symt]) for t in range_trees
    # }
    # end Modify by CAMBRICON

    # Also see Note: Constant mask optimisation
    # if ynumel / YBLOCK > max_ygrid, then the z dimension is used to handle
    # the remaining programs that cannot fit into the y dimension. This means
    # it's possible that more than the required number of programs are launched,
    # possibly leading to out-of-bounds accesses. So even if ynumel divides YBLOCK,
    # boundary checking is required in the dimensions that are based on YBLOCK
    # e.g. for [YBLOCK // 16, YBLOCK, XBLOCK] dimensions 0 and 1 need boundary
    # checks when max_ygrid is exceeded.
    needs_overflow_grid = any(map(V.kernel.needs_yz_grid_overflow, range_trees))
    # Modify by CAMBRICON
    # self._boundary_check = [
    #     idx
    #     for idx in range(len(self.shape))
    #     if (
    #         not sizevars.statically_known_equals(self.strides[idx], sympy.S.Zero)
    #         and (
    #             (
    #                needs_overflow_grid
    #                and TritonSymbols.block_sizes[SymT.YBLOCK] in self.block_shape[idx].free_symbols
    #             )
    #             or (
    #                 not sizevars.statically_known_multiple_of(
    #                     self.shape[idx], self.block_shape[idx]
    #                 )
    #                 and not sizevars.statically_known_multiple_of(
    #                     self.shape[idx],
    #                     sympy_subs(self.block_shape[idx], block_to_max),
    #                 )
    #             )
    #         )
    #         and not (
    #             V.kernel.no_x_dim
    #             and self.block_shape[idx] == TritonSymbols.block_sizes[SymT.XBLOCK]
    #         )
    #     )
    # ]
    self._boundary_check = [
        idx
        for idx in range(len(self.shape))
        if (
            not sizevars.statically_known_equals(self.strides[idx], sympy.Integer(0))
            and not (
                V.kernel.no_x_dim
                and self.block_shape[idx] == TritonSymbols.block_sizes[SymT.XBLOCK]
            )
        )
    ]
    # end Modify by CAMBRICON


patch = gorilla.Patch(
    torch._inductor.codegen.triton.BlockPtrOptions,
    "compute_boundary_check",
    compute_boundary_check,
)
gorilla.apply(patch)


def benchmark_codegened_module(
    self, mod, n_spills_threshold=8, node_names: Optional[OrderedSet[str]] = None
) -> tuple[float, str]:
    """Benchmark an already compiled module"""
    device_interface = get_interface_for_device(V.graph.device_type)
    with (
        preserve_rng_state(),
        device_interface.device(V.graph.get_current_device_or_throw()),  # type: ignore[attr-defined]
    ):
        ms = None

        def cache_file_path():
            assert mod.__file__ is not None
            return os.path.splitext(mod.__file__)[0] + ".kernel_perf"

        def store_cache():
            path = cache_file_path()
            # Modify by CAMBRICON
            # write_atomic(path, str(ms))
            from torch._inductor import codecache

            codecache.write_atomic(path, str(ms))
            # end Modify by CAMBRICON

        def load_cache():
            path = cache_file_path()
            if os.path.exists(path):
                with open(path) as fd:
                    return float(fd.read())
            return None

        node_names = node_names if node_names is not None else OrderedSet(["unknown"])
        log.debug(
            "kernel src code for %s written to: %s",
            node_names,
            mod.__file__,
        )
        ms = load_cache()
        if ms is not None:
            return ms, mod.__file__

        args = mod.get_args()
        call = mod.call
        wrapped_jit_function = mod.triton_
        # call once to trigger the compilation
        try:
            call(wrapped_jit_function.clone_args(*args)[0])
        except Exception as e:
            if config.triton.disallow_failing_autotune_kernels_TESTING_ONLY:
                raise
            log.debug(  # noqa: G200
                "Exception (%s) in compiling fused nodes %s",
                e,
                node_names,
            )
            ms = float("inf")
            store_cache()
            return ms, mod.__file__

        launchers = wrapped_jit_function.launchers
        assert len(launchers) == 1
        # n_spills does not necessarily mean it's not profitable to fuse,
        # and sometimes it can be inaccurate
        if launchers[0].n_spills > n_spills_threshold:
            # skip benchmarking the kernel if there are register spills
            ms = float("inf")
        else:
            device = V.graph.get_current_device_or_throw()
            # We have to clone the inplace updated arguments to avoid earlier calls
            # generating out of range indices for later calls.
            ms = benchmarker.benchmark(
                lambda: call(wrapped_jit_function.clone_args(*args)[0]),
                device=device,
            )
            # overhead of cloning args gives bias for fusing the kernel
            # in the case of mutating/in-placeable second fusion
            # TODO - would be better as a hook in triton do_bench that reset
            # the input values between benchmarking
            if len(wrapped_jit_function.mutated_arg_names) > 0:
                ms = ms - benchmarker.benchmark(
                    lambda: wrapped_jit_function.clone_args(*args),
                    device=str(device),
                )

        log.debug(
            "The fused kernel for %s took %.3f ms to run",
            node_names,
            ms,
        )
        store_cache()
        return ms, mod.__file__


patch = gorilla.Patch(
    TritonScheduling, "benchmark_codegened_module", benchmark_codegened_module
)
gorilla.apply(patch)


def benchmark_combo_kernel(self, node_list, node_benchmark_results):
    """
    Benchmark combo kernel partitions and return total execution time.

    Generates kernel code for each partition and benchmarks them.
    Single-node partitions use benchmark_fused_nodes(), while multi-node
    partitions use the combo kernel benchmarking path.

    Returns (total_ms, total_clone_ms, file_list).
    """
    mod: ModuleType
    ms: float
    ms_clone: float

    def cache_file_path():
        assert mod.__file__ is not None
        return os.path.splitext(mod.__file__)[0] + ".kernel_perf"

    def load_cache():
        path = cache_file_path()
        if os.path.exists(path):
            with open(path) as fd:
                return tuple(float(e) for e in fd.read().split())
        return (None, None)

    def store_cache():
        path = cache_file_path()
        # Modify by CAMBRICON
        # write_atomic(path, str(ms) + " " + str(ms_clone))
        from torch._inductor import codecache

        codecache.write_atomic(path, str(ms) + " " + str(ms_clone))
        # end Modify by CAMBRICON

    total_ms, file_list = 0, []
    total_clone_ms: float = 0.0
    removed_buffers_orig = V.graph.removed_buffers
    V.graph.removed_buffers = OrderedSet(removed_buffers_orig)
    inplaced_to_remove_orig = V.graph.inplaced_to_remove
    V.graph.inplaced_to_remove = OrderedSet(inplaced_to_remove_orig)
    enable_autotune = config.combo_kernels_autotune > 0
    mixed_sizes = config.combo_kernel_allow_mixed_sizes > 0
    kernel_code_list = self.generate_combo_kernel_code(
        subkernel_nodes=node_list,
        custom_part_algorithm=True,
        enable_autotune=enable_autotune,
        mixed_sizes=mixed_sizes,
        only_gen_src_code=True,
    )

    # pyrefly: ignore [bad-assignment]
    for src_code, kernel, node_group in kernel_code_list:
        fused_node_lists = [node.get_nodes() for node in node_group]
        names = [n.get_name() for nodes in fused_node_lists for n in nodes]

        if len(node_group) == 1:
            # Single-node partition: use cached benchmark results from speedup_by_combo_kernel
            node_ms, path = node_benchmark_results[node_group[0]]
            # Regular kernels have negligible clone overhead
            total_ms += node_ms
            total_clone_ms += 0
            file_list.append(path)
            continue

        assert src_code is not None
        src_code = src_code.replace(str(Placeholder.KERNEL_NAME), "triton_")
        mod = PyCodeCache.load(src_code)

        log.debug(
            "kernel src code for %s written to: %s",
            names,
            mod.__file__,
        )
        ms, ms_clone = load_cache()
        if ms is not None:
            total_ms += ms  # type: ignore[assignment]
            total_clone_ms += ms_clone
            file_list.append(mod.__file__)
            continue

        args = mod.get_args()
        call = mod.call
        wrapped_jit_function = mod.triton_

        # call once to trigger the compilation
        call(wrapped_jit_function.clone_args(*args)[0])

        launchers = wrapped_jit_function.launchers
        assert len(launchers) == 1
        if launchers[0].n_spills > 0:
            # skip benchmarking the kernel if there are register spills
            ms = ms_clone = float("inf")
        else:
            device = V.graph.get_current_device_or_throw()
            # We have to clone the inplace updated arguments to avoid earlier calls
            # generating out of range indices for later calls.
            ms = benchmarker.benchmark(
                lambda: call(wrapped_jit_function.clone_args(*args)[0]),
                device=device,
            )
            ms_clone = benchmarker.benchmark(
                lambda: wrapped_jit_function.clone_args(*args)[0],
                device=device,
            )

        log.debug(
            "The fused kernel for %s took %.3f ms to run, %.3f ms to clone inputs",
            OrderedSet(n.get_name() for n in node_group),
            ms,
            ms_clone,
        )
        store_cache()
        total_ms += ms
        total_clone_ms += ms_clone
        file_list.append(mod.__file__)
    V.graph.removed_buffers = removed_buffers_orig
    V.graph.inplaced_to_remove = inplaced_to_remove_orig
    return total_ms, total_clone_ms, file_list


patch = gorilla.Patch(
    TritonScheduling, "benchmark_combo_kernel", benchmark_combo_kernel
)
gorilla.apply(patch)


def _print_Float(self, expr: sympy.Expr) -> str:
    if expr.is_integer:
        # sympy considers 0.0 to be integer, but triton doesn't.
        # this workaround prints the float as an integer
        # xref: https://github.com/sympy/sympy/issues/26620
        ret = str(int(expr))
    elif config.is_fbcode() and torch.version.hip:
        ret = f"{expr}"
    else:
        # Modify by cambricon, because of RuntimeError: MLU unsupported floating-point type fp64
        # ret = f"tl.full([], {expr}, tl.float64)"
        ret = f"tl.full([], {expr}, tl.float32)"
        # end Modify by cambricon
    return ret


patch = gorilla.Patch(TritonPrinter, "_print_Float", _print_Float)
gorilla.apply(patch)


def _print_ToFloat(self, expr: sympy.Expr) -> str:
    assert len(expr.args) == 1
    # pyrefly: ignore [bad-argument-type]
    s = self.parenthesize(expr.args[0], PRECEDENCE["Atom"] - 0.5)
    # Modify by cambricon, because of RuntimeError: MLU unsupported floating-point type fp64
    # return f"{s}.to(tl.float64)"
    return f"{s}.to(tl.float32)"
    # end Modify by cambricon


patch = gorilla.Patch(TritonPrinter, "_print_ToFloat", _print_ToFloat)
gorilla.apply(patch)


def _print_FloatPow(self, expr: sympy.Expr) -> str:
    # pyrefly: ignore [missing-attribute]
    base = self._print(expr.args[0])
    # pyrefly: ignore [missing-attribute]
    exp = self._print(expr.args[1])
    # Modify by cambricon, because use float32 instead of float64 (MLU does not support fp64)
    # libdevice.pow requires both arguments to have the same type.
    # Always cast to float64 for consistency. This is scalar shape math,
    # not tensor ops, so the performance impact is negligible.
    # return f"libdevice.pow(({base}).to(tl.float64), ({exp}).to(tl.float64))"
    return f"libdevice.pow(({base}).to(tl.float32), ({exp}).to(tl.float32))"
    # end Modify by cambricon


patch = gorilla.Patch(TritonPrinter, "_print_FloatPow", _print_FloatPow)
gorilla.apply(patch)


def _print_PowByNatural(self, expr: sympy.Expr) -> str:
    if expr.args[0].is_Integer:
        # Modify by cambricon, because use float32 instead of float64 (MLU does not support fp64)
        # base = f"tl.full([], {float(expr.args[0])}, tl.float64)"
        base = f"tl.full([], {float(expr.args[0])}, tl.float32)"
        # end Modify by cambricon
    else:
        # pyrefly: ignore [missing-attribute]
        # Modify by cambricon
        # base = f"({self._print(expr.args[0])}).to(tl.float32)"
        base = f"({self._print(expr.args[0])}).to(tl.float64)"
        # end Modify by cambricon
    exp_val = expr.args[1]
    if exp_val.is_Integer:
        # Modify by cambricon
        # exp = f"tl.full([], {float(exp_val)}, tl.float64)"
        exp = f"tl.full([], {float(exp_val)}, tl.float32)"
        # end Modify by cambricon
    else:
        # pyrefly: ignore [missing-attribute]
        # Modify by cambricon
        # exp = f"({self._print(exp_val)}).to(tl.float64)"
        exp = f"({self._print(exp_val)}).to(tl.float32)"
        # end Modify by cambricon
    # libdevice.pow requires both arguments to have the same type.
    # Always cast to float64 for consistency. This is scalar shape math,
    # not tensor ops, so the performance impact is negligible.
    return f"libdevice.pow({base}, {exp})"


patch = gorilla.Patch(TritonPrinter, "_print_PowByNatural", _print_PowByNatural)
gorilla.apply(patch)
