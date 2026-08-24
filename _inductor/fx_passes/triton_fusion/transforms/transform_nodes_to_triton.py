"""
@Copyright (C) [2022-2025] by Cambricon.
@File    :   transform_nodes_to_triton.py
"""

import os, copy
from typing import Callable, List, Mapping, Optional, Tuple
import sympy
from sympy import *
import torch
import torch._inductor.config as inductor_config
from torch.utils._sympy.value_ranges import bound_sympy
from torch.fx.node import Node
import importlib, math
import operator
from .utils import (
    IndentedBuffer,
    stride_order,
    get_inputs_outputs,
    get_total_core_num,
    NUM_WARPS,
)
from ..processors import (
    convert_to_triton,
    infer_tiledim_back_all,
    get_externkernelchoice,
    get_op_processor,
)
from ..config import (
    cache_dir_path,
    test_fallback_kernel,
    target_batch_size,
    get_simple_logger,
)
from ..common import (
    is_shape_dynamic,
    TORCH2TRITON_DTYPE_STR,
    TILEDIMNAME,
    NUMTASKSNAME,
    EVENBSBLOCKNAME,
    TENSORMETANAME,
    VALMETANAME,
    TRITONFUSIONDEBUGNAME,
    TRITONFUSION_SAVE_TENSOR_ENV,
    TRITONFUSION_LOAD_TENSOR_ENV,
    TORCH2TRITON_LOAD_STORE_DTYPE_STR,
    BATCHBLOCKNAME,
    TRITONFUSION_ENABLE,
    TRITONFUSION_CUSTOM_OPS_NAME,
    FORCE_USE_SM,
    DEVICENAME,
    get_target_name,
    is_tensor_node,
    get_tensor_metas,
    get_shape_exprs,
)
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
PID = os.getpid()
logger = get_simple_logger(__name__)


class TransformTorch2Triton:
    def __init__(
        self,
        fxgraph: torch.fx.graph.Graph,
        targetnodes: dict[Node, list] = {},
        graphname: str = "triton_graph",
        packagename: str = f"triton_package_{TIMESTAMP}_{PID}",
        start_node: Node = None,
    ):
        self.targetnodes = sorted(targetnodes)
        self.targetnodes_tiledims = targetnodes.copy()
        self.set_tiledims_to_nodes(self.targetnodes_tiledims)

        self.fxgraph = fxgraph
        self.graphname = graphname
        if start_node is not None:
            assert (
                start_node in targetnodes
            ), f"start node must in targetnodes, which are {start_node} and {targetnodes}"
        self.start_node = start_node

        self.fused_kernel_name_ops = self.get_tritontemplate_name(
            graphname, self.targetnodes
        )
        self.graphname_wrapper = self.graphname + "_wrapper"
        self.graphname_wrapper_custom = self.graphname + "_wrapper_custom"
        self.graphname_wrapper_fake = self.graphname + "_fake"
        self.graphname_wrapper_fake_custom = self.graphname + "_fake_custom"
        self.graphname_triton = self.graphname + "_triton"
        # for inductor template
        self.graphname_tritontemplate = self.graphname + "_triton_template"
        self.triton_template_name = self.graphname_tritontemplate
        self.graphname_tuned_inductor = "triton_fusion_tuned_" + self.graphname
        self.graphname_fallback_aten = self.graphname + "_fallback_aten"
        # for diff/perf tests
        self.graphname_diff_torch = "diff_torch_" + self.graphname
        self.graphname_diff = "diff_" + self.graphname
        self.graphname_testdiff = "test_diff_" + self.graphname

        self.graphname_perf_triton = "perf_triton_" + self.graphname
        self.graphname_perf_torch = "perf_torch_" + self.graphname

        # For var perf test funcs.
        self.graphname_testperf_localtriton = "test_perf_localtriton_" + self.graphname
        self.graphname_getperf_localtriton = "get_perf_localtriton_" + self.graphname

        self.graphname_testperf_eager = "test_perf_eager_" + self.graphname
        self.graphname_getperf_eager = "get_perf_eager_" + self.graphname

        self.graphname_testperf_inductor = "test_perf_inductor_" + self.graphname
        self.graphname_getperf_inductor = "get_perf_inductor_" + self.graphname

        self.graphname_testperf_tritonfusion = (
            "test_perf_tritonfusion_" + self.graphname
        )
        self.graphname_getperf_tritonfusion = "get_perf_tritonfusion_" + self.graphname

        self.graphname_testperf_all = "test_perf_all_" + self.graphname
        self.graphname_getperf_all = "get_perf_all_" + self.graphname
        # for other utils
        self.graphname_getinputs = "get_inputs"
        self.graphname_getinoutbytes = "get_in_out_bytes"
        self.graphname_torchfunc = self.graphname + "_torch"
        self.packagename = packagename
        self.indented_buffer = IndentedBuffer()
        self.extral_lines = IndentedBuffer()
        self.input_nodes, self.output_nodes = get_inputs_outputs(self.targetnodes)
        assert (
            len(self.output_nodes) >= 1
        ), f"get no output nodes for fused nodes: {self.targetnodes}"
        self.input_shape_symbols = self.get_nodes_sym_inds(
            self.input_nodes, lambda x: list(x.shape)
        )
        self.input_stride_symbols = self.get_nodes_sym_inds(
            self.input_nodes, lambda x: list(x.stride)
        )
        logger.debug(
            f"get subgraph input node shapes' all symbols: {self.input_shape_symbols}"
        )
        logger.debug(
            f"get subgraph input node strides' all symbols: {self.input_stride_symbols}"
        )

        self.module_all = None
        self.triton_wraper_func = None
        self.diff_func_torch = None
        self.diff_func = None
        self.diff_test_func = None

        self.perf_func_triton = None
        self.perf_func_torch = None
        # All perf get func interfaces.
        self.perf_get_func_eager = None
        self.perf_get_func_localtriton = None
        self.perf_get_func_inductor = None
        self.perf_get_func_tritonfusion = None
        self.perf_get_func_all = None

        self.torch_func = None
        self.input_func = None
        self.in_out_bytes_func = None
        self.inductor_func = None
        self.tiledimsize = self.get_tiledimsize(self.start_node, self.output_nodes)
        # generate all kernels and functions
        self.transform()

        # clean up tile dimensions
        self.clean_tiledims(self.targetnodes_tiledims)

    def get_tritontemplate_name(self, graphname, targetnodes: dict[Node, set]):
        ret = "tritonfusion_" + graphname
        last_name = ""
        last_cnt = 1
        for node in targetnodes:
            if node.target == operator.getitem:
                fn_name = "getitem"
            else:
                fn_name = node.target.name().split("::")[-1].replace(".", "_")
            if fn_name == last_name:
                last_cnt += 1
            else:
                if last_name:
                    ret += f"_{last_name.split('_')[0]}"
                    if last_cnt > 1:
                        ret += f"_X{last_cnt}"
                last_name = fn_name
                last_cnt = 1
        if last_name:
            ret += f"_{last_name.split('_')[0]}"
            if last_cnt > 1:
                ret += f"_X{last_cnt}"
        ret += "_"
        return ret.replace("-", "_")

    def get_valid_expr_str(self, expr: sympy.Expr):
        free_syms = list(expr.free_symbols)
        if len(free_syms) == 0 or (
            len(free_syms) == 1 and isinstance(expr, sympy.core.symbol.Symbol)
        ):
            return str(expr)
        else:
            ret_str = (
                ("FuseSym_" + str(expr))
                .replace(" ", "")
                .replace("+", "Add")
                .replace("-", "Sub")
                .replace("*", "Mul")
                .replace("/", "Div")
                .replace("(", "_")
                .replace(")", "_")
            )
            return ret_str

    def get_valid_expr_str_with_inputs(self, expr: sympy.Expr | int):
        if not is_shape_dynamic(expr):
            return str(expr)
        expr = get_shape_exprs(expr)
        if expr in self.input_shape_symbols or expr in self.input_stride_symbols:
            return self.get_valid_expr_str(expr)
        return str(expr)

    def get_nodes_sym_inds(self, nodes, fn=lambda x: list(x.shape)):
        """
        First update single sym define, including multi syms with only
        one sym undefined, the dict after python3.7 store keys with
        insertion order will keep the generation inorder.
        """
        ret = {}

        update_sym = True
        while update_sym:
            update_sym = False
            for node in nodes:
                if not is_tensor_node(node):
                    continue
                metas = get_tensor_metas(node)
                for mind, meta in enumerate(metas):
                    for sind, sha in enumerate(fn(meta)):
                        if is_shape_dynamic(sha):
                            expr = get_shape_exprs(sha)
                            if expr in ret:
                                continue
                            free_syms = list(expr.free_symbols)
                            unresolve_syms = [x for x in free_syms if x not in ret]
                            if len(unresolve_syms) == 1:
                                ret[unresolve_syms[0]] = (node, mind, sind)
                                update_sym = True

        for node in nodes:
            if not is_tensor_node(node):
                continue
            metas = get_tensor_metas(node)
            for mind, meta in enumerate(metas):
                for sind, sha in enumerate(fn(meta)):
                    if is_shape_dynamic(sha):
                        expr = get_shape_exprs(sha)
                        if expr in ret:
                            continue
                        free_syms = list(expr.free_symbols)
                        unresolve_syms = [x for x in free_syms if x not in ret]
                        if len(unresolve_syms) > 1:
                            ret[expr] = (node, mind, sind)
        return ret

    def solve_result_on_new_symbol(self, expr, new_sym, solve_sym=None):
        expr = get_shape_exprs(expr)
        free_syms = list(expr.free_symbols)
        assert len(free_syms) >= 1, f"only support >=1 symbol in expr, but get: {expr}"
        if solve_sym is not None:
            assert (
                solve_sym in free_syms
            ), f"solve symbol should in expr, but get: {solve_sym}  expr: {expr}"
        else:
            solve_sym = free_syms[0]
        if isinstance(new_sym, str):
            new_sym = symbols(new_sym)
        res = solve(Eq(expr, new_sym), solve_sym)
        return res[0]

    def gen_symbol_def_for_inputs(
        self,
        buffer: IndentedBuffer,
        use_sizehint=False,
        shape_str="%s.shape[%s]",
        stride_str="%s.stride()[%s]",
        cvt_str=None,
    ):
        # Process syms in input shapes.
        for sym, nodeinfo in self.input_shape_symbols.items():
            node, mind, sind = nodeinfo
            shape_name_str = f"{node}_shape_{sind}"
            shape_sym = get_tensor_metas(node)[mind].shape[sind]
            free_syms = list(sym.free_symbols)

            if not use_sizehint:
                buffer.writeline(
                    f"{shape_name_str} = {shape_str%(str(node), str(sind))}"
                )
            else:
                hint_size = self.get_shape_size_hint(shape_sym)
                buffer.writeline(f"{shape_name_str} = {hint_size}")
            # Single symbol, write directly.
            if len(free_syms) == 1:
                solve_res = self.solve_result_on_new_symbol(
                    shape_sym, shape_name_str, sym
                )
                buffer.writeline(f"{sym} = {solve_res}")
                if cvt_str:
                    buffer.writeline(f"{sym} = {cvt_str%(str(sym))}")
            elif len(free_syms) > 1:
                buffer.writeline(f"{self.get_valid_expr_str(sym)} = {shape_name_str}")

        # Stride is same as shape process.
        for sym, nodeinfo in self.input_stride_symbols.items():
            if sym in self.input_shape_symbols:
                continue
            free_syms = list(sym.free_symbols)

            node, mind, sind = nodeinfo
            stride_name_str = f"{node}_stride_{sind}"
            stride_sym = get_tensor_metas(node)[mind].stride[sind]
            if not use_sizehint:
                buffer.writeline(
                    f"{stride_name_str} = {stride_str%(str(node), str(sind))}"
                )
            else:
                hint_size = self.get_shape_size_hint(stride_sym)
                buffer.writeline(f"{stride_name_str} = {hint_size}")
            # Single symbol, write directly.
            if len(free_syms) == 1:
                solve_res = self.solve_result_on_new_symbol(
                    stride_sym, stride_name_str, sym
                )
                buffer.writeline(f"{sym} = {solve_res}")
                if cvt_str:
                    buffer.writeline(f"{sym} = {cvt_str%(str(sym))}")
            elif len(free_syms) > 1:
                buffer.writeline(f"{self.get_valid_expr_str(sym)} = {stride_name_str}")

    def get_shape_size_hint(
        self, sha: torch.SymInt, fallback=inductor_config.unbacked_symint_fallback
    ):
        if not is_shape_dynamic(sha):
            return sha
        if target_batch_size:
            return target_batch_size
        exprs = sha.node.expr
        if sha.node.has_hint():
            return sha.node.shape_env.size_hint(exprs)
        syms = list(exprs.free_symbols)
        unbacked_sym_vrs = {
            s: sha.node.shape_env.var_to_range.get(s, None) for s in syms
        }
        if all(vr is not None for vr in unbacked_sym_vrs.values()):
            hint_vr = bound_sympy(exprs, unbacked_sym_vrs)  # type: ignore[arg-type]
            if isinstance(hint_vr.lower, (int, sympy.Integer)):
                fallback = max(fallback, int(hint_vr.lower))
            if isinstance(hint_vr.upper, (int, sympy.Integer)):
                fallback = min(fallback, int(hint_vr.upper))
        return fallback

    def set_tiledims_to_nodes(self, targetnodes_tiledims: dict[Node, set]):
        res = infer_tiledim_back_all(list(targetnodes_tiledims), targetnodes_tiledims)
        assert (
            res is not None
        ), f"infer_tiledim_back_all failed to find tile dims: {targetnodes_tiledims}"
        for node, tiledims in targetnodes_tiledims.items():
            node.meta[TILEDIMNAME] = [list(x) for x in tiledims]

    def clean_tiledims(self, targetnodes_tiledims: dict[Node, set]):
        """
        Clean the tile dimensions from the target nodes.
        This is used to reset the tile dimensions after processing.
        """
        for node in targetnodes_tiledims:
            if TILEDIMNAME in node.meta:
                del node.meta[TILEDIMNAME]

    def get_tiledimsize(self, start_node, output_nodes):
        """
        Get the tile dimension size from the output nodes.
        """
        if start_node is not None:
            start_metas = get_tensor_metas(start_node)
            st_spdims = start_node.meta.get(TILEDIMNAME, None)
            assert (
                st_spdims
            ), f"start node: {start_node} get error tiledims: {st_spdims}"
            for ind, meta in enumerate(start_metas):
                st_shape = meta.shape
                st_spdim = st_spdims[ind]
                return [st_shape[x] for x in st_spdim]
        maxtiledimsize = 0
        for out in output_nodes:
            out_spdims = out.meta.get(TILEDIMNAME, None)
            if out_spdims:
                for out_tile in out_spdims:
                    maxtiledimsize = max(maxtiledimsize, len(out_tile))
        assert maxtiledimsize > 0, "No tile dimension found in output nodes"

        tiledimsize = [-1] * maxtiledimsize
        for out in output_nodes:
            out_metas = get_tensor_metas(out)
            out_spdims = out.meta.get(TILEDIMNAME, None)
            for ind, out_meta in enumerate(out_metas):
                out_shape = out_meta.shape
                if out_spdims:
                    for i, spdim in enumerate(out_spdims[ind]):
                        tiledimsize[i] = max(tiledimsize[i], out_shape[spdim])
        return tiledimsize

    def get_compiled_module(self):
        assert (
            self.module_all is not None
        ), "should call transform first to generate triton module"
        return self.module_all

    def get_input_func(self):
        assert (
            self.input_func is not None
        ), "should call transform first to generate triton module"
        return self.input_func

    def get_in_out_bytes_func(self):
        assert (
            self.in_out_bytes_func is not None
        ), "should call transform first to generate triton module"
        return self.in_out_bytes_func

    def get_diff_func_torch(self):
        assert (
            self.diff_func_torch is not None
        ), "should call transform first to generate triton module"
        return self.diff_func_torch

    def get_diff_func(self):
        assert (
            self.diff_func is not None
        ), "should call transform first to generate triton module"
        return self.diff_func

    def get_diff_test_func(self):
        assert (
            self.diff_test_func is not None
        ), "should call transform first to generate triton module"
        return self.diff_test_func

    def get_perf_func_triton(self):
        assert (
            self.perf_func_triton is not None
        ), "should call transform first to generate triton module"
        return self.perf_func_triton

    def get_perf_func_torch(self):
        assert (
            self.perf_func_torch is not None
        ), "should call transform first to generate triton module"
        return self.perf_func_torch

    # Get perfs in ms funcs.
    def get_perf_test_func_eager(self):
        assert (
            self.perf_get_func_eager is not None
        ), "should call transform first to generate triton module"
        return self.perf_get_func_eager

    def get_perf_test_func_inductor(self):
        assert (
            self.perf_get_func_inductor is not None
        ), "should call transform first to generate triton module"
        return self.perf_get_func_inductor

    def get_perf_test_func_tritonfusion(self):
        assert (
            self.perf_get_func_tritonfusion is not None
        ), "should call transform first to generate triton module"
        return self.perf_get_func_tritonfusion

    def get_perf_test_func_localtriton(self):
        assert (
            self.perf_get_func_localtriton is not None
        ), "should call transform first to generate triton module"
        return self.perf_get_func_localtriton

    def get_perf_test_func_all(self):
        assert (
            self.perf_get_func_all is not None
        ), "should call transform first to generate triton module"
        return self.perf_get_func_all

    # Origin torch func.
    def get_torch_func(self):
        assert (
            self.torch_func is not None
        ), "should call transform first to generate triton module"
        return self.torch_func

    def get_inductor_func(self):
        assert (
            self.inductor_func is not None
        ), "should call transform first to generate inductor_func"
        return self.inductor_func

    def get_triton_wraper_func(self):
        assert (
            self.triton_wraper_func is not None
        ), "should call transform first to generate triton module"
        return self.triton_wraper_func

    def gen_triton_kernel(self):
        """
        Generate the Triton kernel function.
        """
        self.indented_buffer.writeline()
        self.indented_buffer.writeline(
            f"# Triton kernel normal style for {self.graphname}"
        )
        keys_str = f"'{NUMTASKSNAME}'"
        # gen autotuning
        self.indented_buffer.writeline(f"@triton.autotune(")
        self.indented_buffer.writeline(f"configs = [")
        with self.indented_buffer.indent():
            self.indented_buffer.writelines(
                [
                    f"triton.Config({{'{BATCHBLOCKNAME}': 1}}, num_stages=1, num_warps={NUM_WARPS}),",
                    f"triton.Config({{'{BATCHBLOCKNAME}': 1}}, num_stages=3, num_warps={NUM_WARPS}),",
                ]
            )
        self.indented_buffer.writeline(f"],")
        self.indented_buffer.writeline(f"key = [{keys_str}],")
        self.indented_buffer.writeline(
            f"prune_configs_by={{'early_config_prune': triton_fusion_config_prune}},"
        )
        self.indented_buffer.writeline(f")")
        # heuristic for event bs block
        self.indented_buffer.writeline("@triton.heuristics({")
        self.indented_buffer.writeline(
            f"'{EVENBSBLOCKNAME}': lambda args: (args['{NUMTASKSNAME}'] <= get_total_core_num()) or (args['{NUMTASKSNAME}'] % (get_total_core_num() * args['{BATCHBLOCKNAME}']) == 0),"
        )
        self.indented_buffer.writeline("})")
        # triton jit
        self.indented_buffer.writeline(f"@triton.jit")
        self.indented_buffer.writeline(f"def {self.graphname_triton}(")
        for inp in self.input_nodes + self.output_nodes:
            if not is_tensor_node(inp):
                continue
            self.indented_buffer.writeline(f"   {str(inp)}_ptr: tl.tensor,")

        self.indented_buffer.writeline(f"   {NUMTASKSNAME}: int,")
        # Add symbols after numtasks.
        for sym in self.input_shape_symbols:
            self.indented_buffer.writeline(
                f"   {self.get_valid_expr_str_with_inputs(sym)}: int,"
            )
        for sym in self.input_stride_symbols:
            if sym in self.input_shape_symbols:
                continue
            self.indented_buffer.writeline(
                f"   {self.get_valid_expr_str_with_inputs(sym)}: int,"
            )

        self.indented_buffer.writeline(f"   {BATCHBLOCKNAME}: tl.constexpr,")
        self.indented_buffer.writeline(f"   {EVENBSBLOCKNAME}: tl.constexpr,")
        self.indented_buffer.writeline("):")
        with self.indented_buffer.indent():
            # some constant
            self.indented_buffer.writeline("c0 = 0.0")
            self.indented_buffer.writeline("c1 = 1.0")
            # Here we would add the actual Triton kernel logic
            self.indented_buffer.writeline("pid = tl.program_id(0)")
            self.indented_buffer.writeline("num_ctas = tl.num_programs(0)")
            self.indented_buffer.writeline(
                f"num_tiles = tl.cdiv({NUMTASKSNAME}, {BATCHBLOCKNAME})"
            )
            self.indented_buffer.writeline(
                "tiles_per_cta = tl.cdiv(num_tiles, num_ctas)"
            )
            self.indented_buffer.writeline("for j in range(0, tiles_per_cta):")
            with self.indented_buffer.indent():
                self.indented_buffer.writeline("tile_id = pid + num_ctas * j")

                for inp in self.input_nodes + self.output_nodes:
                    if not is_tensor_node(inp):
                        self.indented_buffer.writeline(
                            f"{inp} = {inp.meta[VALMETANAME]}"
                        )
                        continue
                    inpstr = str(inp)
                    inp_tensormeta = get_tensor_metas(inp)[0]
                    inpshape = list(inp_tensormeta.shape)
                    inpstride = inp_tensormeta.stride
                    tiledims = inp.meta.get(TILEDIMNAME, None)
                    checktile = bool(tiledims) and any(
                        [inpshape[x] > 1 for x in tiledims[0]]
                    )
                    # block shape
                    bptr_shape = ", ".join(
                        [self.get_valid_expr_str_with_inputs(i) for i in inpshape]
                    )
                    if not bptr_shape:
                        bptr_shape = "1"
                    # block stride
                    bptr_stride = ", ".join(
                        [self.get_valid_expr_str_with_inputs(i) for i in inpstride]
                    )
                    if not bptr_stride:
                        bptr_stride = "1"
                    # block offsets
                    bptr_offsets = ["0"] * len(inpshape)
                    if checktile:
                        for tiledim in tiledims[0]:
                            if inpshape[tiledim] > 1:
                                bptr_offsets[tiledim] = f"tile_id * {BATCHBLOCKNAME}"
                    bptr_offsets = ", ".join(bptr_offsets)
                    if not bptr_offsets:
                        bptr_offsets = "0"
                    # block output shape
                    bptr_outshape = [
                        f"{self.get_valid_expr_str_with_inputs(i)}" for i in inpshape
                    ]
                    if checktile:
                        for tiledim in tiledims[0]:
                            if inpshape[tiledim] > 1:
                                bptr_outshape[tiledim] = BATCHBLOCKNAME
                    bptr_outshape = ", ".join(bptr_outshape)
                    if not bptr_outshape:
                        bptr_outshape = "1"
                    # block order
                    bptr_order = ", ".join([str(x) for x in stride_order(inpstride)])
                    if not bptr_order:
                        bptr_order = "0"
                    self.indented_buffer.writeline(
                        f"{str(inp)}_bptr = tl.make_block_ptr(base = {str(inp)}_ptr, shape = [{bptr_shape}], strides = [{bptr_stride}], offsets = [{bptr_offsets}], block_shape = [{bptr_outshape}], order = [{bptr_order}])"
                    )
                # load the input tensors
                for inp in self.input_nodes:
                    if not is_tensor_node(inp):
                        continue
                    inpstr = str(inp)
                    inp_tensormeta = get_tensor_metas(inp)[0]
                    inpshape = inp_tensormeta.shape
                    checkdims = inp.meta.get(TILEDIMNAME, None)
                    if checkdims:
                        checkdims = [x for x in checkdims[0] if inpshape[x] > 1]
                    checkdim_str = ""
                    if checkdims:
                        checkdim_str = ",".join([str(x) for x in checkdims]) + (
                            "," if len(checkdims) <= 1 else ""
                        )

                    self.indented_buffer.writeline(
                        f"{inpstr} = tl.load({inpstr}_bptr, boundary_check=({checkdim_str}), padding_option='zero')"
                    )
                # load end
                self.indented_buffer.writeline()
                self.indented_buffer.writeline(
                    "# Add the logic for processing the tensors here"
                )

                for node in self.targetnodes:
                    tritonlines, extralines = convert_to_triton(node)
                    self.indented_buffer.writelines(tritonlines)
                    self.extral_lines.writelines(extralines)

                # store the output tensors
                for out in self.output_nodes:
                    if not is_tensor_node(out):
                        continue
                    outstr = str(out)
                    out_tensormeta = get_tensor_metas(out)[0]
                    outshape = out_tensormeta.shape
                    checkdims = out.meta.get(TILEDIMNAME, None)
                    if checkdims:
                        checkdims = [x for x in checkdims[0] if outshape[x] > 1]
                    checkdim_str = ""
                    if checkdims:
                        checkdim_str = ",".join([str(x) for x in checkdims]) + (
                            "," if len(checkdims) <= 1 else ""
                        )
                    targetdtype = TORCH2TRITON_LOAD_STORE_DTYPE_STR[
                        str(out_tensormeta.dtype)
                    ]
                    self.indented_buffer.writeline(
                        f"tl.store({outstr}_bptr, {outstr}.to({targetdtype}), boundary_check=({checkdim_str}))"
                    )
        self.indented_buffer.writeline()

    def gen_wrapper(self):
        """
        Generate a wrapper function for the Triton kernel.
        """
        ret_anno = (
            f" -> Tuple[{', '.join(['torch.Tensor' for out in self.output_nodes])}]"
            if len(self.output_nodes) > 1
            else " -> torch.Tensor"
        )
        self.indented_buffer.writeline(
            f"@torch.library.custom_op('{TRITONFUSION_CUSTOM_OPS_NAME}::{self.graphname_wrapper_custom}', mutates_args=(), device_types='mlu')"
        )
        self.indented_buffer.writeline(f"def {self.graphname_wrapper_custom}(")
        self.indented_buffer.writeline("    allinputs: List[torch.Tensor]")
        self.indented_buffer.writeline(f"){ret_anno}:")
        with self.indented_buffer.indent():
            self.indented_buffer.writeline(
                f"return {self.graphname_wrapper}(allinputs)"
            )

        self.indented_buffer.writeline(f"def {self.graphname_wrapper}(")
        self.indented_buffer.writeline("    allinputs: List[torch.Tensor]")

        inputargs = ",".join([str(x) for x in self.input_nodes])
        sufix_str = ".to('mlu')"
        allinputs_unpacking = ",".join(
            [
                f"allinputs[{ind}]{'' if not is_tensor_node(inp) else sufix_str}"
                for ind, inp in enumerate(self.input_nodes)
            ]
        )
        self.indented_buffer.writeline(f"){ret_anno}:")
        cachetensorpath = os.path.join(cache_dir_path, self.packagename, self.graphname)
        with self.indented_buffer.indent():
            # save inputs if debug is enabled
            self.indented_buffer.writeline(f"if {TRITONFUSION_SAVE_TENSOR_ENV}:")
            with self.indented_buffer.indent():
                inp_t_path = f"{cachetensorpath}_in.pth"
                self.indented_buffer.writeline(
                    f"logging.info('save {self.graphname} input tensors to : {inp_t_path}')"
                )
                self.indented_buffer.writeline(f"torch.save(allinputs, '{inp_t_path}')")
            self.indented_buffer.writeline(f"{inputargs} = {allinputs_unpacking}")
            # Init dynamic shape.
            self.gen_symbol_def_for_inputs(self.indented_buffer, cvt_str="int(%s)")

            # Output tensor init.
            for out in self.output_nodes:
                out_tensormeta = get_tensor_metas(out)[0]
                # Assuming the output node is a tensor that needs to be created
                dtype = out_tensormeta.dtype
                shape = [
                    self.get_valid_expr_str_with_inputs(x)
                    for x in list(out_tensormeta.shape)
                ]
                shape_str = ", ".join(shape)
                strides = [
                    self.get_valid_expr_str_with_inputs(x)
                    for x in out_tensormeta.stride
                ]
                strides_str = ", ".join(strides)
                device_str = "mlu"
                if isinstance(out.meta.get(VALMETANAME, None), torch.Tensor):
                    device_str = str(out.meta[VALMETANAME].device)
                self.indented_buffer.writeline(
                    f"{str(out)} = torch.empty_strided([{shape_str}], [{strides_str}], dtype={dtype}, device='{device_str}')"
                )

            self.indented_buffer.writeline(
                f"{NUMTASKSNAME} = {self.get_valid_expr_str_with_inputs(self.tiledimsize[0])}"
            )
            self.indented_buffer.writeline(f"num_warps = {NUM_WARPS}")
            self.indented_buffer.writeline(
                f"grid = lambda meta: (min(get_total_core_num() // num_warps, meta['{NUMTASKSNAME}']),)"
            )

            tritonargs = [
                str(node)
                for node in (self.input_nodes + self.output_nodes)
                if is_tensor_node(node)
            ]

            self.indented_buffer.writeline(f"{self.graphname_triton}[grid](")
            for ind, arg in enumerate(tritonargs):
                self.indented_buffer.writeline("    " + arg + ",")
            self.indented_buffer.writeline(f"    {NUMTASKSNAME},")
            # Add symbols after numtasks.
            for sym in self.input_shape_symbols:
                self.indented_buffer.writeline(
                    f"    {self.get_valid_expr_str_with_inputs(sym)},"
                )
            for sym in self.input_stride_symbols:
                if sym in self.input_shape_symbols:
                    continue
                self.indented_buffer.writeline(
                    f"    {self.get_valid_expr_str_with_inputs(sym)},"
                )
            # Other options for compile.
            self.indented_buffer.writeline(
                f"    force_use_shared_memory={FORCE_USE_SM}"
            )
            self.indented_buffer.writeline(")")
            outargsstr = ", ".join([str(out) for out in self.output_nodes])
            # save outputs if debug is enabled
            self.indented_buffer.writeline(f"if {TRITONFUSION_SAVE_TENSOR_ENV}:")
            with self.indented_buffer.indent():
                out_t_path = f"{cachetensorpath}_out.pth"
                self.indented_buffer.writeline(
                    f"logging.info('save {self.graphname} output tensors to : {out_t_path}')"
                )
                self.indented_buffer.writeline(
                    f"torch.save([{outargsstr}], '{out_t_path}')"
                )
            self.indented_buffer.writeline(f"return {outargsstr}")
        self.indented_buffer.writeline("")

        # Fake wrap fn.
        self.indented_buffer.writeline(
            f"@{self.graphname_wrapper_custom}.register_fake"
        )
        self.indented_buffer.writeline(f"def {self.graphname_wrapper_fake_custom}(")
        self.indented_buffer.writeline("    allinputs: List[torch.Tensor]")
        self.indented_buffer.writeline(f"){ret_anno}:")
        with self.indented_buffer.indent():
            self.indented_buffer.writeline(
                f"return {self.graphname_wrapper_fake}(allinputs)"
            )

        self.indented_buffer.writeline(f"def {self.graphname_wrapper_fake}(")
        self.indented_buffer.writeline("    allinputs: List[torch.Tensor]")
        self.indented_buffer.writeline(f"){ret_anno}:")
        with self.indented_buffer.indent():
            inputargs = ",".join([str(x) for x in self.input_nodes])
            sufix_str = ".to('mlu')"
            allinputs_unpacking = ",".join(
                [
                    f"allinputs[{ind}]{'' if not is_tensor_node(inp) else sufix_str}"
                    for ind, inp in enumerate(self.input_nodes)
                ]
            )
            self.indented_buffer.writeline(f"{inputargs} = {allinputs_unpacking}")
            self.gen_symbol_def_for_inputs(self.indented_buffer, cvt_str="int(%s)")
            for out in self.output_nodes:
                out_tensormeta = get_tensor_metas(out)[0]
                # Assuming the output node is a tensor that needs to be created.
                dtype = out_tensormeta.dtype
                shape = [
                    self.get_valid_expr_str_with_inputs(x) for x in out_tensormeta.shape
                ]
                shape_str = ", ".join(shape)
                strides = [
                    self.get_valid_expr_str_with_inputs(x)
                    for x in out_tensormeta.stride
                ]
                strides_str = ", ".join(strides)
                self.indented_buffer.writeline(
                    f"{str(out)} = torch.empty_strided([{shape_str}], [{strides_str}], dtype={dtype}, device='mlu')"
                )
            self.indented_buffer.writeline(
                f"return {', '.join([str(out) for out in self.output_nodes])}"
            )
        self.indented_buffer.writeline("")

    def gen_import(self):
        """
        Generate import statements for Triton.
        """
        self.indented_buffer.writeline("# Generated Triton code from PyTorch FX nodes")
        self.indented_buffer.writeline("import sys, os, logging")
        self.indented_buffer.writeline("import torch, sympy")
        self.indented_buffer.writeline("from torch import nn")
        self.indented_buffer.writeline(
            "from torch._inductor.select_algorithm import autotune_select_algorithm, \
            TritonTemplate, realize_inputs"
        )
        self.indented_buffer.writeline("from torch._inductor.ir import FixedLayout")
        self.indented_buffer.writeline(
            f"from torch._inductor.lowering import empty_strided"
        )
        self.indented_buffer.writeline(
            "from torch._inductor.ir import FixedLayout, FlexibleLayout"
        )
        self.indented_buffer.writeline(
            "import torch._inductor.config as inductor_config"
        )
        self.indented_buffer.writeline("import triton")
        self.indented_buffer.writeline(
            "from torch_mlu._inductor.fx_passes.triton_fusion import torch_compile_without_cache,\
            get_triton_inductor_config, triton_fusion_config_prune, get_triton_inductor_grid,\
            get_tensor_strided, get_triton_inductor_grid_fn, get_externkernelchoice, get_total_core_num"
        )
        self.indented_buffer.writeline(
            "from torch_mlu._inductor.fx_passes.triton_fusion import config as tt_config"
        )
        self.indented_buffer.writeline(
            "from torch_mlu._inductor import config as torch_mlu_config"
        )
        self.indented_buffer.writeline("from typing import Tuple, List")
        self.indented_buffer.writeline("import triton.language as tl")
        self.indented_buffer.writeline("")

        self.indented_buffer.writeline("torch.mlu.manual_seed_all(17)")
        self.indented_buffer.writeline("torch.set_printoptions(precision=8)")
        self.indented_buffer.writeline(
            f"{TRITONFUSIONDEBUGNAME} = os.environ.get('{TRITONFUSIONDEBUGNAME}', '0') == '1'"
        )
        self.indented_buffer.writeline(
            f"{TRITONFUSION_SAVE_TENSOR_ENV} = os.environ.get('{TRITONFUSION_SAVE_TENSOR_ENV}', '0') == '1'"
        )
        self.indented_buffer.writeline(
            f"{TRITONFUSION_LOAD_TENSOR_ENV} = os.environ.get('{TRITONFUSION_LOAD_TENSOR_ENV}', '0') == '1'"
        )
        self.indented_buffer.writeline()

    def gen_module(self):
        _cache_dir = cache_dir_path
        package_cache_dir = os.path.join(_cache_dir, self.packagename)
        os.makedirs(package_cache_dir, exist_ok=True)
        logger.info(
            f"new triton fusion graph {self.graphname} save to: {package_cache_dir}"
        )
        initpath = os.path.join(package_cache_dir, "__init__.py")
        if not os.path.exists(initpath):
            with open(initpath, "wt", encoding="utf-8") as f1:
                f1.write("# Generated Triton code package\n")
        fpath = os.path.join(package_cache_dir, f"{self.graphname}.py")
        with open(fpath, "wt", encoding="utf-8") as f:
            f.write(self.indented_buffer.getvalue())
            f.write(self.extral_lines.getvalue())

        spec = importlib.util.spec_from_file_location(
            f"_gen_module_{self.graphname}_pid_{os.getpid()}",
            f.name,
        )
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)

        self.module_all = m
        self.triton_wraper_func = getattr(m, self.graphname_wrapper)
        self.diff_func_torch = getattr(m, self.graphname_diff_torch)
        self.diff_func = getattr(m, self.graphname_diff)
        self.diff_test_func = getattr(m, self.graphname_testdiff)

        self.perf_func_triton = getattr(m, self.graphname_perf_triton)
        self.perf_func_torch = getattr(m, self.graphname_perf_torch)

        # All perf get funcs.
        self.perf_get_func_eager = getattr(m, self.graphname_getperf_eager)
        self.perf_get_func_localtriton = getattr(m, self.graphname_getperf_localtriton)
        self.perf_get_func_inductor = getattr(m, self.graphname_getperf_inductor)
        self.perf_get_func_tritonfusion = getattr(
            m, self.graphname_getperf_tritonfusion
        )
        self.perf_get_func_all = getattr(m, self.graphname_getperf_all)

        self.input_func = getattr(m, self.graphname_getinputs)
        self.in_out_bytes_func = getattr(m, self.graphname_getinoutbytes)
        self.torch_func = getattr(m, self.graphname_torchfunc)

        self.inductor_func = getattr(m, self.graphname_tuned_inductor)

    def format_torch_arg(self, x):
        if isinstance(x, str):
            return f"'{x}'"
        elif isinstance(x, torch.device):
            return f"'{x.type}'"
        return str(x)

    def gen_tests(self):
        """
        Generate diff and perf test funcs.
        """
        self.extral_lines.writeline("")
        self.extral_lines.writeline("# tests for diff and perf")

        # gen get inputs
        inputargs = ", ".join([str(x) for x in self.input_nodes])
        allinputs_unpacking = ",".join(
            [f"allinputs[{ind}]" for ind in range(len(self.input_nodes))]
        )
        self.extral_lines.writeline(f"def {self.graphname_getinputs}():")
        with self.extral_lines.indent():
            self.extral_lines.writeline(f"if {TRITONFUSION_LOAD_TENSOR_ENV}:")
            with self.extral_lines.indent():
                input_pt_name = f"{self.graphname}_in.pth"
                self.extral_lines.writeline(
                    f"logging.info('{self.graphname} get_input using saved tensors: {input_pt_name}')"
                )
                self.extral_lines.writeline(
                    "current_path = os.path.dirname(os.path.abspath(__file__))"
                )
                self.extral_lines.writeline(
                    f"return torch.load(os.path.join(current_path, '{input_pt_name}'))"
                )

            self.gen_symbol_def_for_inputs(
                self.extral_lines, use_sizehint=True, cvt_str="int(%s)"
            )

            # Gen input tensors.
            for node in self.input_nodes:
                if not is_tensor_node(node):
                    self.extral_lines.writeline(
                        f"{str(node)} = torch.tensor({node.meta[VALMETANAME]}, device='{DEVICENAME}')"
                    )
                else:
                    node_tensormeta = get_tensor_metas(node)[0]
                    node_shape = list(node_tensormeta.shape)
                    node_stride = list(node_tensormeta.stride)
                    dtype = node_tensormeta.dtype
                    shape = [self.get_valid_expr_str_with_inputs(x) for x in node_shape]
                    shape_str = ", ".join(shape)
                    symbol_dims = [
                        ind for ind, x in enumerate(node_shape) if is_shape_dynamic(x)
                    ]
                    stride = [
                        self.get_valid_expr_str_with_inputs(x) for x in node_stride
                    ]
                    stride_str = ", ".join(stride)
                    self.extral_lines.writeline(
                        f"{str(node)} = get_tensor_strided([{shape_str}], [{stride_str}], {dtype}, 'mlu')"
                    )
                    for symdim in symbol_dims:
                        self.extral_lines.writeline(
                            f"torch._dynamo.mark_dynamic({node}, {symdim})"
                        )
            self.extral_lines.writeline(f"return [{inputargs}]")
        self.extral_lines.writeline("")
        # Gen in out bytes func.
        self.extral_lines.writeline(
            f"def {self.graphname_getinoutbytes}(allinputs: List[torch.Tensor]):"
        )
        with self.extral_lines.indent():
            self.extral_lines.writeline(f"cnt_in = cnt_out = 0")
            self.extral_lines.writeline(f"for t in allinputs:")
            with self.extral_lines.indent():
                self.extral_lines.writeline(
                    f"if not isinstance(t, torch.Tensor): continue"
                )
                self.extral_lines.writeline(f"cnt_in += t.numel() * t.element_size()")
            self.extral_lines.writeline(
                f"result = {self.graphname_wrapper_fake}(allinputs)"
            )
            self.extral_lines.writeline(
                f"if not isinstance(result, (list, tuple)): result = [result]"
            )
            self.extral_lines.writeline(f"for t in result:")
            with self.extral_lines.indent():
                self.extral_lines.writeline(
                    f"if not isinstance(t, torch.Tensor): continue"
                )
                self.extral_lines.writeline(f"cnt_out += t.numel() * t.element_size()")
            self.extral_lines.writeline("return cnt_in, cnt_out")
        self.extral_lines.writeline("")
        # gen torch compute
        ret_anno = (
            f" -> Tuple[{', '.join(['torch.Tensor' for out in self.output_nodes])}]"
            if len(self.output_nodes) > 1
            else " -> torch.Tensor"
        )
        self.extral_lines.writeline(
            f"def {self.graphname_torchfunc}(allinputs: List[torch.Tensor]){ret_anno}:"
        )
        with self.extral_lines.indent():
            self.extral_lines.writeline(f"{inputargs} = {allinputs_unpacking}")
            # For dynamic shape arg, use tensor's shape but not static shape.
            self.gen_symbol_def_for_inputs(self.extral_lines, cvt_str="int(%s)")
            for node in self.input_nodes:
                if not is_tensor_node(node):
                    self.extral_lines.writeline(
                        f"{node} = {self.get_valid_expr_str_with_inputs(node.meta[VALMETANAME])}"
                    )

            # Gen assert to constrain dynamic sizes and strides.
            kep_symstr_node_shape = {}
            for inp in self.input_nodes:
                if not is_tensor_node(inp):
                    continue
                inp_tensormeta = get_tensor_metas(inp)[0]
                shape = list(inp_tensormeta.shape)
                node_stride = list(inp_tensormeta.stride)
                for ind, sha in enumerate(shape):
                    if not is_shape_dynamic(sha):
                        continue
                    sha_str = str(sha)
                    if sha_str not in kep_symstr_node_shape:
                        kep_symstr_node_shape[sha_str] = []
                    kep_symstr_node_shape[sha_str].append([inp, ind])

                    if not is_shape_dynamic(node_stride[ind]):
                        if ind < len(shape) - 1:
                            if (
                                node_stride[ind]
                                != node_stride[ind + 1] * shape[ind + 1]
                            ):
                                self.extral_lines.writeline(
                                    f"assert {inp}.stride()[{ind}] == {node_stride[ind]}"
                                )
                        elif node_stride[ind] != 1:
                            self.extral_lines.writeline(
                                f"assert {inp}.stride()[{ind}] == {node_stride[ind]}"
                            )
            for sha, nodes in kep_symstr_node_shape.items():
                if len(nodes) <= 1:
                    continue
                ass_str = " == ".join([f"{nod}.shape[{ind}]" for nod, ind in nodes])
                self.extral_lines.writeline(f"assert {ass_str}")

            # Generate torch func ops.
            for node in self.targetnodes:
                if node.target == operator.getitem:
                    self.extral_lines.writeline(
                        f"{node} = {node.args[0]}[{node.args[1]}]"
                    )
                else:
                    nodeargs = ", ".join([self.format_torch_arg(x) for x in node.args])
                    nodekwargs = ", ".join(
                        [
                            f"{k}={self.format_torch_arg(v)}"
                            for k, v in node.kwargs.items()
                        ]
                    )
                    self.extral_lines.writeline(
                        f"{str(node)} = torch.ops.{get_target_name(node)}({nodeargs}{',' if nodekwargs else ''}{nodekwargs})"
                    )
            self.extral_lines.writeline(
                f"return {', '.join([str(out) for out in self.output_nodes])}"
            )

        self.extral_lines.writeline("")
        # Gen test diff.
        self.extral_lines.writeline(
            f"def {self.graphname_diff_torch}(allinputs: List[torch.Tensor], compiled = False, withtritonfusion = False):"
        )
        with self.extral_lines.indent():
            self.extral_lines.writeline(
                f"kepenv = torch_mlu_config.enable_triton_fusion"
            )
            self.extral_lines.writeline(
                "kep_cache_env = inductor_config.force_disable_caches"
            )
            self.extral_lines.writeline(
                f"torch_mlu_config.enable_triton_fusion = withtritonfusion"
            )
            self.extral_lines.writeline(f"torch_perf_func = {self.graphname_torchfunc}")
            self.extral_lines.writeline(f"if compiled:")
            with self.extral_lines.indent():
                self.extral_lines.writeline(
                    "inductor_config.force_disable_caches = True"
                )
                self.extral_lines.writeline(
                    f"torch_perf_func = torch.compile(torch_perf_func)"
                )
                self.extral_lines.writeline("torch.compiler.reset()")

            self.extral_lines.writeline(f"result = torch_perf_func(allinputs)")
            self.extral_lines.writeline(
                f"torch_mlu_config.enable_triton_fusion = kepenv"
            )
            self.extral_lines.writeline(
                "inductor_config.force_disable_caches = kep_cache_env"
            )
            self.extral_lines.writeline(f"if not isinstance(result, (list, tuple)):")
            with self.extral_lines.indent():
                self.extral_lines.writeline(f"result = [result]")
            self.extral_lines.writeline(f"return result")
        self.extral_lines.writeline("")
        # Gen diff test.
        self.extral_lines.writeline(
            f"def {self.graphname_diff}(allinputs: List[torch.Tensor]):"
        )
        with self.extral_lines.indent():
            self.extral_lines.writeline(
                "diff_test_local_triton = os.environ.get('TRITONFUSION_DIFF_TEST_LOCALTRITON', '0') == '1'"
            )
            self.extral_lines.writeline(
                "diff_test_eager = os.environ.get('TRITONFUSION_DIFF_TEST_EAGER', '1') == '1'"
            )
            self.extral_lines.writeline(
                "diff_test_inductor = os.environ.get('TRITONFUSION_DIFF_TEST_INDUCTOR', '0') == '1'"
            )
            self.extral_lines.writeline(
                "diff_test_atol_val = float(os.environ.get('TRITONFUSION_DIFF_TEST_ATOL', 0.002))"
            )
            self.extral_lines.writeline(
                "diff_test_rtol_val = float(os.environ.get('TRITONFUSION_DIFF_TEST_RTOL', 0.001))"
            )
            self.extral_lines.writeline(
                "diff_test_equal_nan = os.environ.get('TRITONFUSION_DIFF_TEST_EQUAL_NAN', '1') == '1'"
            )
            # Torch res.
            self.extral_lines.writeline(
                f"torch_allresults = {self.graphname_diff_torch}(allinputs, False)"
            )
            self.extral_lines.writeline(f"if diff_test_local_triton:")
            with self.extral_lines.indent():
                self.extral_lines.writeline(
                    "kep_ck_env = tt_config.pre_check_triton_kernel"
                )
                self.extral_lines.writeline("tt_config.pre_check_triton_kernel = False")
                # Localtriton res.
                self.extral_lines.writeline(
                    f"triton_allresults = {self.graphname_wrapper}(allinputs)"
                )
                self.extral_lines.writeline(
                    "tt_config.pre_check_triton_kernel = kep_ck_env"
                )
                self.extral_lines.writeline(
                    f"if not isinstance(triton_allresults, (list, tuple)):"
                )
                with self.extral_lines.indent():
                    self.extral_lines.writeline(
                        f"triton_allresults = [triton_allresults]"
                    )

                # Compare.
                self.extral_lines.writeline(f"# Test for localtriton and eager diff.")
                self.extral_lines.writeline(
                    "assert len(triton_allresults) == len(torch_allresults), \
                    'get different length of torch and localtriton results'"
                )
                self.extral_lines.writeline(
                    f"for ind, (trires, torres) in enumerate(zip(triton_allresults, torch_allresults)):"
                )
                with self.extral_lines.indent():
                    self.extral_lines.writeline(
                        f"print (f'begin testing diff of local triton and torch eager result {{ind}}: ', trires.shape, trires.dtype)"
                    )
                    self.extral_lines.writeline(
                        f"torch.testing.assert_close(trires, torres, atol = diff_test_atol_val, rtol= diff_test_rtol_val, equal_nan=diff_test_equal_nan)"
                    )
                self.extral_lines.writeline(
                    "print ('All Diff Tests Done: LocalTriton and Eager DIFF PASSED')"
                )
            # TritonFusion diff test.
            self.extral_lines.writeline(f"# Test for tritonfusion and eager diff.")
            self.extral_lines.writeline(
                f"tritonfusion_allresults = {self.graphname_diff_torch}(allinputs, True, True)"
            )
            # Test diff eager.
            self.extral_lines.writeline(f"if diff_test_eager:")
            with self.extral_lines.indent():
                self.extral_lines.writeline(
                    "assert len(tritonfusion_allresults) == len(torch_allresults), \
                    'get different length of torch and tritonfusion results'"
                )
                self.extral_lines.writeline(
                    f"for ind, (trires, torres) in enumerate(zip(tritonfusion_allresults, torch_allresults)):"
                )
                with self.extral_lines.indent():
                    self.extral_lines.writeline(
                        f"print (f'begin testing diff of tritonfusion and torch eager result {{ind}}: ', trires.shape, trires.dtype)"
                    )
                    self.extral_lines.writeline(
                        f"torch.testing.assert_close(trires, torres, atol = diff_test_atol_val, rtol= diff_test_rtol_val, equal_nan=diff_test_equal_nan)"
                    )
                self.extral_lines.writeline(
                    "print ('All Diff Tests Done: TritonFusion and Eager DIFF PASSED')"
                )
            # Test diff inductor.
            self.extral_lines.writeline(f"if diff_test_inductor:")
            with self.extral_lines.indent():
                self.extral_lines.writeline(
                    f"inductor_allresults = {self.graphname_diff_torch}(allinputs, True, False)"
                )
                self.extral_lines.writeline(
                    "assert len(inductor_allresults) == len(tritonfusion_allresults), \
                    'get different length of inductor and tritonfusion results'"
                )
                self.extral_lines.writeline(
                    f"for ind, (trires, torres) in enumerate(zip(tritonfusion_allresults, inductor_allresults)):"
                )
                with self.extral_lines.indent():
                    self.extral_lines.writeline(
                        f"print (f'begin testing diff of tritonfusion and inductor result {{ind}}: ', trires.shape, trires.dtype)"
                    )
                    self.extral_lines.writeline(
                        f"torch.testing.assert_close(trires, torres, atol = diff_test_atol_val, rtol= diff_test_rtol_val, equal_nan=diff_test_equal_nan)"
                    )
                self.extral_lines.writeline(
                    "print ('All Diff Tests Done: TritonFusion and Inductor DIFF PASSED')"
                )
            self.extral_lines.writeline("return True")

        self.extral_lines.writeline("# pytest interface")
        self.extral_lines.writeline(f"def {self.graphname_testdiff}():")
        with self.extral_lines.indent():
            self.extral_lines.writeline(f"allinputs = {self.graphname_getinputs}()")
            self.extral_lines.writeline(f"{self.graphname_diff}(allinputs)")

        self.extral_lines.writeline("")

        # Gen perf tests
        # Triton perf test base func.
        self.extral_lines.writeline(
            f"def {self.graphname_perf_triton}(allinputs: List[torch.Tensor]):"
        )
        with self.extral_lines.indent():
            self.extral_lines.writeline(
                f"triton_lambda = lambda: {self.graphname_wrapper}(allinputs)"
            )
            self.extral_lines.writeline(
                f"tritonms = triton.testing.do_bench(triton_lambda, warmup=10, rep=100)"
            )
            self.extral_lines.writeline(
                f"# print ('perf test triton get triton time(ms): ', tritonms)"
            )
            self.extral_lines.writeline("return tritonms")

        # Eager/Inductor/Tritonfusion perf test base func.
        self.extral_lines.writeline(
            f"def {self.graphname_perf_torch}(allinputs: List[torch.Tensor], compiled = False, withtritonfusion = False):"
        )
        with self.extral_lines.indent():
            self.extral_lines.writeline(
                f"kepenv = torch_mlu_config.enable_triton_fusion"
            )
            self.extral_lines.writeline(
                "kep_cache_env = inductor_config.force_disable_caches"
            )
            self.extral_lines.writeline(
                f"torch_mlu_config.enable_triton_fusion = withtritonfusion"
            )
            self.extral_lines.writeline(f"torch_perf_func = {self.graphname_torchfunc}")
            self.extral_lines.writeline(f"if compiled:")
            with self.extral_lines.indent():
                self.extral_lines.writeline(
                    "inductor_config.force_disable_caches = True"
                )
                self.extral_lines.writeline(
                    f"torch_perf_func = torch.compile(torch_perf_func)"
                )
                self.extral_lines.writeline("torch.compiler.reset()")

            self.extral_lines.writeline(
                f"torch_lambda = lambda: torch_perf_func(allinputs)"
            )
            self.extral_lines.writeline(
                f"torchms = triton.testing.do_bench(torch_lambda, warmup=10, rep=100)"
            )
            self.extral_lines.writeline(
                f"torch_mlu_config.enable_triton_fusion = kepenv"
            )
            self.extral_lines.writeline(
                "inductor_config.force_disable_caches = kep_cache_env"
            )
            self.extral_lines.writeline(
                f"# print (f'perf test: compiled:{{compiled}}  torch get torch time(ms): ', torchms)"
            )
            self.extral_lines.writeline(f"return torchms")

        self.extral_lines.writeline("\n")
        self.extral_lines.writeline("# get eager perf interface")
        # Get eager perf in ms.
        self.extral_lines.writeline(
            f"def {self.graphname_getperf_eager}(allinputs: List[torch.Tensor]):"
        )
        with self.extral_lines.indent():
            self.extral_lines.writeline(
                f"torchms = {self.graphname_perf_torch}(allinputs, False)"
            )
            self.extral_lines.writeline(f"return torchms")
        self.extral_lines.writeline("# pytest interface")
        self.extral_lines.writeline(f"def {self.graphname_testperf_eager}():")
        with self.extral_lines.indent():
            self.extral_lines.writeline(f"allinputs = {self.graphname_getinputs}()")
            self.extral_lines.writeline(
                f"toms = {self.graphname_getperf_eager}(allinputs)"
            )
            self.extral_lines.writeline(
                f"cnt_in, cnt_out = {self.graphname_getinoutbytes}(allinputs)"
            )
            self.extral_lines.writeline(
                f"print (f'\\ngraph: {self.graphname} perf get result times: torch(ms): {{toms}}   InBytes: {{cnt_in}}   OutBytes: {{cnt_out}}   BandWidth(Gb/s): {{(cnt_in+cnt_out)/toms*1e-6}}')"
            )

        # Get inductor perf in ms.
        self.extral_lines.writeline("\n")
        self.extral_lines.writeline("# get perf interface inductor")
        self.extral_lines.writeline(
            f"def {self.graphname_getperf_inductor}(allinputs: List[torch.Tensor]):"
        )
        with self.extral_lines.indent():
            self.extral_lines.writeline(
                f"torchms = {self.graphname_perf_torch}(allinputs, True, False)"
            )
            self.extral_lines.writeline(f"return torchms")
        self.extral_lines.writeline("# pytest interface inductor")
        self.extral_lines.writeline(f"def {self.graphname_testperf_inductor}():")
        with self.extral_lines.indent():
            self.extral_lines.writeline(f"allinputs = {self.graphname_getinputs}()")
            self.extral_lines.writeline(
                f"toms = {self.graphname_getperf_inductor}(allinputs)"
            )
            self.extral_lines.writeline(
                f"cnt_in, cnt_out = {self.graphname_getinoutbytes}(allinputs)"
            )
            self.extral_lines.writeline(
                f"print (f'\\ngraph: {self.graphname} perf inductor get result times: torch_inductor(ms): {{toms}}   InBytes: {{cnt_in}}   OutBytes: {{cnt_out}}   BandWidth(Gb/s): {{(cnt_in+cnt_out)/toms*1e-6}}')"
            )

        # Get tritonfusion perf in ms.
        self.extral_lines.writeline("\n")
        self.extral_lines.writeline("# get perf interface tritonfusion")
        self.extral_lines.writeline(
            f"def {self.graphname_getperf_tritonfusion}(allinputs: List[torch.Tensor]):"
        )
        with self.extral_lines.indent():
            self.extral_lines.writeline(
                f"tritonfusion_ms = {self.graphname_perf_torch}(allinputs, True, True)"
            )
            self.extral_lines.writeline(f"return tritonfusion_ms")
        self.extral_lines.writeline("# pytest interface tritonfusion")
        self.extral_lines.writeline(f"def {self.graphname_testperf_tritonfusion}():")
        with self.extral_lines.indent():
            self.extral_lines.writeline(f"allinputs = {self.graphname_getinputs}()")
            self.extral_lines.writeline(
                f"tritonfusion_ms = {self.graphname_getperf_tritonfusion}(allinputs)"
            )
            self.extral_lines.writeline(
                f"cnt_in, cnt_out = {self.graphname_getinoutbytes}(allinputs)"
            )
            self.extral_lines.writeline(
                f"print (f'\\ngraph: {self.graphname} perf inductor get result times: torch_tritonfusion(ms): {{tritonfusion_ms}}   InBytes: {{cnt_in}}   OutBytes: {{cnt_out}}   BandWidth(Gb/s): {{(cnt_in+cnt_out)/tritonfusion_ms*1e-6}}')"
            )

        # Get localtriton perf in ms.
        self.extral_lines.writeline("\n")
        self.extral_lines.writeline("# get perf interface: localtriton")
        self.extral_lines.writeline(
            f"def {self.graphname_getperf_localtriton}(allinputs: List[torch.Tensor]):"
        )
        with self.extral_lines.indent():
            self.extral_lines.writeline(
                "kep_ck_env = tt_config.pre_check_triton_kernel"
            )
            self.extral_lines.writeline("tt_config.pre_check_triton_kernel = False")
            self.extral_lines.writeline(
                f"localtriton_ms = {self.graphname_perf_triton}(allinputs)"
            )
            self.extral_lines.writeline(
                "tt_config.pre_check_triton_kernel = kep_ck_env"
            )
            self.extral_lines.writeline(f"return localtriton_ms")
        self.extral_lines.writeline("# pytest interface localtriton")
        self.extral_lines.writeline(f"def {self.graphname_testperf_localtriton}():")
        with self.extral_lines.indent():
            self.extral_lines.writeline(f"allinputs = {self.graphname_getinputs}()")
            self.extral_lines.writeline(
                f"localtriton_ms = {self.graphname_getperf_localtriton}(allinputs)"
            )
            self.extral_lines.writeline(
                f"cnt_in, cnt_out = {self.graphname_getinoutbytes}(allinputs)"
            )
            self.extral_lines.writeline(
                f"print (f'\\ngraph: {self.graphname} perf inductor get result times: localtriton(ms): {{localtriton_ms}}   InBytes: {{cnt_in}}   OutBytes: {{cnt_out}}   BandWidth(Gb/s): {{(cnt_in+cnt_out)/localtriton_ms*1e-6}}')"
            )

        # Get all perf in ms.
        self.extral_lines.writeline("\n")
        self.extral_lines.writeline("# get perf interface for perf all")
        self.extral_lines.writeline(
            f"def {self.graphname_getperf_all}(allinputs: List[torch.Tensor]):"
        )
        with self.extral_lines.indent():
            self.extral_lines.writeline(
                f"torchms_inductor = {self.graphname_getperf_inductor}(allinputs)"
            )
            self.extral_lines.writeline(
                f"torchms_eager = {self.graphname_getperf_eager}(allinputs)"
            )
            self.extral_lines.writeline(
                f"torchms_tritonfusion = {self.graphname_getperf_tritonfusion}(allinputs)"
            )
            self.extral_lines.writeline(
                f"tritonms = {self.graphname_getperf_localtriton}(allinputs)"
            )
            self.extral_lines.writeline(
                f"return torchms_inductor, torchms_eager, torchms_tritonfusion, tritonms"
            )
        self.extral_lines.writeline("# pytest interface for perf all")
        self.extral_lines.writeline(f"def {self.graphname_testperf_all}():")
        with self.extral_lines.indent():
            self.extral_lines.writeline(f"allinputs = {self.graphname_getinputs}()")
            self.extral_lines.writeline(
                f"torchms_inductor, torchms_eager, torchms_tritonfusion, tritonms = {self.graphname_getperf_all}(allinputs)"
            )
            self.extral_lines.writeline(
                f"cnt_in, cnt_out = {self.graphname_getinoutbytes}(allinputs)"
            )
            self.extral_lines.writeline(
                f"print (f'\\n\\ngraph: {self.graphname} perf_all get InBytes: {{cnt_in}} OutBytes: {{cnt_out}} get result times and bws:\\n\
                torch_inductor(ms): {{torchms_inductor}}   BandWidth(Gb/s): {{(cnt_in+cnt_out)/torchms_inductor*1e-6}}\\n\
                torch_eager(ms): {{torchms_eager}}   BandWidth(Gb/s): {{(cnt_in+cnt_out)/torchms_eager*1e-6}}\\n\
                torch_tritonfusion(ms): {{torchms_tritonfusion}}   BandWidth(Gb/s): {{(cnt_in+cnt_out)/torchms_tritonfusion*1e-6}}\\n\
                triton(ms): {{tritonms}}   BandWidth(Gb/s): {{(cnt_in+cnt_out)/tritonms*1e-6}}\\n')"
            )
            self.extral_lines.writeline(
                f"print (f'Triton perf diff(TritonFusion - LocalTriton)(ms): {{torchms_tritonfusion - tritonms}} {{(torchms_tritonfusion - tritonms)/tritonms*100}}%')"
            )
            self.extral_lines.writeline(
                f"print (f'Speedup TritonFusion_Eager(ms): {{torchms_eager - torchms_tritonfusion}}')"
            )
            self.extral_lines.writeline(
                f"print (f'Speedup LocalTriton_Eager(ms): {{torchms_eager - tritonms}}')"
            )
            self.extral_lines.writeline(
                f"print (f'Speedup TritonFusion_Inductor(ms): {{torchms_inductor - torchms_tritonfusion}}')"
            )
            self.extral_lines.writeline(
                f"print (f'Speedup LocalTriton_Inductor(ms): {{torchms_inductor - tritonms}}')"
            )

    def gen_inductor_tritontemplate(self):
        """
        Generate the Inductor Triton kernel template.
        """
        extralines_kep = []
        self.indented_buffer.writeline(
            f"# Triton inductor kernel for {self.graphname} with {len(self.input_nodes)} inputs and {len(self.output_nodes)} outputs"
        )
        self.indented_buffer.writeline(f'{self.graphname_tritontemplate} = r"""')
        # first gen def kernel
        input_args = []
        for inp in self.input_nodes + self.output_nodes[1:]:
            if not is_tensor_node(inp):
                continue
            input_args.append(f'"{str(inp)}_ptr"')
        input_args_str = ", ".join(input_args)
        self.indented_buffer.writeline(f"{{{{def_kernel({input_args_str})}}}}")
        with self.indented_buffer.indent():
            # Init dynamic shapes symbols.
            self.gen_symbol_def_for_inputs(
                self.indented_buffer,
                shape_str="{{size('%s_ptr', %s)}}",
                stride_str="{{stride('%s_ptr', %s)}}",
                cvt_str="%s.to(tl.int32)",
            )
            self.indented_buffer.writeline(
                f"{NUMTASKSNAME} = {self.get_valid_expr_str_with_inputs(self.tiledimsize[0])}"
            )
            # some constant
            if test_fallback_kernel:
                self.indented_buffer.writeline(
                    "-MAKE_KERNEL_ERROR_TO_TEST_FALLBACK = 0"
                )
            self.indented_buffer.writeline("c0 = 0.0")
            self.indented_buffer.writeline("c1 = 1.0")
            # Here we would add the actual Triton kernel logic
            self.indented_buffer.writeline("pid = tl.program_id(0)")
            self.indented_buffer.writeline("num_ctas = tl.num_programs(0)")
            self.indented_buffer.writeline(
                f"num_tiles = tl.cdiv({NUMTASKSNAME}, {BATCHBLOCKNAME})"
            )
            self.indented_buffer.writeline(
                "tiles_per_cta = tl.cdiv(num_tiles, num_ctas)"
            )
            self.indented_buffer.writeline("for j in range(0, tiles_per_cta):")
            with self.indented_buffer.indent():
                self.indented_buffer.writeline("tile_id = pid + num_ctas * j")

                blockptr_nodes = []
                for inp in self.input_nodes + self.output_nodes[1:]:
                    if not is_tensor_node(inp):
                        continue
                    inpstr = str(inp)
                    inp_tensormeta = get_tensor_metas(inp)[0]
                    inpshape = inp_tensormeta.shape
                    inpstride = inp_tensormeta.stride
                    tiledims = inp.meta.get(TILEDIMNAME, None)
                    checktile = bool(tiledims) and any(
                        inpshape[x] > 1 for x in tiledims[0]
                    )
                    # Inductor will change placeholder scalar node to scalar, so we don't need to gen block ptr for it.
                    if (
                        not inpshape
                        and (VALMETANAME in inp.meta)
                        and (inp.meta[VALMETANAME].fake_device.type == "cpu")
                    ):
                        self.indented_buffer.writeline(f"{inpstr} = {inpstr}_ptr")
                        continue

                    # Block shape.
                    bptr_shape_list = []
                    for ind, sha in enumerate(inpshape):
                        if is_shape_dynamic(sha):
                            bptr_shape_list.append(
                                f"{self.get_valid_expr_str_with_inputs(sha)}"
                            )
                        else:
                            bptr_shape_list.append(f'{{{{size("{inp}_ptr", {ind})}}}}')
                    bptr_shape = ", ".join(bptr_shape_list)
                    if not bptr_shape:
                        bptr_shape = "1"

                    # Block stride.
                    bptr_stride = []
                    for ind, sha in enumerate(inpstride):
                        if is_shape_dynamic(sha):
                            bptr_stride.append(
                                f"{self.get_valid_expr_str_with_inputs(sha)}"
                            )
                        else:
                            bptr_stride.append(f'{{{{stride("{inp}_ptr", {ind})}}}}')
                    bptr_stride = ", ".join(bptr_stride)
                    if not bptr_stride:
                        bptr_stride = "1"

                    # Block offsets.
                    bptr_offsets = ["0"] * len(inpshape)
                    if checktile:
                        for tiledim in tiledims[0]:
                            if inpshape[tiledim] > 1:
                                bptr_offsets[tiledim] = f"tile_id * {BATCHBLOCKNAME}"
                    bptr_offsets = ", ".join(bptr_offsets)
                    if not bptr_offsets:
                        bptr_offsets = "0"

                    # Block output shape.
                    bptr_outshape = list(bptr_shape_list)
                    if checktile:
                        for tiledim in tiledims[0]:
                            if inpshape[tiledim] > 1:
                                bptr_outshape[tiledim] = BATCHBLOCKNAME
                    bptr_outshape = ", ".join(bptr_outshape)
                    if not bptr_outshape:
                        bptr_outshape = "1"

                    # Block order.
                    bptr_order = ", ".join([str(x) for x in stride_order(inpstride)])
                    if not bptr_order:
                        bptr_order = "0"
                    self.indented_buffer.writeline(
                        f"{inpstr}_bptr = tl.make_block_ptr(base = {inpstr}_ptr, shape = [{bptr_shape}], strides = [{bptr_stride}], offsets = [{bptr_offsets}], block_shape = [{bptr_outshape}], order = [{bptr_order}])"
                    )
                    blockptr_nodes.append(inp)

                for inp in self.input_nodes:
                    if not is_tensor_node(inp):
                        continue
                    if inp not in blockptr_nodes:
                        continue
                    inpstr = str(inp)
                    inpshape = get_tensor_metas(inp)[0].shape
                    checkdims = inp.meta.get(TILEDIMNAME, None)
                    # incase broadcast.
                    if checkdims:
                        checkdims = [x for x in checkdims[0] if inpshape[x] > 1]
                    checkdim_str = ""
                    if checkdims:
                        checkdim_str = ",".join([str(x) for x in checkdims]) + (
                            "," if len(checkdims) <= 1 else ""
                        )
                    self.indented_buffer.writeline(
                        f"{inpstr} = tl.load({inpstr}_bptr, boundary_check=({checkdim_str}), padding_option='zero')"
                    )
                # load end
                self.indented_buffer.writeline()
                self.indented_buffer.writeline(
                    "# Add the logic for processing the tensors here"
                )

                for node in self.targetnodes:
                    tritonlines, extralines = convert_to_triton(node)
                    self.indented_buffer.writelines(tritonlines)
                    # self.extral_lines.writelines(extralines)
                    extralines_kep += extralines

                # store the output tensors 0
                self.indented_buffer.writeline()
                self.indented_buffer.writeline(f"# gen store output[0] ptrs and masks")
                nosymbol_prefix = "nosymbol"
                for inp in self.output_nodes[:1]:
                    if not is_tensor_node(inp):
                        continue
                    inpstr = str(inp)
                    inpshape = get_tensor_metas(inp)[0].shape
                    tiledims = inp.meta.get(TILEDIMNAME, None)
                    # in case broadcast.
                    checktile = bool(tiledims) and any(
                        [inpshape[x] > 1 for x in tiledims[0]]
                    )
                    arg_dims = []
                    dim_prefix_name = f"{nosymbol_prefix}_{inpstr}_dim"
                    for dimind, dim in enumerate(inpshape):
                        dim_ptr_str = f"{dim_prefix_name}{dimind}"
                        dim_slice = (
                            ["None"] * dimind
                            + [":"]
                            + ["None"] * (len(inpshape) - dimind - 1)
                        )
                        dim_slice_str = ",".join(dim_slice)
                        dim_arange_str = f"tl.arange(0, {dim})"
                        if checktile and (dimind in tiledims[0]):
                            dim_arange_str = f"(tile_id * {BATCHBLOCKNAME} + tl.arange(0, {BATCHBLOCKNAME}))"
                        dim_arange_str += f"[{dim_slice_str}]"
                        self.indented_buffer.writeline(
                            f"{dim_ptr_str} = {dim_arange_str}"
                        )
                        arg_dims.append(f'"{dim_ptr_str}"')
                    # mask
                    mask_name = f"{nosymbol_prefix}_{inpstr}_mask"
                    self.indented_buffer.writeline(
                        f"{mask_name} = {dim_prefix_name}{tiledims[0][0]} < {NUMTASKSNAME}"
                    )
                    arg_dims_str = ", ".join(arg_dims)
                    arg_inpstr = f'"{str(inp)}"'
                    arg_mask_str = f'"{mask_name}"'
                    self.indented_buffer.writeline(
                        f"{{{{store_output([{arg_dims_str}], {arg_inpstr}, {arg_mask_str}, indent_width = 8)}}}}"
                    )

                # store other outputs tensors
                for out in self.output_nodes[1:]:
                    if not is_tensor_node(out):
                        continue
                    outstr = str(out)
                    checkdims = out.meta.get(TILEDIMNAME, None)
                    # in case broadcast.
                    out_tensormeta = get_tensor_metas(out)[0]
                    outshape = out_tensormeta.shape
                    if checkdims:
                        checkdims = [x for x in checkdims[0] if outshape[x] > 1]
                    checkdim_str = ""
                    if checkdims:
                        checkdim_str = ",".join([str(x) for x in checkdims]) + (
                            "," if len(checkdims) <= 1 else ""
                        )
                    targetdtype = TORCH2TRITON_LOAD_STORE_DTYPE_STR[
                        str(out_tensormeta.dtype)
                    ]
                    self.indented_buffer.writeline(
                        f"tl.store({outstr}_bptr, {outstr}.to({targetdtype}), boundary_check=({checkdim_str}))"
                    )
        self.indented_buffer.writelines(extralines_kep)
        self.indented_buffer.writeline(f'"""')
        self.indented_buffer.writeline()
        self.indented_buffer.writeline(f"num_warps = {NUM_WARPS}")

        self.indented_buffer.writeline()
        self.indented_buffer.writeline(
            f"{self.graphname_tritontemplate} = TritonTemplate(name = '{self.triton_template_name}', \
                grid=get_triton_inductor_grid_fn(), source={self.graphname_tritontemplate}, \
                debug={TRITONFUSIONDEBUGNAME})"
        )
        self.indented_buffer.writeline()
        self.indented_buffer.writeline(
            f"def {self.graphname_tuned_inductor}(allinputs: Tuple[torch.Tensor, ...]):"
        )
        with self.indented_buffer.indent():
            inputargs = ",".join([str(x) for x in self.input_nodes])
            real_inputs = []
            for ind, nod in enumerate(self.input_nodes):
                if is_tensor_node(nod):
                    real_inputs.append(f"realize_inputs(allinputs[{ind}])")
                else:
                    real_inputs.append(f"allinputs[{ind}]")
            real_inputs = ", ".join(real_inputs)
            self.indented_buffer.writeline(f"{inputargs} = {real_inputs}")

            # Init dynamic shape symbols.
            self.gen_symbol_def_for_inputs(
                self.indented_buffer,
                stride_str="%s.get_stride()[%s]",
                cvt_str="sympy.floor(%s)",
            )
            # Numtask.
            self.indented_buffer.writeline(
                f"{NUMTASKSNAME}_target = {self.get_shape_size_hint(self.tiledimsize[0])}"
            )

            self.indented_buffer.writeline(f"choices = []")
            # first out tensor use FixedLayout
            out_node = self.output_nodes[0]
            out_node_tile_dims = out_node.meta.get(TILEDIMNAME, None)
            assert (
                out_node_tile_dims is not None
            ), f"get wrong tile dims of out node 0: {out_node}   tiledims: {out_node_tile_dims}"
            # Get device.
            device_str = f"{DEVICENAME}:0"
            if isinstance(out_node.meta.get(VALMETANAME, None), torch.Tensor):
                device_str = str(node.meta[VALMETANAME].device)
            out_tensormeta = get_tensor_metas(out_node)[0]
            out_shapes = [
                self.get_valid_expr_str_with_inputs(x) for x in out_tensormeta.shape
            ]
            out_shapes_str = ", ".join(out_shapes)
            out_strides = [
                self.get_valid_expr_str_with_inputs(x) for x in out_tensormeta.stride
            ]
            out_strides_str = ", ".join(out_strides)
            self.indented_buffer.writeline(
                f"layout = FixedLayout(torch.device('{device_str}'), {str(get_tensor_metas(out_node)[0].dtype)}, \
                    [{out_shapes_str}], [{out_strides_str}])"
            )
            # other output tensors use empty_strided
            for outind, inp in enumerate(self.output_nodes[1:]):
                if not is_tensor_node(inp):
                    continue
                inpstr = str(inp)
                inp_tensormeta = get_tensor_metas(inp)[0]
                inpshape = [
                    self.get_valid_expr_str_with_inputs(x)
                    for x in list(inp_tensormeta.shape)
                ]
                inpshape_str = ", ".join(inpshape)
                inpstride = [
                    self.get_valid_expr_str_with_inputs(x)
                    for x in list(inp_tensormeta.stride)
                ]
                inpstride_str = ", ".join(inpstride)
                inpdtype = inp_tensormeta.dtype
                self.indented_buffer.writeline(
                    f"{inpstr} = empty_strided([{inpshape_str}], [{inpstride_str}], dtype={inpdtype}, device=layout.device)"
                )
            self.indented_buffer.writeline("try:")
            with self.indented_buffer.indent():
                self.indented_buffer.writeline(
                    f"for config in get_triton_inductor_config({NUMTASKSNAME}_target):"
                )
                with self.indented_buffer.indent():
                    inpnodes_str = ", ".join(
                        [
                            str(x)
                            for x in self.input_nodes + self.output_nodes[1:]
                            if is_tensor_node(x)
                        ]
                    )
                    mutated_str = (
                        "[" + (", ".join([str(x) for x in self.output_nodes[1:]])) + "]"
                        if self.output_nodes[1:]
                        else "None"
                    )
                    call_sizes = [f"layout.size[{x}]" for x in out_node_tile_dims[0]]
                    call_sizes = f"[{','.join(call_sizes)}]"
                    self.indented_buffer.writeline(
                        f"{self.graphname_tritontemplate}.maybe_append_choice(choices, \
                            input_nodes=[{inpnodes_str}], layout=layout, mutated_inputs={mutated_str}, \
                                call_sizes={call_sizes}, **config)"
                    )
                inpnodes_str = ", ".join(
                    [
                        str(x)
                        for x in self.input_nodes + self.output_nodes[1:]
                        if is_tensor_node(x)
                    ]
                )
                outnodes_str = " ".join(["," + str(x) for x in self.output_nodes[1:]])
                self.indented_buffer.writeline(
                    f"return (autotune_select_algorithm('{self.triton_template_name}', \
                    choices, [{inpnodes_str}], layout){outnodes_str})"
                )
            self.indented_buffer.writeline(
                "except Exception as e: # NoValidChoicesError:"
            )
            with self.indented_buffer.indent():
                self.indented_buffer.writeline(
                    "if not inductor_config.autotune_fallback_to_aten: raise e"
                )
                self.indented_buffer.writeline(f"import traceback")
                self.indented_buffer.writeline(f"traceback.print_exc()")
                self.indented_buffer.writeline(
                    f"logging.error(f'All choice for {self.graphname} were invalid, get error:\\n{{e}}')"
                )
                self.indented_buffer.writeline(
                    f"logging.info(f'Using Tritonfusion Aten Fallback Func: {self.graphname_fallback_aten} Instead')"
                )
                self.indented_buffer.writeline(
                    f"return {self.graphname_fallback_aten}(allinputs)"
                )

    def gen_fallback_aten(self):
        """
        Generate inductor fallback aten func.
        """
        self.indented_buffer.writeline(
            f"\n# Aten fallback kernel for {self.graphname} with {len(self.input_nodes)} inputs and {len(self.output_nodes)} outputs"
        )
        self.indented_buffer.writeline(
            f"def {self.graphname_fallback_aten}(allinputs):"
        )
        inputargs = ", ".join([str(x) for x in self.input_nodes])
        allinputs_unpacking = ",".join(
            [f"allinputs[{ind}]" for ind in range(len(self.input_nodes))]
        )
        with self.indented_buffer.indent():
            self.indented_buffer.writeline(f"{inputargs} = {allinputs_unpacking}")
            # For dynamic shape arg, use tensor's shape but not static shape.
            self.gen_symbol_def_for_inputs(
                self.indented_buffer,
                stride_str="%s.get_stride()[%s]",
                cvt_str="sympy.floor(%s)",
            )
            for node in self.input_nodes:
                if not is_tensor_node(node):
                    self.indented_buffer.writeline(
                        f"{node} = {self.get_valid_expr_str_with_inputs(node.meta[VALMETANAME])}"
                    )
            # Gen torch ops.
            for node in self.targetnodes:
                node_metas = get_tensor_metas(node)
                node_target = get_target_name(node)
                ext_choic = get_externkernelchoice(node_target)
                torch_fn = ext_choic.torch_fn
                if torch_fn == operator.getitem:
                    inp, index = node.args[:2]
                    self.indented_buffer.writeline(f"{str(node)} = {inp}[{index}]")
                else:
                    args_names = [x.name for x in torch_fn._schema.arguments]
                    nodeargs_tensor = [
                        self.format_torch_arg(x)
                        for x in node.all_input_nodes
                        if is_tensor_node(x)
                    ]
                    nodeargs_str = ", ".join(nodeargs_tensor)
                    # Process kwargs, add const args.
                    nkwargs = dict(copy.copy(node.kwargs))
                    for ind, x in enumerate(node.args):
                        nkwargs[args_names[ind]] = x

                    nodekwargs = ", ".join(
                        [f"{k}={self.format_torch_arg(v)}" for k, v in nkwargs.items()]
                    )
                    nodekwargs = "" if not nodekwargs else ("," + nodekwargs)
                    # Get device.
                    device_str = "mlu:0"
                    if isinstance(node.meta.get(VALMETANAME, None), torch.Tensor):
                        device_str = str(node.meta[VALMETANAME].device)

                    out_shapes = [
                        self.get_valid_expr_str_with_inputs(x)
                        for x in node_metas[0].shape
                    ]
                    out_shapes_str = ", ".join(out_shapes)
                    out_strides = [
                        self.get_valid_expr_str_with_inputs(x)
                        for x in node_metas[0].stride
                    ]
                    out_strides_str = ", ".join(out_strides)
                    layout_str = f"FixedLayout(torch.device('{device_str}'), {str(node_metas[0].dtype)}, \
                            [{out_shapes_str}], [{out_strides_str}])"
                    self.indented_buffer.writeline(
                        f"{str(node)} = get_externkernelchoice('{node_target}').bind([{nodeargs_str}], {layout_str}{nodekwargs}).output_node()"
                    )
            self.indented_buffer.writeline(
                f"return {', '.join([str(out) for out in self.output_nodes])}"
            )
        self.indented_buffer.writeline("\n")

    def transform(self):
        """
        Transform the symbolic traced graph to Triton code.
        """
        self.gen_import()
        # Gen fallback aten.
        self.gen_fallback_aten()
        # gen inductor triton code.
        self.gen_inductor_tritontemplate()
        # gen general triton kernel.
        self.gen_triton_kernel()
        # gen wrapper and tests.
        self.gen_wrapper()
        self.gen_tests()
        # gen all module code.
        self.gen_module()
