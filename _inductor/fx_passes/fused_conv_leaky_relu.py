import torch

aten = torch.ops.aten
prims = torch.ops.prims


# convolution = torch.ops.aten.convolution.default(x, ...)
# gt = torch.ops.aten.gt.Scalar(convolution, 0)
# mul = torch.ops.aten.mul.Tensor(convolution, 0.2)
# where = torch.ops.aten.where.self(gt, convolution, mul)


# convolution = torch.ops.aten.convolution.default(x, ...)
# convert_element_type = torch.ops.prims.convert_element_type.default(convolution, torch.float32)
# gt = torch.ops.aten.gt.Scalar(convert_element_type, 0)
# mul = torch.ops.aten.mul.Tensor(convert_element_type, 0.2)
# where = torch.ops.aten.where.self(gt, convert_element_type, mul)
# convert_element_type_1 = torch.ops.prims.convert_element_type.default(where, torch.float16)


def conv_leaky_relu_fusion_pass(graph: torch.fx.Graph):
    # ---------- collect matched pattern nodes ----------
    post_convert_list = []
    where_list = []
    gt_list = []
    mul_list = []
    slope_list = []
    pre_convert_list = []
    conv_list = []
    replace_anchor_list = []

    # ---------- stage 1: pattern matching ----------
    for node in graph.nodes:
        # 1) match 'where(gt, x, mul)'
        where = node
        if node.target != aten.where.self or len(node.args) != 3:
            continue
        # 2) match 'gt(x > 0)', must be used only once
        gt = where.args[0]
        if (
            gt.target != aten.gt.Scalar
            or len(gt.args) != 2
            or gt.args[1] != 0
            or len(gt.users) != 1
            or list(gt.users.keys())[0] is not where
        ):
            continue
        # 3) match 'mul(x * negative_slope)', must be used only once
        mul = where.args[2]
        if (
            mul.target != aten.mul.Tensor
            or len(mul.args) != 2
            or len(mul.users) != 1
            or list(mul.users.keys())[0] is not where
        ):
            continue
        slope = mul.args[1]
        if not isinstance(slope, (float, int)):
            continue

        # 4) match 'conv' or 'convert_element_type(conv)'
        conv = where.args[1]
        pre_convert = None
        if conv.target == prims.convert_element_type.default:
            if conv.args[1] != torch.float32:
                continue
            pre_convert = conv
            conv = conv.args[0]
        if conv.target != aten.convolution.default:
            continue

        # 5) enforce consistent input across gt/mul/where
        conv_ref = pre_convert if pre_convert is not None else conv
        if not (
            gt.args[0] is conv_ref
            and mul.args[0] is conv_ref
            and where.args[1] is conv_ref
        ):
            continue

        # 6) enforce users constraint to ensure safe fusion
        if pre_convert is None:
            # conv output must only feed gt/mul/where
            if len(conv.users) != 3:
                continue
        else:
            # conv -> pre_convert -> gt/mul/where
            if (
                len(conv.users) != 1
                or list(conv.users.keys())[0] is not pre_convert
                or len(pre_convert.users) != 3
            ):
                continue

        # 7) match 'convert_element_type(where)'
        post_convert = None
        if (
            len(where.users) == 1
            and list(where.users.keys())[0].target == prims.convert_element_type.default
            and list(where.users.keys())[0].args[1] == torch.float16
        ):
            post_convert = list(where.users.keys())[0]

        # 8) device/dtype constraint (only fuse on MLU)
        conv_args = conv.args
        if len(conv_args) != 9:
            continue
        bias = conv_args[2]
        if isinstance(bias, torch.fx.Node):
            bias = bias.meta["val"]
        if (
            "val" not in conv.meta
            or not hasattr(conv.meta["val"], "device")
            or conv.meta["val"].device.type != "mlu"
            or not hasattr(conv.meta["val"], "dtype")
            or conv.meta["val"].dtype
            not in (torch.float16, torch.float32, torch.bfloat16, torch.float64)
            or conv_args[1].meta["val"].dtype != conv.meta["val"].dtype
            or (bias is not None and bias.dtype != conv.meta["val"].dtype)
            or conv_args[6] is not False
            or conv_args[8] != 1
        ):
            continue

        # 9) replacement anchor: use post_convert if exists, else where
        replace_anchor = post_convert if post_convert is not None else where

        # ---------- collect matched node ----------
        post_convert_list.append(post_convert)
        where_list.append(where)
        gt_list.append(gt)
        mul_list.append(mul)
        slope_list.append(float(slope))
        pre_convert_list.append(pre_convert)
        conv_list.append(conv)
        replace_anchor_list.append(replace_anchor)

    # ---------- stage 2: fusion + graph rewrite ----------
    for i in range(len(where_list)):
        post_convert = post_convert_list[i]
        where = where_list[i]
        gt = gt_list[i]
        mul = mul_list[i]
        slope = slope_list[i]
        pre_convert = pre_convert_list[i]
        conv = conv_list[i]
        replace_anchor = replace_anchor_list[i]

        # https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml
        # convolution(
        #     Tensor input,
        #     Tensor weight,
        #     Tensor? bias,
        #     SymInt[] stride,
        #     SymInt[] padding,
        #     SymInt[] dilation,
        #     bool transposed,
        #     SymInt[] output_padding,
        #     SymInt groups
        # ) -> Tensor
        conv_args = conv.args
        if len(conv_args) != 9:
            continue

        # insert fused op after anchor
        with graph.inserting_after(replace_anchor):
            fused = graph.call_function(
                torch.ops.torch_mlu.fused_convolution.default,
                args=tuple(conv_args) + ("leaky_relu",) + (slope,),
            )
            if hasattr(replace_anchor, "meta"):
                fused.meta.update(replace_anchor.meta)

        # replace original output
        replace_anchor.replace_all_uses_with(fused)

        # remove old nodes (bottom-up delete order)
        if post_convert is not None:
            graph.erase_node(post_convert)
        graph.erase_node(where)
        graph.erase_node(gt)
        graph.erase_node(mul)
        if pre_convert is not None:
            graph.erase_node(pre_convert)
        graph.erase_node(conv)

    return
