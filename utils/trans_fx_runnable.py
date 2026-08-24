import argparse
import re


def convert_fx_to_demo(input_file, output_file):
    with open(input_file, "r") as f:
        lines = f.readlines()

    inside_init = False
    init_lines = []  # 提取 Repro 类中的 __init__ 方法
    inside_forward = False
    forward_lines = []  # 提取 Repro 类中的 forward 方法
    symints = {}
    args = {}
    for line in lines:
        # Match symbolic integers (e.g., s0 = reader.symint(16)  # primals_1)
        symint_match = re.match(r"\s*reader\.symint\((-?\d+)\).*?#\s*(\w+)", line)
        if symint_match:
            sym_value, sym_name = symint_match.groups()
            symints[sym_name] = int(sym_value)

    for line in lines:
        # Extract __init__ function
        if line.strip().startswith("def __init__("):
            inside_init = True
        if inside_init:
            init_lines.append(line)
            if line.strip() == "":
                inside_init = False

        if "def forward" in line:
            inside_forward = True
            # Modify the function signature to exclude symbolic arguments
            match = re.search(r"def forward\(self, (.*?)\):", line)
            if match:
                args = match.group(1).split(", ")

        if inside_forward:
            # remove part of ";..."
            cleaned_line = re.sub(r";.*", "", line)
            forward_lines.append(cleaned_line)
        if inside_forward and line.strip() == "":
            inside_forward = False

    # Get device, dtype from reader.storage
    buf_to_info = {}
    for line in lines:
        storage_match = re.search(
            r"(\w+) = reader\.storage\(.*?,\s*(\d+),\s*device=device\(type=['\"](.*?)['\"],\s*index=(\d+)\)(?:,\s*dtype_hint=(torch\.\w+))?\)",
            line,
        )
        if storage_match:
            buf_name = storage_match.group(1)
            size_bytes = int(storage_match.group(2))
            device_type = storage_match.group(3)
            device_index = storage_match.group(4)
            dtype = storage_match.group(5) or "torch.float32"  # 默认 dtype
            buf_to_info[buf_name] = {
                "device": f"{device_type}:{device_index}",
                "dtype": dtype,
                "size_bytes": size_bytes,  # 新增
            }

    # Get tensor_name, shape from reader.tensor
    inputs = []
    buffers_to_create = set()  # 新增：记录需要创建的buffer
    for line in lines:
        # 新增：匹配带strides和offset的情况
        tensor_match = re.search(
            r"reader\.tensor\((\w+),\s*\((.*?)\)"  # name, shape
            r"(?:,\s*\((.*?)\))?"  # optional stride
            r"(?:,\s*storage_offset=(\d+))?"  # optional storage_offset
            r"(?:,\s*dtype=torch\.(\w+))?"  # optional dtype
            r".*?\)\s*#\s*(\w+)",  # closing and comment
            line,
        )
        if tensor_match:
            buf_name = tensor_match.group(1)
            shape = tensor_match.group(2)
            strides = tensor_match.group(3)
            offset = tensor_match.group(4) or "0"
            dtype = (
                f"torch.{tensor_match.group(5)}"
                if tensor_match.group(5) is not None
                else f"torch.float32"
            )
            tensor_name = tensor_match.group(6)
            device = buf_to_info.get(buf_name, {}).get("device", "cpu")
            # 新增：判断是否需要特殊内存布局
            needs_strided = (strides is not None) or (offset != "0")

            inputs.append(
                {
                    "tensor_name": tensor_name,
                    "shape": shape,
                    "device": device,
                    "dtype": dtype,
                    "needs_strided": needs_strided,  # 新增
                    "buf_name": buf_name if needs_strided else None,  # 新增
                    "strides": strides if needs_strided else None,  # 新增
                    "offset": offset if needs_strided else None,  # 新增
                }
            )

            if needs_strided:
                buffers_to_create.add(buf_name)  # 记录需要创建的buffer

    # Create a mapping for symbolic variables
    symbolic_mapping = {}
    for name in symints:
        match = re.search(r"arg(\d+)_\d+|primals_(\d+)", name)
        if match:
            if "primals_" in name:
                idx = int(match.group(2)) - 1  # Subtract 1 for primals
            else:
                idx = int(match.group(1))  # Use the index as-is for arg
            symbolic_mapping[f"s{idx}"] = symints[name]

    # 生成 PyTorch Demo
    with open(output_file, "w") as f:
        f.write("import torch\n\n")
        f.write("from torch import tensor, device\n")
        f.write("from math import inf, nan\n\n")
        f.write("import torch_mlu_ops\n\n")
        f.write("class Repro(torch.nn.Module):\n")
        f.write("".join(init_lines))
        f.write("".join(forward_lines) + "\n")
        f.write("if __name__ == '__main__':\n")
        f.write("    model = Repro()\n")
        f.write("    model.eval()\n\n")

        for sym_name in symints:
            f.write(f"    {sym_name} = {symints[sym_name]}\n")

        # 新增：首先创建需要的buffer
        for buf_name in buffers_to_create:
            info = buf_to_info[buf_name]
            dtype_size = 2 if "float16" in info["dtype"] else 4  # 简化处理
            num_elements = info["size_bytes"] // dtype_size
            if info["dtype"] in ("torch.int", "torch.int64", "torch.int32"):
                f.write(
                    f"    {buf_name} = torch.randint(1, 10, ({num_elements},), device='{info['device']}', dtype={info['dtype']})\n"
                )
            elif info["dtype"] in ("torch.bool"):
                f.write(
                    f"    {buf_name} = torch.randint(0, 2, ({num_elements},), device='{info['device']}').to({info['dtype']})\n"
                )
            else:
                f.write(
                    f"    {buf_name} = torch.randn(({num_elements}), device='{info['device']}', dtype={info['dtype']})\n"
                )

        # 然后创建tensor
        for item in inputs:
            resolved_shape = item["shape"]
            for sym, value in symbolic_mapping.items():
                resolved_shape = resolved_shape.replace(sym, str(value))

            if item["needs_strided"]:
                # 处理需要特殊内存布局的情况
                resolved_strides = item["strides"]
                for sym, value in symbolic_mapping.items():
                    resolved_strides = resolved_strides.replace(sym, str(value))

                f.write(
                    f"    {item['tensor_name']} = torch.as_strided("
                    f"{item['buf_name']}, "
                    f"({resolved_shape}), "
                    f"({resolved_strides}), "
                    f"storage_offset={item['offset']})\n"
                )
            else:
                # 原有逻辑
                if item["dtype"] in ("torch.int", "torch.int64", "torch.int32"):
                    f.write(
                        f"    {item['tensor_name']} = torch.randint(1, 10, ({resolved_shape}), device='{item['device']}', dtype={item['dtype']})\n"
                    )
                elif item["dtype"] in ("torch.bool"):
                    f.write(
                        f"    {item['tensor_name']} = torch.randint(0, 2, ({resolved_shape}), device='{item['device']}').to({item['dtype']})\n"
                    )
                else:
                    f.write(
                        f"    {item['tensor_name']} = torch.randn(({resolved_shape}), device='{item['device']}', dtype={item['dtype']})\n"
                    )

        input_names_str = ", ".join(args)
        f.write("\n    opt_model = torch.compile(model)\n")
        f.write(f"\n    output = opt_model({input_names_str})\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="将 fx_graph_runnable.py 转换为 Pytorch demo 脚本",
    )
    parser.add_argument(
        "-i",
        "--input_file",
        required=True,
        help="输入文件路径 (必需)",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        required=True,
        help="输出文件路径 (必需)",
    )
    return parser.parse_args()


def main():
    try:
        args = parse_args()
        convert_fx_to_demo(args.input_file, args.output_file)
        print(f"成功将 {args.input_file} 转换为 {args.output_file}")
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"转换过程中发生错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
