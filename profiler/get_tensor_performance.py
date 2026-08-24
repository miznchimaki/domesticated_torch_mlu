import subprocess
import shutil
import tempfile
import sys
import re
import math
from pathlib import Path


def get_tensor_performance():
    cnvs_path = shutil.which("cnvs")
    if cnvs_path is None:
        print("cnvs not found in PATH")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # gen default config file
        subprocess.run(
            ["cnvs -y"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=True,
            check=True,
            cwd=tmp_dir,
        )

        yml_files = list((Path(tmp_dir) / Path("cnvs_default_config")).glob("*.yml"))
        if len(yml_files) != 1:
            raise RuntimeError(
                f"Expected exactly one yml in cnvs_default_config, got {len(yml_files)}"
            )

        yml_file = yml_files[0]
        lines = yml_file.read_text().splitlines()
        new_lines = []
        in_matmul = False

        # set data_type to bfloat16
        for line in lines:
            stripped = line.strip()

            if stripped.startswith("matmul_performance:"):
                in_matmul = True
                new_lines.append(line)
                continue

            if in_matmul:
                if stripped.startswith("input_data_type:"):
                    indent = line[: len(line) - len(line.lstrip())]
                    line = f"{indent}input_data_type: bfloat16"
                elif stripped.startswith("output_data_type:"):
                    indent = line[: len(line) - len(line.lstrip())]
                    line = f"{indent}output_data_type: bfloat16"

                if not line.startswith(" ") and ":" in line:
                    in_matmul = False

            new_lines.append(line)

        yml_file.write_text("\n".join(new_lines) + "\n")

        # gen performance
        result = subprocess.run(
            [f"cnvs -r matmul_performance -c {yml_file}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
            text=True,
            check=True,
            cwd=tmp_dir,
        )

        # get performace value
        pattern = re.compile(r":\s*([0-9]+(?:\.[0-9]+)?)\s*\(MLU\s*\d+\)")
        values = []

        for line in result.stdout.splitlines():
            m = pattern.search(line)
            if m:
                values.append(float(m.group(1)) / 1000)

        if values:
            return math.ceil(max(values))
        else:
            return 0


if __name__ == "__main__":
    print(get_tensor_performance())
