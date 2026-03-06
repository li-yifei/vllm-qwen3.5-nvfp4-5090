#!/usr/bin/env python3
"""Fix NVFP4 quantization exclusion for Qwen3.5 Mamba-hybrid models.

Patches modelopt.py to exclude layers that should remain BF16, and
patches qwen3_5.py to handle any remaining size mismatches gracefully.
"""
import sys, os, glob


def remove_pyc():
    for pattern in [
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/__pycache__/modelopt*.pyc",
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/__pycache__/qwen3_5*.pyc",
    ]:
        for f in glob.glob(pattern):
            os.remove(f)
            print(f"Removed: {f}")


def patch_modelopt():
    target = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py"
    with open(target) as f:
        content = f.read()

    if "PATCH_V5" in content:
        print("modelopt.py: already patched v5")
        remove_pyc()
        return

    # Remove all previous patch versions
    for marker in ["PATCH_V4", "QWEN35_LINEAR_ATTN_PATCH_V3", "QWEN35_LINEAR_ATTN_PATCH_V2"]:
        while marker in content:
            idx = content.index(marker)
            line_start = content.rfind("\n", 0, idx) + 1
            guard = "        if len(self.exclude_modules) == 0:\n            return False"
            guard_idx = content.index(guard, idx)
            content = content[:line_start] + content[guard_idx:]

    old = "        if len(self.exclude_modules) == 0:\n            return False"
    # Exclude: linear_attn (Mamba), mtp, shared_expert_gate, mlp.gate (MoE router)
    # These are all stored as BF16 in Qwen3.5 NVFP4 checkpoints
    new = """\
        # PATCH_V5: Exclude BF16 layers in Qwen3.5 NVFP4 checkpoints.
        # The model's ignore list has these but HF-to-vLLM name mapping
        # fails to translate the patterns correctly.
        _bf16_markers = ["linear_attn", "shared_expert_gate", ".mlp.gate"]
        for _m in _bf16_markers:
            if _m in prefix:
                return True
        if prefix.startswith("mtp."):
            return True

        if len(self.exclude_modules) == 0:
            return False"""
    if old not in content:
        print("ERROR: cannot find target in modelopt.py")
        sys.exit(1)
    content = content.replace(old, new, 1)
    with open(target, "w") as f:
        f.write(content)
    print("modelopt.py: patched v5 (linear_attn + gate + shared_expert_gate + mtp)")
    remove_pyc()


def patch_qwen3_5():
    target = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py"
    with open(target) as f:
        content = f.read()

    if "LOAD_PATCH_V2" in content:
        print("qwen3_5.py: already patched v2")
        remove_pyc()
        return

    # Remove old v1 patch if present
    old_v1 = (
        '                    # LOAD_PATCH_V1: handle BF16/FP4 size mismatch for linear_attn\n'
        '                    if param.size() != loaded_weight.size() and "linear_attn" in name:\n'
        '                        import torch\n'
        '                        new_data = torch.empty(loaded_weight.size(), dtype=loaded_weight.dtype, device=param.device)\n'
        '                        new_data.copy_(loaded_weight)\n'
        '                        param.data = new_data\n'
        '                        loaded_params.add(name)\n'
        '                        continue\n'
    )
    if old_v1 in content:
        content = content.replace(old_v1, '')

    old = (
        '                    param = params_dict[name]\n'
        '                    weight_loader = getattr(\n'
        '                        param, "weight_loader", default_weight_loader\n'
        '                    )\n'
        '                    weight_loader(param, loaded_weight)'
    )
    new = (
        '                    param = params_dict[name]\n'
        '                    # LOAD_PATCH_V2: handle size mismatch for unquantized layers\n'
        '                    if param.size() != loaded_weight.size():\n'
        '                        import logging as _logging\n'
        '                        _log = _logging.getLogger(__name__)\n'
        '                        _log.warning(\n'
        '                            f"Size mismatch for {name}: param={param.size()} "\n'
        '                            f"loaded={loaded_weight.size()}, "\n'
        '                            f"re-materializing as unquantized"\n'
        '                        )\n'
        '                        import torch\n'
        '                        param.data = loaded_weight.to(\n'
        '                            dtype=param.dtype, device=param.device\n'
        '                        )\n'
        '                        loaded_params.add(name)\n'
        '                        continue\n'
        '                    weight_loader = getattr(\n'
        '                        param, "weight_loader", default_weight_loader\n'
        '                    )\n'
        '                    weight_loader(param, loaded_weight)'
    )

    if old not in content:
        print("ERROR: cannot find target in qwen3_5.py")
        sys.exit(1)
    content = content.replace(old, new, 1)
    with open(target, "w") as f:
        f.write(content)
    print("qwen3_5.py: patched load_weights v2")
    remove_pyc()


if __name__ == "__main__":
    patch_modelopt()
    patch_qwen3_5()
