"""Qwen3 / Qwen3-Coder preset for the swe-agent harbor-docker recipe.

Injects the Qwen3 model identity and SGLang/TITO parser flags, then delegates to
run.py's CLI. Every run.py flag still works, and anything you pass overrides the
preset. Use --coder for Qwen3-Coder-30B-A3B-Instruct (rope_theta 1e7).

    python run-qwen3-swe.py --prompt-data /root/swe_train.jsonl
    python run-qwen3-swe.py --coder --prompt-data /root/swe_train.jsonl
"""

import os
import sys

import typer

from run import main

_QWEN3 = [
    "--model-name", "Qwen3-30B-A3B",
    "--hf-checkpoint", "Qwen/Qwen3-30B-A3B",
    "--ref-load", "/root/Qwen3-30B-A3B_torch_dist",
    "--save-dir", "/root/Qwen3-30B-A3B_agent_v2/",
]

_QWEN3_CODER = [
    "--model-name", "Qwen3-Coder-30B-A3B-Instruct",
    "--hf-checkpoint", "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "--ref-load", "/root/Qwen3-Coder-30B-A3B-Instruct_torch_dist",
    "--save-dir", "/root/Qwen3-Coder-30B-A3B-Instruct_agent_v2/",
]

# Qwen3-30B-A3B covers the general and coder models (same arch); only the
# checkpoint and rope base differ. Parsers are shared across both.
_QWEN3_COMMON = [
    "--megatron-model-type", "qwen3-30B-A3B",
    "--sglang-tool-call-parser", "qwen25",
    "--sglang-reasoning-parser", "qwen3",
    "--tito-model", "qwen3",
]


if __name__ == "__main__":
    user_args = sys.argv[1:]
    coder = "--coder" in user_args
    user_args = [arg for arg in user_args if arg != "--coder"]

    # rope_theta: coder is 1e7, general is 1e6. Set it explicitly (never rely on
    # "unset") so a stale MODEL_ARGS_ROTARY_BASE exported in the shell can't leak
    # the wrong base into either path. Read by scripts/models/qwen3-30B-A3B.py.
    os.environ["MODEL_ARGS_ROTARY_BASE"] = "10000000" if coder else "1000000"

    preset = _QWEN3_CODER if coder else _QWEN3
    sys.argv = [sys.argv[0], *_QWEN3_COMMON, *preset, *user_args]
    typer.run(main)
