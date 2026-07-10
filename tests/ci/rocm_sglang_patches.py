"""Apply ROCm-specific SGLang patches needed by the runnert-test suite.

These patch the SGLang install baked into the CI image (rocm/sgl-dev:...),
NOT miles source, so they live here and are applied as a CI workflow step
(rocm-mi350-pr-test.yml) rather than committed into SGLang. Idempotent.

1. serving_chat.py: attach choice.meta_info (with output_token_logprobs) when
   logprobs is requested, not only when return_meta_info is set. Required by
   the session-verify test (miles/rollout/session/core.py demands
   choice.meta_info.output_token_logprobs). Without it every session
   chat/completions returns 502.
2. deepseek_v2.py: narrow the yarn rope_type condition so transformers-v5
   auto-populated rope_scaling (rope_type="default") is not misclassified as
   deepseek_yarn -> avoids the GLM-4.7-Flash KeyError:
   'original_max_position_embeddings' at model load.
"""
from __future__ import annotations

import os

SGLANG = os.environ.get("SGLANG_DIR", "/sgl-workspace/sglang/python/sglang")

PATCHES = [
    (
        os.path.join(SGLANG, "srt/entrypoints/openai/serving_chat.py"),
        'ret_item["meta_info"] if request.return_meta_info else None',
        'ret_item["meta_info"] if (request.return_meta_info or request.logprobs) else None',
    ),
    (
        os.path.join(SGLANG, "srt/models/deepseek_v2.py"),
        '        if rope_scaling:\n            rope_scaling["rope_type"] = "deepseek_yarn"',
        '        if rope_scaling and rope_scaling.get("rope_type") in ("yarn", "deepseek_yarn"):\n'
        '            rope_scaling["rope_type"] = "deepseek_yarn"',
    ),
]


def main() -> int:
    rc = 0
    for path, old, new in PATCHES:
        name = os.path.basename(path)
        if not os.path.exists(path):
            print(f"SKIP {name}: not found at {path}")
            continue
        src = open(path).read()
        if new in src:
            print(f"OK   {name}: already patched")
            continue
        count = src.count(old)
        if count != 1:
            print(f"WARN {name}: expected 1 occurrence of target, found {count}; not patching")
            rc = 1
            continue
        open(path, "w").write(src.replace(old, new))
        print(f"PATCHED {name}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
