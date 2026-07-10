# ROCm PR Local Test Report

- Container: `rlsys_miles`  |  Image: `rlsys/miles:MI350-355-latest`  |  Base: `radixark/main` (`8dccb85b9`)
- Overlay: `rlsys-compat` (12 shims)  |  Env prereq: `nvidia-modelopt` installed in container
- Last updated: 2026-07-10 03:34 (UTC-4)
- PR notation: `#OLD/NEW` = old closed PR / new superseding PR. **What is actually merged are the local `rocm/*` branches (= the NEW PRs), rebased onto `radixark/main`.** The old closed PRs are never fetched.

## Where we are
- Phase: RUNNING group 1 (`#1489/1611`)
- Completed: 1 / 14 testcases  (PASS 1, FAIL 0)
- Currently running: `test_sglang_config_mixed_offload` (#2) — log `/data/sreerohi/cache/work/1489_sglang_config_mixed_offload.log`
- Next up: `test_qwen2.5_0.5B_gsm8k_short` (`#1489/1611`)

### How to watch live progress
- This board (status + `Completed X/14`) is the top-level view; refresh this file.
- For the test whose row shows `RUN`, its live log path is in that row's notes column. Tail it:
  ```
  docker exec rlsys_miles bash -lc 'tail -f /root/work/<name>.log'   # from container
  tail -f /data/sreerohi/cache/work/<name>.log                       # from host (same file)
  ```

Legend: PASS - passed | FAIL - failed (log kept) | RUN - running now (log path in notes) | WAIT - pending | SKIP - skipped | DEFER - deferred

## Status board (ordered fastest -> slowest)

| # | PR (new/old) | testcase | branch @ commit — what was applied | status | duration | notes / error+log path |
|---|--------------|----------|-------------------------------------|--------|----------|------------------------|
| 1 | #1489/1611 | test_qwen2.5_0.5B_gsm8k_async_short | test/bgradb-all @ 8f4e2b81e — radixark/main + rocm/te-bgradb-workaround(#1618) + rocm/bgradb-gfx950-all(#1611) + rlsys-compat | PASS | ~5m30s | eval/gsm8k=0.48; job raysubmit_Nh7MWGHjMzmvSNFW succeeded |
| 2 | #1489/1611 | test_sglang_config_mixed_offload | test/bgradb-all @ 8a08770c5 — main + rocm/te-bgradb-workaround(#1618) + rocm/bgradb-gfx950-all(#1611) + rlsys-compat (overlay updated: +sglang_cuda_graph_backend_prefill default) | RUN | - | re-run after overlay fix; log: /data/sreerohi/cache/work/1489_sglang_config_mixed_offload.log |
| 3 | #1489/1611 | test_qwen2.5_0.5B_gsm8k_short | test/bgradb-all — main + rocm/te-bgradb-workaround(#1618) + rocm/bgradb-gfx950-all(#1611) + rlsys-compat | WAIT | - | - |
| 4 | #1489/1611 | test_sglang_config | test/bgradb-all — main + rocm/te-bgradb-workaround(#1618) + rocm/bgradb-gfx950-all(#1611) + rlsys-compat | WAIT | - | - |
| 5 | #1489/1611 | test_sglang_config_mixed_offload_ft | test/bgradb-all — main + rocm/te-bgradb-workaround(#1618) + rocm/bgradb-gfx950-all(#1611) + rlsys-compat | WAIT | - | - |
| 6 | #1240/1610 | test_session_server_multi_role/test_glm47 | test/session-verify — main + rocm/session-verify(#1610) + rlsys-compat | WAIT | - | - |
| 7 | #1163/1615 | test_mimo_7B_mtp_only_grad | test/mimo — main + rocm/qwen3-4B-ckpt(#1621) + rocm/te-bgradb-workaround(#1618) + rocm/mimo-7B-mtp-async-4plus4(#1615) + rlsys-compat | WAIT | - | - |
| 8 | #1166/1614 | test_quick_start_glm4_9B | test/glm4-9b — main + rocm/te-bgradb-workaround(#1618) + rocm/glm4-9b-gradient-fusion-fix(#1614) + rlsys-compat | WAIT | - | - |
| 9 | #1118/1621 | test_qwen3_4B_ckpt | test/qwen4b — main + rocm/qwen3-4B-ckpt(#1621) + rlsys-compat | WAIT | - | - |
| 10 | #1126/1620 | test_glm47_flash_ckpt | test/glm47-ckpt — main + rocm/glm4.7-Flash-fixes(#1620) + rlsys-compat (+SGLang deepseek_v2 patch) | WAIT | - | needs SGLang deepseek_v2 patch |
| 11 | #1160/1616 | test_qwen2.5_0.5B_gsm8k_async | test/gsm8k-async — main + rocm/te-bgradb-workaround(#1618) + rocm/bgradb-qwen-long-async(#1616) + rlsys-compat | WAIT | - | - |
| 12 | #1159/1617 | test_qwen2.5_0.5B_gsm8k | test/gsm8k — main + rocm/te-bgradb-workaround(#1618) + rocm/bgradb-qwen-long(#1617) + rlsys-compat | WAIT | - | - |
| 13 | #1242/1612 | test_glm47_flash (R3+MTP) | test/glm47-r3 — main + rocm/te-bgradb-workaround(#1618) + rocm/glm47-flash-r3-mtp-ci(#1612) + rlsys-compat (+SGLang deepseek_v2 patch) | WAIT | - | needs SGLang deepseek_v2 patch |
| 14 | #1172/1613 | test_glm5_744b_a40b_4layer | test/glm5 — main + dep/1122 + dep/1123 + rocm/glm5_744b_a440_4layer_changes(#1613) + rlsys-compat | DEFER | - | run last on signal; also needs modelopt + #1122/#1123 |

## Notes / running observations
- `modelopt` missing in image broke the first attempt (megatron.bridge hard-imports it); fixed by `pip install nvidia-modelopt` in the container. Applies to all bridge-mode Megatron tests.
- Direct `python <test>.py` needs `PYTHONPATH=/workspace/miles` so `tests.ci` resolves.
