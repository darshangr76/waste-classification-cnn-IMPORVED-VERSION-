# TensorRT-LLM on Colab T4 — Failure Analysis

## Summary
TRT-LLM 1.0.0 installed successfully on Colab (Python 3.12, CUDA 12.8)
and loaded Qwen3-4B, but crashed during engine initialization.

## Environment
- GPU: Tesla T4 (sm_75, Turing)
- CUDA: 12.8
- Python: 3.12.3 (venv)
- TRT-LLM: 1.0.0
- TensorRT: 10.11.0.33 (CUDA 12 wheels)

## What Worked
- TRT-LLM install (CUDA 12 wheels)
- Qwen3-4B download (7.6 GB)
- Qwen3 architecture support confirmed
- Single-process mode (bypassing MPI)
- FlashInfer JIT compilation (with ninja + PATH fix)

## What Failed
- Inference crashes in `AttentionOp::initialize` inside libtensorrt_llm.so
- Cause: Turing (sm_75) lacks fused multi-head attention (FMHA) kernels
  required by TRT-LLM 1.x

## Conclusion
TRT-LLM 1.x is not viable on T4 for Qwen3-4B. Requires Ampere or
newer GPU (A100, L40S, RTX 30/40 series).

## Next Steps
- Try on college Ubuntu (check lspci for A100/L40S)
- Or rent on AIC Cloud (UPI accepted)
- Or pivot Minor #3 to vLLM / custom CUDA kernel
