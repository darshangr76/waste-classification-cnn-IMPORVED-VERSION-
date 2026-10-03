# TensorRT-LLM Setup on Colab T4

## Working configuration (Sep 2026)
- Colab base: Python 3.12.3
- TRT-LLM: 1.0.0
- TensorRT: 10.11.0.33 (CUDA 12 wheels)
- PyTorch: 2.7.1

## Install steps (fresh Colab session)

1. Ensure runtime is Python 3.12 (not 3.13)
2. sudo apt-get install -y python3.12-venv
3. /usr/bin/python3.12 -m venv /content/trtllm-venv
4. /content/trtllm-venv/bin/pip install --upgrade pip
5. /content/trtllm-venv/bin/pip install tensorrt_llm==1.0.0
6. Verify: /content/trtllm-venv/bin/python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"

## Why version 1.0.0
- Latest TRT-LLM (1.2.x) pulls CUDA 13 wheels -> libcublasLt.so.13 error on T4
- 1.0.0 is the last version that ships CUDA 12 wheels by default
- Works with Colab T4 (sm_75) and CUDA 12.8 driver
