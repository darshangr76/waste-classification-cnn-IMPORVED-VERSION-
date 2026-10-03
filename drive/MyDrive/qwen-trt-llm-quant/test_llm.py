import os
import shutil

# Force venv bin onto PATH so ninja + nvcc are findable
venv_bin = "/content/trtllm-venv/bin"
os.environ["PATH"] = venv_bin + ":" + os.environ.get("PATH", "")

# Verify ninja is findable BEFORE importing TRT-LLM
ninja_path = shutil.which("ninja")
print(f"ninja found at: {ninja_path}")
if not ninja_path:
    # Fallback: use the one inside the venv directly
    import sys
    candidate = os.path.join(os.path.dirname(sys.executable), "ninja")
    if os.path.exists(candidate):
        os.environ["PATH"] = os.path.dirname(candidate) + ":" + os.environ["PATH"]
        ninja_path = candidate
        print(f"fallback ninja: {ninja_path}")
    else:
        raise RuntimeError("ninja not found anywhere")

# Single-process mode (no MPI)
os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
os.environ["OMPI_MCA_rmaps_base_oversubscribe"] = "1"

import time
from tensorrt_llm import LLM, SamplingParams

print("Loading Qwen3-4B via TRT-LLM...")
t0 = time.time()

llm = LLM(model="/content/models/qwen3-4b", dtype="float16", max_seq_len=2048)

t1 = time.time()
print(f"Loaded in {t1-t0:.1f}s")

sampling = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=100)
outputs = llm.generate(["What is machine learning? Explain in 3 sentences."], sampling)

for out in outputs:
    print("Output:", out.outputs[0].text)
    print("Tokens:", len(out.outputs[0].token_ids))
