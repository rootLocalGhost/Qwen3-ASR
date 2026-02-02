import base64
import io
import os
import urllib.request
from typing import Tuple
import numpy as np
import soundfile as sf
import torch

# Import the new XPU specific class
# Ensure you are running this from the root directory of the project
from qwen_asr.inference.qwen3_asr_xpu import Qwen3ASRModelXPU

# --- CONFIGURATION CHANGED HERE ---
# Point to the local directories where you downloaded the models
ASR_MODEL_PATH = "./Qwen3-ASR-1.7B"
FORCED_ALIGNER_PATH = "./Qwen3-ForcedAligner-0.6B"
# ----------------------------------

# Test Audio URLs
URL_ZH = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav"
URL_EN = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"

def _download_audio_bytes(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()

def _to_data_url_base64(audio_bytes: bytes, mime: str = "audio/wav") -> str:
    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def main() -> None:
    # Validate paths exist before running
    if not os.path.exists(ASR_MODEL_PATH):
        raise FileNotFoundError(f"Model path not found: {ASR_MODEL_PATH}. Make sure you are running from the project root.")
    
    # Check if XPU is actually available
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print(f"XPU Available: {torch.xpu.get_device_name(0)}")
    else:
        print("WARNING: Torch XPU not detected. This might fallback to CPU or fail.")

    print(f"Initializing Qwen3-ASR on XPU from local path: {ASR_MODEL_PATH}...")
    
    # Initialize using the XPU wrapper
    asr = Qwen3ASRModelXPU.from_pretrained(
        ASR_MODEL_PATH,
        dtype=torch.bfloat16,     # A770 supports bfloat16 well
        device_id=0,              # Maps to xpu:0
        attn_implementation="sdpa", # Use Scaled Dot Product Attention (native Torch)
        forced_aligner=FORCED_ALIGNER_PATH,
        forced_aligner_kwargs=dict(
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ),
        max_inference_batch_size=16, 
        max_new_tokens=256,
    )

    # 1. Test Single Inference
    print("\n--- Testing Single Inference (English) ---")
    results = asr.transcribe(
        audio=URL_EN,
        language="English",
        return_time_stamps=True,
    )
    
    for r in results:
        print(f"Text: {r.text}")
        if r.time_stamps:
            print(f"First Word Timestamp: {r.time_stamps[0]}")

    # 2. Test Batch Inference (Mixed)
    print("\n--- Testing Batch Inference (Mixed) ---")
    zh_bytes = _download_audio_bytes(URL_ZH)
    zh_b64 = _to_data_url_base64(zh_bytes, mime="audio/wav")
    
    results = asr.transcribe(
        audio=[URL_ZH, zh_b64, URL_EN],
        language=["Chinese", "Chinese", "English"],
        return_time_stamps=True,
    )
    
    for i, r in enumerate(results):
        print(f"\n[Sample {i}] Language: {r.language}")
        print(f"Text: {r.text}")

if __name__ == "__main__":
    main()