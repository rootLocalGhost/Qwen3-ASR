import torch
import numpy as np
from typing import Optional, Dict, Any, List, Union
from transformers import AutoModel, AutoProcessor
from .qwen3_asr import Qwen3ASRModel, ASRTranscription
from .qwen3_forced_aligner_xpu import Qwen3ForcedAlignerXPU
from .utils import normalize_audio_input

class Qwen3ASRModelXPU(Qwen3ASRModel):
    """
    XPU-compatible version of Qwen3ASRModel for Intel ARC GPUs.
    Supports Transformers backend with XPU optimization.
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        forced_aligner: Optional[str] = None,
        forced_aligner_kwargs: Optional[Dict[str, Any]] = None,
        max_inference_batch_size: int = 32,
        max_new_tokens: Optional[int] = 512,
        device_id: int = 0,
        **kwargs,
    ) -> "Qwen3ASRModelXPU":
        
        device_str = f"xpu:{device_id}"
        print(f"[XPU] Loading ASR model from {pretrained_model_name_or_path} to {device_str}...")

        # Force SDPA for XPU (Flash Attn 2 is CUDA only)
        if "attn_implementation" in kwargs and kwargs["attn_implementation"] == "flash_attention_2":
            kwargs["attn_implementation"] = "sdpa"
        
        kwargs["device_map"] = device_str

        model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, fix_mistral_regex=True)

        forced_aligner_model = None
        if forced_aligner is not None:
            f_kwargs = forced_aligner_kwargs or {}
            f_kwargs["device_map"] = device_str
            # Ensure aligner uses XPU wrapper
            forced_aligner_model = Qwen3ForcedAlignerXPU.from_pretrained(
                forced_aligner, **f_kwargs
            )

        return cls(
            backend="transformers",
            model=model,
            processor=processor,
            sampling_params=None,
            forced_aligner=forced_aligner_model,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
        )

    @classmethod
    def LLM(cls, *args, **kwargs):
        raise NotImplementedError("vLLM backend is not supported on XPU yet.")

    def transcribe_stream_chunk(self, new_audio, full_buffer, language=None):
        """
        Simulate streaming by appending audio and re-transcribing.
        This is necessary because true streaming relies on vLLM.
        """
        # Normalize incoming chunk
        new_audio = normalize_audio_input(new_audio)
        
        # Append to buffer (numpy concat)
        if full_buffer is None:
            full_buffer = new_audio
        else:
            full_buffer = np.concatenate((full_buffer, new_audio))

        # Transcribe the full buffer (or you could implement windowing here)
        # For A770, re-transcribing up to 30s is very fast.
        results = self.transcribe(
            audio=full_buffer,
            language=language,
            return_time_stamps=False # Timestamps slow down streaming
        )
        
        text = results[0].text if results else ""
        return full_buffer, text