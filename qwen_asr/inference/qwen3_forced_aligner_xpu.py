import torch
from transformers import AutoModel, AutoProcessor, AutoConfig
from .qwen3_forced_aligner import Qwen3ForcedAligner, Qwen3ForceAlignProcessor
from qwen_asr.core.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)

class Qwen3ForcedAlignerXPU(Qwen3ForcedAligner):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ) -> "Qwen3ForcedAlignerXPU":
        
        AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
        AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
        AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)

        if "attn_implementation" in kwargs and kwargs["attn_implementation"] == "flash_attention_2":
            kwargs["attn_implementation"] = "sdpa"

        print(f"[XPU] Loading Forced Aligner from {pretrained_model_name_or_path} to {kwargs.get('device_map', 'xpu')}...")
        
        model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        if not isinstance(model, Qwen3ASRForConditionalGeneration):
            raise TypeError(
                f"AutoModel returned {type(model)}, expected Qwen3ASRForConditionalGeneration."
            )
            
        processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, fix_mistral_regex=True)
        aligner_processor = Qwen3ForceAlignProcessor()
        
        return cls(model=model, processor=processor, aligner_processor=aligner_processor)