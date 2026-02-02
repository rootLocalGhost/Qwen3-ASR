import argparse
import gradio as gr
import torch
import numpy as np
import os
import sys
from typing import Dict, Optional

# Import Hugging Face Hub for auto-downloading
from huggingface_hub import snapshot_download

# Import our XPU wrappers
from qwen_asr.inference.qwen3_asr_xpu import Qwen3ASRModelXPU
from qwen_asr.inference.qwen3_forced_aligner_xpu import Qwen3ForcedAlignerXPU

# --- CONFIGURATION ---
# Define the mapping: Friendly Name -> (HuggingFace Repo ID, Local Directory)
ASR_CONFIG = {
    "1.7B": ("Qwen/Qwen3-ASR-1.7B", "./Qwen3-ASR-1.7B"),
    "0.6B": ("Qwen/Qwen3-ASR-0.6B", "./Qwen3-ASR-0.6B")
}
ALIGNER_CONFIG = ("Qwen/Qwen3-ForcedAligner-0.6B", "./Qwen3-ForcedAligner-0.6B")

# --- GLOBAL CACHE (Lazy Loaded) ---
# Models are loaded into VRAM only when requested
LOADED_ASR_MODELS: Dict[str, Qwen3ASRModelXPU] = {}
LOADED_ALIGNER: Optional[Qwen3ForcedAlignerXPU] = None
DEVICE_ID = 0

def ensure_content(repo_id, local_dir):
    """Checks if local_dir exists, if not, downloads from HF."""
    if not os.path.exists(local_dir):
        print(f"\n[Auto-Download] '{local_dir}' not found.")
        print(f"[Auto-Download] Downloading '{repo_id}' from Hugging Face... (Check terminal for progress)")
        try:
            snapshot_download(repo_id=repo_id, local_dir=local_dir)
            print(f"[Auto-Download] Successfully downloaded to {local_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}")
    return local_dir

def get_aligner():
    """Lazy loads the Aligner. Downloads if missing."""
    global LOADED_ALIGNER
    if LOADED_ALIGNER is not None:
        return LOADED_ALIGNER

    repo_id, local_dir = ALIGNER_CONFIG
    ensure_content(repo_id, local_dir)
    
    print(f"[Loader] Loading Aligner to xpu:{DEVICE_ID}...")
    LOADED_ALIGNER = Qwen3ForcedAlignerXPU.from_pretrained(
        local_dir,
        dtype=torch.bfloat16,
        device_map=f"xpu:{DEVICE_ID}",
        attn_implementation="sdpa"
    )
    return LOADED_ALIGNER

def get_asr_model(friendly_name):
    """Lazy loads the requested ASR model. Downloads if missing."""
    if friendly_name in LOADED_ASR_MODELS:
        return LOADED_ASR_MODELS[friendly_name]

    if friendly_name not in ASR_CONFIG:
        raise ValueError(f"Unknown model: {friendly_name}")

    repo_id, local_dir = ASR_CONFIG[friendly_name]
    ensure_content(repo_id, local_dir)

    print(f"[Loader] Loading Qwen3-ASR-{friendly_name} to xpu:{DEVICE_ID}...")
    
    # We try to load the aligner first so we can inject it
    # If aligner download fails, we proceed without it
    aligner_obj = None
    try:
        aligner_obj = get_aligner()
    except Exception as e:
        print(f"[Warning] Aligner could not be loaded: {e}")

    model = Qwen3ASRModelXPU.from_pretrained(
        local_dir,
        dtype=torch.bfloat16,
        device_id=DEVICE_ID,
        attn_implementation="sdpa",
        forced_aligner=None, # Inject manually
        max_inference_batch_size=1,
        max_new_tokens=1024,
    )
    
    if aligner_obj:
        model.forced_aligner = aligner_obj
        
    LOADED_ASR_MODELS[friendly_name] = model
    return model

# --- INFERENCE FUNCTIONS ---

def run_file_asr(model_name, audio, language, use_timestamps):
    if audio is None: 
        return "Please upload an audio file.", ""
    
    try:
        # This line triggers download/load if not ready
        model = get_asr_model(model_name)
        
        lang_in = None if language == "Auto" else language
        
        # Handle timestamp request if aligner failed to load
        if use_timestamps and not model.forced_aligner:
            print("Warning: Timestamps requested but aligner is missing.")
            use_timestamps = False

        results = model.transcribe(audio, language=lang_in, return_time_stamps=use_timestamps)
        res = results[0]
        
        txt = f"[{res.language}] {res.text}"
        
        html = "<div style='font-family: monospace; max-height: 400px; overflow-y: scroll; border:1px solid #444; padding:10px;'>"
        if res.time_stamps:
            for t in res.time_stamps:
                html += f"<div style='margin-bottom:4px;'><span style='color:#00AAFF;'>{t.start_time:.2f}s -> {t.end_time:.2f}s</span>: <b>{t.text}</b></div>"
        elif use_timestamps:
            html += "<div>No timestamps generated.</div>"
        else:
            html += "<div>Timestamps disabled.</div>"
        html += "</div>"
        
        return txt, html
    except Exception as e:
        return f"Error: {str(e)}\n(Check terminal for download progress if this was the first run)", ""

def run_mic_stream(model_name, audio, state, language):
    if audio is None:
        return state, state, ""
    
    try:
        # Trigger load
        model = get_asr_model(model_name)
        
        sr, y = audio
        if y.dtype.kind == 'i':
            y = y.astype(np.float32) / np.iinfo(y.dtype).max
        
        full_buffer = state
        lang_in = None if language == "Auto" else language

        new_buffer, text = model.transcribe_stream_chunk(y, full_buffer, language=lang_in)
        return new_buffer, new_buffer, text
    except Exception as e:
        print(f"Streaming Error: {e}")
        return state, state, ""

def run_pure_align(audio, text, language):
    if audio is None or not text.strip():
        return "Please provide audio and text."
    
    try:
        # Trigger load aligner only
        aligner = get_aligner()
        lang_in = "English" if language == "Auto" else language

        results = aligner.align(audio=audio, text=[text], language=[lang_in])
        res = results[0]
        
        html = "<div style='font-family: monospace; max-height: 400px; overflow-y: scroll; border:1px solid #444; padding:10px;'>"
        for t in res.items:
                html += f"<div style='margin-bottom:4px;'><span style='color:#00AAFF;'>{t.start_time:.2f}s -> {t.end_time:.2f}s</span>: <b>{t.text}</b></div>"
        html += "</div>"
        return html
    except Exception as e:
        return f"Error: {str(e)}"

# --- UI ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()
    
    global DEVICE_ID
    DEVICE_ID = args.device_id

    # Supported languages list (Hardcoded or partial to avoid loading model at startup)
    # This ensures the UI launches instantly.
    supported_langs = ["Auto", "English", "Chinese", "Cantonese", "Japanese", "Korean", "French", "German", "Spanish", "Russian"]

    with gr.Blocks(title="Qwen3-ASR XPU Ultimate", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# ⚡ Qwen3-ASR XPU Ultimate")
        gr.Markdown(f"Running on **Intel Arc (xpu:{args.device_id})**. Models will **auto-download** and load upon first use.")

        with gr.Row():
            model_selector = gr.Radio(
                choices=list(ASR_CONFIG.keys()), 
                value="1.7B", 
                label="ASR Model Version",
                info="If not found locally, it will download automatically when you click Transcribe."
            )

        with gr.Tabs():
            # TAB 1
            with gr.Tab("📁 File Transcription"):
                with gr.Row():
                    with gr.Column():
                        f_input = gr.Audio(type="filepath", label="Input Audio")
                        f_lang = gr.Dropdown(choices=supported_langs, value="Auto", label="Language")
                        f_ts = gr.Checkbox(value=True, label="Timestamps")
                        f_btn = gr.Button("Transcribe File", variant="primary")
                    with gr.Column():
                        f_txt = gr.TextArea(label="Text")
                        f_html = gr.HTML(label="Visual Timestamps")
                f_btn.click(run_file_asr, [model_selector, f_input, f_lang, f_ts], [f_txt, f_html])

            # TAB 2
            with gr.Tab("🎤 Microphone"):
                with gr.Row():
                    with gr.Column():
                        m_input = gr.Audio(sources=["microphone"], streaming=True)
                        m_lang = gr.Dropdown(choices=supported_langs, value="Auto", label="Language")
                        m_clear = gr.Button("Clear Buffer")
                    with gr.Column():
                        m_out = gr.TextArea(label="Live Transcription", lines=10)
                m_state = gr.State(None)
                m_input.stream(run_mic_stream, [model_selector, m_input, m_state, m_lang], [m_state, m_state, m_out])
                m_clear.click(lambda: (None, None, ""), None, [m_input, m_state, m_out])

            # TAB 3
            with gr.Tab("⏱️ Aligner Tool"):
                gr.Markdown("Align text with audio without running ASR.")
                with gr.Row():
                    with gr.Column():
                        a_audio = gr.Audio(type="filepath")
                        a_text = gr.TextArea(label="Transcript")
                        a_lang = gr.Dropdown(choices=supported_langs, value="Auto", label="Language")
                        a_btn = gr.Button("Align", variant="primary")
                    with gr.Column():
                        a_out = gr.HTML(label="Result")
                a_btn.click(run_pure_align, [a_audio, a_text, a_lang], [a_out])

    print(f"Ready on http://{args.ip}:{args.port}")
    demo.launch(server_name=args.ip, server_port=args.port)

if __name__ == "__main__":
    main()