#!/usr/bin/env python3
"""
Python inference bridge for Qwen3-TTS.
Called by the Rust server to generate audio.
"""

import sys
import os
import io

# Redirect stderr IMMEDIATELY to capture all warnings
_original_stderr = sys.stderr
sys.stderr = io.StringIO()

import json
import base64
import tempfile
import warnings

# Suppress warnings to avoid polluting JSON output
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


def check_dependencies():
    """Check if required packages are installed."""
    try:
        import torch
        import soundfile as sf
        from qwen_tts import Qwen3TTSModel

        return True
    except ImportError as e:
        return str(e)


def get_hf_model_id(model_path: str) -> str:
    """Convert local model path to HuggingFace model ID."""
    # Extract model name from path
    model_name = os.path.basename(model_path.rstrip("/"))

    # Map to HuggingFace model IDs
    hf_models = {
        "Qwen3-TTS-12Hz-0.6B-Base": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "Qwen3-TTS-12Hz-0.6B-CustomVoice": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "Qwen3-TTS-12Hz-1.7B-Base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "Qwen3-TTS-12Hz-1.7B-CustomVoice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "Qwen3-TTS-12Hz-1.7B-VoiceDesign": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    }

    return hf_models.get(model_name, f"Qwen/{model_name}")


def generate_tts(request: dict) -> dict:
    """Generate TTS audio from text."""
    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    model_path = request.get("model_path", "")
    text = request.get("text", "")
    speaker = request.get(
        "speaker", "Vivian"
    )  # Valid: aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian
    language = request.get("language", "Auto")
    instruct = request.get("instruct", "")

    # Use HuggingFace model ID instead of local path
    model_id = get_hf_model_id(model_path)

    # Determine device - use float32 on MPS to avoid numerical instability
    if torch.cuda.is_available():
        device = "cuda:0"
        dtype = torch.bfloat16
        attn_impl = "flash_attention_2"
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32  # MPS has issues with float16 causing inf/nan
        attn_impl = "eager"
    else:
        device = "cpu"
        dtype = torch.float32
        attn_impl = "eager"

    # Load model from HuggingFace
    try:
        model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_impl,
        )
    except Exception as e:
        return {"error": f"Failed to load model {model_id}: {str(e)}"}

    # Generate audio based on model type
    try:
        if "CustomVoice" in model_id:
            wavs, sr = model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct if instruct else None,
            )
        elif "VoiceDesign" in model_id:
            wavs, sr = model.generate_voice_design(
                text=text,
                language=language,
                instruct=instruct if instruct else "Natural speaking voice.",
            )
        else:
            # Base model - simple generation without voice clone
            # For base models, we need ref_audio for voice clone
            # Without ref_audio, generate with default voice
            wavs, sr = model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
            )
    except Exception as e:
        return {"error": f"Generation failed: {str(e)}"}

    # Save to temp file and read as bytes
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name

    try:
        sf.write(temp_path, wavs[0], sr)
        with open(temp_path, "rb") as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    finally:
        os.unlink(temp_path)

    return {
        "audio_base64": audio_b64,
        "sample_rate": sr,
        "format": "wav",
    }


def main():
    """Main entry point - reads JSON from stdin, writes JSON to stdout."""
    # Redirect stderr to suppress warnings from mixing with JSON
    import io

    sys.stderr = io.StringIO()

    # Read request from stdin
    request_json = sys.stdin.read()

    try:
        request = json.loads(request_json)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON: {str(e)}"}))
        sys.exit(1)

    command = request.get("command", "generate")

    if command == "check":
        result = check_dependencies()
        if result is True:
            print(json.dumps({"status": "ok"}))
        else:
            print(json.dumps({"error": f"Missing dependency: {result}"}))
    elif command == "generate":
        result = generate_tts(request)
        print(json.dumps(result))
    else:
        print(json.dumps({"error": f"Unknown command: {command}"}))


if __name__ == "__main__":
    main()
