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
    import numpy as np
    from qwen_tts import Qwen3TTSModel

    model_path = request.get("model_path", "")
    text = request.get("text", "")
    speaker = request.get(
        "speaker", "Vivian"
    )  # Valid: aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian
    language = request.get("language", "Auto")
    instruct = request.get("instruct", "")

    # Voice cloning parameters
    ref_audio_b64 = request.get("ref_audio_base64", None)
    ref_text = request.get("ref_text", None)
    use_voice_clone = request.get("use_voice_clone", False)

    # Debug logging
    print(f"[TTS] use_voice_clone: {use_voice_clone}", file=sys.stderr)
    print(
        f"[TTS] has_ref_audio: {ref_audio_b64 is not None and len(ref_audio_b64) > 0 if ref_audio_b64 else False}",
        file=sys.stderr,
    )
    print(
        f"[TTS] has_ref_text: {ref_text is not None and len(ref_text) > 0 if ref_text else False}",
        file=sys.stderr,
    )

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
        # Voice cloning - works with Base and CustomVoice models
        if use_voice_clone and ref_audio_b64 and ref_text:
            # Decode base64 audio to numpy array
            audio_bytes = base64.b64decode(ref_audio_b64)

            # Try to load audio using different methods
            ref_audio_array = None
            ref_sr = None
            ref_audio_path = None

            try:
                # First, try loading directly with soundfile (works for WAV, FLAC, OGG)
                import io

                try:
                    ref_audio_array, ref_sr = sf.read(io.BytesIO(audio_bytes))
                except Exception:
                    pass

                # If that failed, try pydub (handles WebM, MP3, etc.)
                if ref_audio_array is None:
                    try:
                        from pydub import AudioSegment
                        import io as io_module

                        # Load audio with pydub (auto-detects format)
                        audio_segment = AudioSegment.from_file(
                            io_module.BytesIO(audio_bytes)
                        )

                        # Convert to mono if stereo
                        if audio_segment.channels > 1:
                            audio_segment = audio_segment.set_channels(1)

                        # Get sample rate and samples
                        ref_sr = audio_segment.frame_rate
                        samples = audio_segment.get_array_of_samples()

                        # Convert to numpy float array normalized to [-1, 1]
                        ref_audio_array = np.array(samples, dtype=np.float32)
                        ref_audio_array = ref_audio_array / (
                            2 ** (audio_segment.sample_width * 8 - 1)
                        )

                    except ImportError:
                        # pydub not available, fall back to temp file approach
                        pass
                    except Exception as e:
                        print(f"pydub failed: {e}", file=sys.stderr)

                # Last resort: save to temp file and try different extensions
                if ref_audio_array is None:
                    for ext in [".webm", ".mp3", ".ogg", ".wav", ".m4a"]:
                        try:
                            with tempfile.NamedTemporaryFile(
                                suffix=ext, delete=False
                            ) as f:
                                ref_audio_path = f.name
                                f.write(audio_bytes)

                            ref_audio_array, ref_sr = sf.read(ref_audio_path)
                            break
                        except Exception:
                            if ref_audio_path and os.path.exists(ref_audio_path):
                                os.unlink(ref_audio_path)
                                ref_audio_path = None
                            continue

                if ref_audio_array is None:
                    return {
                        "error": "Could not decode reference audio. Please upload a WAV, MP3, or OGG file."
                    }

                # Generate with voice cloning
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=language,
                    ref_audio=(ref_audio_array, ref_sr),
                    ref_text=ref_text,
                )
            finally:
                if ref_audio_path and os.path.exists(ref_audio_path):
                    os.unlink(ref_audio_path)

        elif "CustomVoice" in model_id:
            # CustomVoice without voice cloning - use speaker presets
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
        elif "Base" in model_id:
            # Base models require voice cloning
            return {
                "error": "Base models require voice cloning. Please provide reference_audio and reference_text, or load a CustomVoice model to use speaker presets like 'Vivian'."
            }
        else:
            return {"error": f"Unknown model type: {model_id}"}

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
