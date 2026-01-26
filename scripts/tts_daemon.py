#!/usr/bin/env python3
"""
Persistent TTS Daemon for Qwen3-TTS.
Keeps models loaded in memory and accepts requests via Unix socket.
"""

import sys
import os
import io
import json
import signal
import socket
import struct
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, Optional, Any
from collections import OrderedDict

# Suppress all warnings before importing heavy libraries
import warnings

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Default socket path
DEFAULT_SOCKET_PATH = "/tmp/izwi_tts_daemon.sock"
MAX_CACHED_MODELS = 2  # Keep at most 2 models in memory


class LRUModelCache:
    """LRU cache for loaded TTS models."""

    def __init__(self, max_size: int = MAX_CACHED_MODELS):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.Lock()

    def get(self, model_id: str):
        """Get model from cache, updating LRU order."""
        with self.lock:
            if model_id in self.cache:
                self.cache.move_to_end(model_id)
                return self.cache[model_id]
            return None

    def put(self, model_id: str, model):
        """Add model to cache, evicting oldest if needed."""
        with self.lock:
            if model_id in self.cache:
                self.cache.move_to_end(model_id)
                return

            # Evict oldest if at capacity
            while len(self.cache) >= self.max_size:
                evicted_id, evicted_model = self.cache.popitem(last=False)
                print(
                    f"[Daemon] Evicting model from cache: {evicted_id}", file=sys.stderr
                )
                del evicted_model
                # Force garbage collection for GPU memory
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass

            self.cache[model_id] = model

    def remove(self, model_id: str):
        """Remove model from cache."""
        with self.lock:
            if model_id in self.cache:
                del self.cache[model_id]
                return True
            return False

    def clear(self):
        """Clear all cached models."""
        with self.lock:
            self.cache.clear()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass

    def list_models(self) -> list:
        """List cached model IDs."""
        with self.lock:
            return list(self.cache.keys())


class TTSDaemon:
    """TTS Daemon that handles requests via Unix socket."""

    def __init__(self, socket_path: str = DEFAULT_SOCKET_PATH):
        self.socket_path = socket_path
        self.model_cache = LRUModelCache()
        self.running = False
        self.server_socket = None
        self.device = None
        self.dtype = None
        self.attn_impl = None
        self._init_device()

    def _init_device(self):
        """Initialize device settings."""
        try:
            import torch

            if torch.cuda.is_available():
                self.device = "cuda:0"
                self.dtype = torch.bfloat16
                self.attn_impl = "flash_attention_2"
            elif torch.backends.mps.is_available():
                self.device = "mps"
                self.dtype = torch.float32
                self.attn_impl = "eager"
            else:
                self.device = "cpu"
                self.dtype = torch.float32
                self.attn_impl = "eager"
            print(f"[Daemon] Using device: {self.device}", file=sys.stderr)
        except ImportError:
            self.device = "cpu"
            self.dtype = None
            self.attn_impl = "eager"

    def _get_hf_model_id(self, model_path: str) -> str:
        """Convert local model path to HuggingFace model ID."""
        model_name = os.path.basename(model_path.rstrip("/"))
        hf_models = {
            "Qwen3-TTS-12Hz-0.6B-Base": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            "Qwen3-TTS-12Hz-0.6B-CustomVoice": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "Qwen3-TTS-12Hz-1.7B-Base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "Qwen3-TTS-12Hz-1.7B-CustomVoice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "Qwen3-TTS-12Hz-1.7B-VoiceDesign": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        }
        return hf_models.get(model_name, f"Qwen/{model_name}")

    def _load_model(self, model_id: str):
        """Load model, using cache if available."""
        cached = self.model_cache.get(model_id)
        if cached is not None:
            print(f"[Daemon] Using cached model: {model_id}", file=sys.stderr)
            return cached

        print(f"[Daemon] Loading model: {model_id}", file=sys.stderr)
        start_time = time.time()

        from qwen_tts import Qwen3TTSModel
        import torch

        model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=self.device,
            dtype=self.dtype,
            attn_implementation=self.attn_impl,
        )

        load_time = time.time() - start_time
        print(f"[Daemon] Model loaded in {load_time:.2f}s: {model_id}", file=sys.stderr)

        self.model_cache.put(model_id, model)
        return model

    def _handle_check(self, request: dict) -> dict:
        """Handle dependency check request."""
        try:
            import torch
            import soundfile
            from qwen_tts import Qwen3TTSModel

            return {"status": "ok", "device": self.device}
        except ImportError as e:
            return {"error": f"Missing dependency: {str(e)}"}

    def _handle_status(self, request: dict) -> dict:
        """Handle status request."""
        return {
            "status": "ok",
            "device": self.device,
            "cached_models": self.model_cache.list_models(),
            "pid": os.getpid(),
        }

    def _handle_preload(self, request: dict) -> dict:
        """Handle model preload request."""
        model_path = request.get("model_path", "")
        model_id = self._get_hf_model_id(model_path)

        try:
            self._load_model(model_id)
            return {"status": "ok", "model_id": model_id}
        except Exception as e:
            return {"error": f"Failed to preload model: {str(e)}"}

    def _handle_unload(self, request: dict) -> dict:
        """Handle model unload request."""
        model_path = request.get("model_path", "")
        if model_path:
            model_id = self._get_hf_model_id(model_path)
            if self.model_cache.remove(model_id):
                return {"status": "ok", "unloaded": model_id}
            return {"error": f"Model not loaded: {model_id}"}
        else:
            self.model_cache.clear()
            return {"status": "ok", "unloaded": "all"}

    def _handle_generate(self, request: dict) -> dict:
        """Handle TTS generation request."""
        import numpy as np
        import soundfile as sf
        import tempfile
        import base64

        model_path = request.get("model_path", "")
        text = request.get("text", "")
        speaker = request.get("speaker", "Vivian")
        language = request.get("language", "Auto")
        instruct = request.get("instruct", "")

        # Voice cloning parameters
        ref_audio_b64 = request.get("ref_audio_base64")
        ref_text = request.get("ref_text")
        use_voice_clone = request.get("use_voice_clone", False)

        # Debug logging
        print(f"[Daemon] Generate request - model_path: {model_path}", file=sys.stderr)
        print(
            f"[Daemon] use_voice_clone: {use_voice_clone}, has_ref_audio: {ref_audio_b64 is not None}, has_ref_text: {ref_text is not None}",
            file=sys.stderr,
        )
        if ref_text:
            print(f"[Daemon] ref_text: {ref_text[:50]}...", file=sys.stderr)

        model_id = self._get_hf_model_id(model_path)
        print(f"[Daemon] Resolved model_id: {model_id}", file=sys.stderr)

        try:
            model = self._load_model(model_id)
        except Exception as e:
            return {"error": f"Failed to load model {model_id}: {str(e)}"}

        print(
            f"[Daemon] Checking voice clone condition: use_voice_clone={use_voice_clone}, has_audio={ref_audio_b64 is not None and len(ref_audio_b64) > 0 if ref_audio_b64 else False}, has_text={ref_text is not None and len(ref_text) > 0 if ref_text else False}",
            file=sys.stderr,
        )

        try:
            # Voice cloning - works with Base and CustomVoice models
            if use_voice_clone and ref_audio_b64 and ref_text:
                ref_audio_array, ref_sr = self._decode_reference_audio(ref_audio_b64)
                if ref_audio_array is None:
                    return {
                        "error": "Could not decode reference audio. Please upload a WAV, MP3, or OGG file."
                    }

                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=language,
                    ref_audio=(ref_audio_array, ref_sr),
                    ref_text=ref_text,
                )
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

        # Convert to WAV bytes
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

    def _decode_reference_audio(self, ref_audio_b64: str):
        """Decode reference audio from base64."""
        import numpy as np
        import soundfile as sf
        import base64
        import tempfile

        audio_bytes = base64.b64decode(ref_audio_b64)

        # Try soundfile directly
        try:
            return sf.read(io.BytesIO(audio_bytes))
        except:
            pass

        # Try pydub for WebM/MP3
        try:
            from pydub import AudioSegment

            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            if audio_segment.channels > 1:
                audio_segment = audio_segment.set_channels(1)
            ref_sr = audio_segment.frame_rate
            samples = audio_segment.get_array_of_samples()
            ref_audio_array = np.array(samples, dtype=np.float32)
            ref_audio_array = ref_audio_array / (
                2 ** (audio_segment.sample_width * 8 - 1)
            )
            return ref_audio_array, ref_sr
        except:
            pass

        # Try temp file with different extensions
        for ext in [".webm", ".mp3", ".ogg", ".wav", ".m4a"]:
            try:
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                    temp_path = f.name
                    f.write(audio_bytes)
                result = sf.read(temp_path)
                os.unlink(temp_path)
                return result
            except:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        return None, None

    def handle_request(self, request: dict) -> dict:
        """Route request to appropriate handler."""
        command = request.get("command", "generate")

        handlers = {
            "check": self._handle_check,
            "status": self._handle_status,
            "preload": self._handle_preload,
            "unload": self._handle_unload,
            "generate": self._handle_generate,
            "shutdown": lambda r: {"status": "shutdown"},
        }

        handler = handlers.get(command)
        if handler:
            return handler(request)
        return {"error": f"Unknown command: {command}"}

    def _recv_message(self, conn: socket.socket) -> Optional[dict]:
        """Receive length-prefixed JSON message."""
        try:
            # Read 4-byte length prefix
            length_data = conn.recv(4)
            if not length_data:
                return None
            length = struct.unpack(">I", length_data)[0]

            # Read message body
            data = b""
            while len(data) < length:
                chunk = conn.recv(min(length - len(data), 65536))
                if not chunk:
                    return None
                data += chunk

            return json.loads(data.decode("utf-8"))
        except Exception as e:
            print(f"[Daemon] Error receiving message: {e}", file=sys.stderr)
            return None

    def _send_message(self, conn: socket.socket, message: dict):
        """Send length-prefixed JSON message."""
        try:
            data = json.dumps(message).encode("utf-8")
            length = struct.pack(">I", len(data))
            conn.sendall(length + data)
        except Exception as e:
            print(f"[Daemon] Error sending message: {e}", file=sys.stderr)

    def _handle_client(self, conn: socket.socket, addr):
        """Handle a single client connection."""
        try:
            while self.running:
                request = self._recv_message(conn)
                if request is None:
                    break

                # Handle shutdown specially
                if request.get("command") == "shutdown":
                    self._send_message(conn, {"status": "shutdown"})
                    self.running = False
                    break

                try:
                    response = self.handle_request(request)
                except Exception as e:
                    response = {"error": f"Internal error: {str(e)}"}
                    traceback.print_exc(file=sys.stderr)

                self._send_message(conn, response)
        finally:
            conn.close()

    def start(self):
        """Start the daemon server."""
        # Remove existing socket file
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(5)
        self.server_socket.settimeout(1.0)  # Allow checking self.running

        # Set socket permissions
        os.chmod(self.socket_path, 0o600)

        self.running = True
        print(
            f"[Daemon] Started on {self.socket_path} (PID: {os.getpid()})",
            file=sys.stderr,
        )

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        while self.running:
            try:
                conn, addr = self.server_socket.accept()
                # Handle each client in a thread
                thread = threading.Thread(target=self._handle_client, args=(conn, addr))
                thread.daemon = True
                thread.start()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[Daemon] Accept error: {e}", file=sys.stderr)

        self.stop()

    def stop(self):
        """Stop the daemon server."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        self.model_cache.clear()
        print("[Daemon] Stopped", file=sys.stderr)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"[Daemon] Received signal {signum}, shutting down...", file=sys.stderr)
        self.running = False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="TTS Daemon for Qwen3-TTS")
    parser.add_argument(
        "--socket", default=DEFAULT_SOCKET_PATH, help="Unix socket path"
    )
    parser.add_argument("--preload", help="Model to preload on startup")
    args = parser.parse_args()

    daemon = TTSDaemon(socket_path=args.socket)

    # Preload model if specified
    if args.preload:
        print(f"[Daemon] Preloading model: {args.preload}", file=sys.stderr)
        try:
            daemon._load_model(daemon._get_hf_model_id(args.preload))
        except Exception as e:
            print(f"[Daemon] Preload failed: {e}", file=sys.stderr)

    daemon.start()


if __name__ == "__main__":
    main()
