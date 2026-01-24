//! Python bridge for Qwen3-TTS inference
//! Calls the official qwen_tts Python package for actual model inference

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use tracing::{debug, info, warn};

use crate::error::{Error, Result};

/// Request to Python inference script
#[derive(Debug, Serialize)]
pub struct PythonTTSRequest {
    pub command: String,
    pub model_path: String,
    pub text: String,
    pub speaker: Option<String>,
    pub language: Option<String>,
    pub instruct: Option<String>,
}

/// Response from Python inference script
#[derive(Debug, Deserialize)]
pub struct PythonTTSResponse {
    pub audio_base64: Option<String>,
    pub sample_rate: Option<u32>,
    pub format: Option<String>,
    pub error: Option<String>,
    pub status: Option<String>,
}

/// Python TTS bridge for calling qwen_tts
pub struct PythonBridge {
    script_path: String,
    python_cmd: String,
}

impl PythonBridge {
    /// Create a new Python bridge
    pub fn new() -> Self {
        // Find the script path relative to the binary
        let script_path = std::env::current_dir()
            .map(|p| p.join("scripts/tts_inference.py"))
            .unwrap_or_else(|_| "scripts/tts_inference.py".into())
            .to_string_lossy()
            .to_string();

        Self {
            script_path,
            python_cmd: "python3".to_string(),
        }
    }

    /// Check if Python dependencies are available
    pub fn check_dependencies(&self) -> Result<bool> {
        let request = serde_json::json!({
            "command": "check"
        });

        match self.call_python(&request.to_string()) {
            Ok(response) => {
                if response.status.as_deref() == Some("ok") {
                    Ok(true)
                } else if let Some(err) = response.error {
                    warn!("Python dependencies not available: {}", err);
                    Ok(false)
                } else {
                    Ok(false)
                }
            }
            Err(e) => {
                warn!("Failed to check Python dependencies: {}", e);
                Ok(false)
            }
        }
    }

    /// Generate TTS audio using Python
    pub fn generate(
        &self,
        model_path: &Path,
        text: &str,
        speaker: Option<&str>,
        language: Option<&str>,
        instruct: Option<&str>,
    ) -> Result<(Vec<f32>, u32)> {
        info!("Calling Python TTS for text: {}", text);

        let request = PythonTTSRequest {
            command: "generate".to_string(),
            model_path: model_path.to_string_lossy().to_string(),
            text: text.to_string(),
            speaker: speaker.map(|s| s.to_string()),
            language: language.map(|s| s.to_string()),
            instruct: instruct.map(|s| s.to_string()),
        };

        let request_json = serde_json::to_string(&request)
            .map_err(|e| Error::InferenceError(format!("Failed to serialize request: {}", e)))?;

        let response = self.call_python(&request_json)?;

        if let Some(err) = response.error {
            return Err(Error::InferenceError(format!("Python TTS error: {}", err)));
        }

        let audio_b64 = response
            .audio_base64
            .ok_or_else(|| Error::InferenceError("No audio in response".to_string()))?;

        let sample_rate = response.sample_rate.unwrap_or(24000);

        // Decode base64 to WAV bytes
        use base64::Engine;
        let wav_bytes = base64::engine::general_purpose::STANDARD
            .decode(&audio_b64)
            .map_err(|e| Error::InferenceError(format!("Failed to decode audio: {}", e)))?;

        // Parse WAV and extract samples
        let samples = parse_wav_samples(&wav_bytes)?;

        debug!("Generated {} samples at {} Hz", samples.len(), sample_rate);

        Ok((samples, sample_rate))
    }

    /// Call Python script with JSON request
    fn call_python(&self, request_json: &str) -> Result<PythonTTSResponse> {
        let mut child = Command::new(&self.python_cmd)
            .arg(&self.script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| Error::InferenceError(format!("Failed to start Python: {}", e)))?;

        // Write request to stdin
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(request_json.as_bytes())
                .map_err(|e| Error::InferenceError(format!("Failed to write to Python: {}", e)))?;
        }

        // Wait for response
        let output = child
            .wait_with_output()
            .map_err(|e| Error::InferenceError(format!("Python process failed: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::InferenceError(format!("Python error: {}", stderr)));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);

        // Filter out flash-attn warning that qwen_tts prints to stdout
        let json_str = stdout
            .lines()
            .find(|line| line.trim().starts_with('{'))
            .unwrap_or(&stdout);

        serde_json::from_str(json_str).map_err(|e| {
            Error::InferenceError(format!(
                "Failed to parse Python response: {} - {}",
                e, json_str
            ))
        })
    }
}

impl Default for PythonBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse WAV bytes and extract f32 samples
fn parse_wav_samples(wav_bytes: &[u8]) -> Result<Vec<f32>> {
    use std::io::Cursor;

    let cursor = Cursor::new(wav_bytes);
    let mut reader = hound::WavReader::new(cursor)
        .map_err(|e| Error::InferenceError(format!("Failed to parse WAV: {}", e)))?;

    let spec = reader.spec();

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
    };

    Ok(samples)
}
