//! LFM2-Audio API endpoints
//!
//! Handles TTS, ASR, and audio-to-audio chat via the LFM2 daemon.

use axum::{http::StatusCode, Json};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use tracing::info;

const LFM2_SOCKET_PATH: &str = "/tmp/izwi_lfm2_daemon.sock";

/// LFM2 TTS request
#[derive(Debug, Deserialize)]
pub struct LFM2TTSRequest {
    pub text: String,
    #[serde(default = "default_voice")]
    pub voice: String,
    #[serde(default)]
    pub max_new_tokens: Option<u32>,
    #[serde(default)]
    pub audio_temperature: Option<f32>,
    #[serde(default)]
    pub audio_top_k: Option<u32>,
}

fn default_voice() -> String {
    "us_female".to_string()
}

/// LFM2 ASR request
#[derive(Debug, Deserialize)]
pub struct LFM2ASRRequest {
    pub audio_base64: String,
    #[serde(default)]
    pub max_new_tokens: Option<u32>,
}

/// LFM2 Audio Chat request
#[derive(Debug, Deserialize)]
pub struct LFM2AudioChatRequest {
    #[serde(default)]
    pub audio_base64: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub max_new_tokens: Option<u32>,
    #[serde(default)]
    pub audio_temperature: Option<f32>,
    #[serde(default)]
    pub audio_top_k: Option<u32>,
}

/// LFM2 TTS response
#[derive(Debug, Serialize)]
pub struct LFM2TTSResponse {
    pub audio_base64: String,
    pub sample_rate: u32,
    pub format: String,
}

/// LFM2 ASR response
#[derive(Debug, Serialize)]
pub struct LFM2ASRResponse {
    pub transcription: String,
}

/// LFM2 Audio Chat response
#[derive(Debug, Serialize)]
pub struct LFM2AudioChatResponse {
    pub text: String,
    pub audio_base64: Option<String>,
    pub sample_rate: u32,
    pub format: String,
}

/// LFM2 Status response
#[derive(Debug, Serialize)]
pub struct LFM2StatusResponse {
    pub status: String,
    pub device: Option<String>,
    pub cached_models: Vec<String>,
    pub voices: Vec<String>,
}

/// Error response
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
}

fn send_daemon_request(request: serde_json::Value) -> Result<serde_json::Value, String> {
    let mut stream = UnixStream::connect(LFM2_SOCKET_PATH).map_err(|e| {
        format!(
            "Failed to connect to LFM2 daemon: {}. Make sure the daemon is running.",
            e
        )
    })?;

    let data =
        serde_json::to_vec(&request).map_err(|e| format!("JSON serialization error: {}", e))?;
    let length = (data.len() as u32).to_be_bytes();

    stream
        .write_all(&length)
        .map_err(|e| format!("Write error: {}", e))?;
    stream
        .write_all(&data)
        .map_err(|e| format!("Write error: {}", e))?;

    let mut length_buf = [0u8; 4];
    stream
        .read_exact(&mut length_buf)
        .map_err(|e| format!("Read error: {}", e))?;
    let response_len = u32::from_be_bytes(length_buf) as usize;

    let mut response_buf = vec![0u8; response_len];
    stream
        .read_exact(&mut response_buf)
        .map_err(|e| format!("Read error: {}", e))?;

    serde_json::from_slice(&response_buf).map_err(|e| format!("JSON parse error: {}", e))
}

/// Get LFM2 daemon status
pub async fn status() -> Result<Json<LFM2StatusResponse>, (StatusCode, Json<ErrorResponse>)> {
    let request = serde_json::json!({
        "command": "status"
    });

    match send_daemon_request(request) {
        Ok(response) => {
            if let Some(error) = response.get("error").and_then(|e| e.as_str()) {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: ErrorDetail {
                            message: error.to_string(),
                        },
                    }),
                ));
            }

            Ok(Json(LFM2StatusResponse {
                status: response
                    .get("status")
                    .and_then(|s| s.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                device: response
                    .get("device")
                    .and_then(|d| d.as_str())
                    .map(|s| s.to_string()),
                cached_models: response
                    .get("cached_models")
                    .and_then(|c| c.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default(),
                voices: response
                    .get("voices")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_else(|| {
                        vec![
                            "us_male".to_string(),
                            "us_female".to_string(),
                            "uk_male".to_string(),
                            "uk_female".to_string(),
                        ]
                    }),
            }))
        }
        Err(e) => Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: ErrorDetail { message: e },
            }),
        )),
    }
}

/// Generate TTS with LFM2
pub async fn tts(
    Json(req): Json<LFM2TTSRequest>,
) -> Result<Json<LFM2TTSResponse>, (StatusCode, Json<ErrorResponse>)> {
    info!(
        "LFM2 TTS request: {} chars, voice: {}",
        req.text.len(),
        req.voice
    );

    let mut request = serde_json::json!({
        "command": "tts",
        "text": req.text,
        "voice": req.voice,
    });

    if let Some(max_tokens) = req.max_new_tokens {
        request["max_new_tokens"] = serde_json::json!(max_tokens);
    }
    if let Some(temp) = req.audio_temperature {
        request["audio_temperature"] = serde_json::json!(temp);
    }
    if let Some(top_k) = req.audio_top_k {
        request["audio_top_k"] = serde_json::json!(top_k);
    }

    match send_daemon_request(request) {
        Ok(response) => {
            if let Some(error) = response.get("error").and_then(|e| e.as_str()) {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: ErrorDetail {
                            message: error.to_string(),
                        },
                    }),
                ));
            }

            Ok(Json(LFM2TTSResponse {
                audio_base64: response
                    .get("audio_base64")
                    .and_then(|a| a.as_str())
                    .unwrap_or("")
                    .to_string(),
                sample_rate: response
                    .get("sample_rate")
                    .and_then(|s| s.as_u64())
                    .unwrap_or(24000) as u32,
                format: response
                    .get("format")
                    .and_then(|f| f.as_str())
                    .unwrap_or("wav")
                    .to_string(),
            }))
        }
        Err(e) => Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: ErrorDetail { message: e },
            }),
        )),
    }
}

/// Transcribe audio with LFM2 ASR
pub async fn asr(
    Json(req): Json<LFM2ASRRequest>,
) -> Result<Json<LFM2ASRResponse>, (StatusCode, Json<ErrorResponse>)> {
    info!("LFM2 ASR request");

    let mut request = serde_json::json!({
        "command": "asr",
        "audio_base64": req.audio_base64,
    });

    if let Some(max_tokens) = req.max_new_tokens {
        request["max_new_tokens"] = serde_json::json!(max_tokens);
    }

    match send_daemon_request(request) {
        Ok(response) => {
            if let Some(error) = response.get("error").and_then(|e| e.as_str()) {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: ErrorDetail {
                            message: error.to_string(),
                        },
                    }),
                ));
            }

            Ok(Json(LFM2ASRResponse {
                transcription: response
                    .get("transcription")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string(),
            }))
        }
        Err(e) => Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: ErrorDetail { message: e },
            }),
        )),
    }
}

/// Audio-to-audio chat with LFM2
pub async fn chat(
    Json(req): Json<LFM2AudioChatRequest>,
) -> Result<Json<LFM2AudioChatResponse>, (StatusCode, Json<ErrorResponse>)> {
    info!("LFM2 Audio Chat request");

    let mut request = serde_json::json!({
        "command": "audio_chat",
    });

    if let Some(audio) = &req.audio_base64 {
        request["audio_base64"] = serde_json::json!(audio);
    }
    if let Some(text) = &req.text {
        request["text"] = serde_json::json!(text);
    }
    if let Some(max_tokens) = req.max_new_tokens {
        request["max_new_tokens"] = serde_json::json!(max_tokens);
    }
    if let Some(temp) = req.audio_temperature {
        request["audio_temperature"] = serde_json::json!(temp);
    }
    if let Some(top_k) = req.audio_top_k {
        request["audio_top_k"] = serde_json::json!(top_k);
    }

    match send_daemon_request(request) {
        Ok(response) => {
            if let Some(error) = response.get("error").and_then(|e| e.as_str()) {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: ErrorDetail {
                            message: error.to_string(),
                        },
                    }),
                ));
            }

            Ok(Json(LFM2AudioChatResponse {
                text: response
                    .get("text")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string(),
                audio_base64: response
                    .get("audio_base64")
                    .and_then(|a| a.as_str())
                    .map(|s| s.to_string()),
                sample_rate: response
                    .get("sample_rate")
                    .and_then(|s| s.as_u64())
                    .unwrap_or(24000) as u32,
                format: response
                    .get("format")
                    .and_then(|f| f.as_str())
                    .unwrap_or("wav")
                    .to_string(),
            }))
        }
        Err(e) => Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: ErrorDetail { message: e },
            }),
        )),
    }
}
