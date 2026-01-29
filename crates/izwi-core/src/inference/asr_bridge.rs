//! Qwen3-ASR bridge for speech-to-text inference
//! Connects to a persistent Python daemon for ASR model inference

use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;
use tracing::{debug, info};

use crate::error::{Error, Result};

/// Default socket path for the ASR daemon
const DEFAULT_SOCKET_PATH: &str = "/tmp/izwi_qwen3_asr_daemon.sock";

/// Request to ASR daemon
#[derive(Debug, Serialize)]
pub struct AsrRequest {
    pub command: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_base64: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}

impl Default for AsrRequest {
    fn default() -> Self {
        Self {
            command: String::new(),
            audio_base64: None,
            model_id: None,
            language: None,
        }
    }
}

/// Response from ASR daemon
#[derive(Debug, Deserialize, Clone)]
pub struct AsrResponse {
    pub transcription: Option<String>,
    pub language: Option<String>,
    pub error: Option<String>,
    pub status: Option<String>,
    pub device: Option<String>,
    pub cached_models: Option<Vec<String>>,
}

/// Qwen3-ASR bridge for calling the ASR daemon
pub struct AsrBridge {
    socket_path: PathBuf,
    daemon_script_path: PathBuf,
    python_cmd: String,
    daemon_process: Mutex<Option<Child>>,
}

impl AsrBridge {
    /// Create a new ASR bridge
    pub fn new() -> Self {
        let base_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

        Self {
            socket_path: PathBuf::from(DEFAULT_SOCKET_PATH),
            daemon_script_path: base_dir.join("scripts/qwen3_asr_daemon.py"),
            python_cmd: "python3".to_string(),
            daemon_process: Mutex::new(None),
        }
    }

    /// Check if the daemon is running
    fn is_daemon_running(&self) -> bool {
        self.socket_path.exists() && self.connect_to_daemon().is_ok()
    }

    /// Start the daemon if not running
    pub fn ensure_daemon_running(&self) -> Result<()> {
        if self.is_daemon_running() {
            debug!("ASR daemon already running");
            return Ok(());
        }

        info!("Starting ASR daemon...");

        let child = Command::new(&self.python_cmd)
            .arg(&self.daemon_script_path)
            .arg("--socket")
            .arg(&self.socket_path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| Error::InferenceError(format!("Failed to start ASR daemon: {}", e)))?;

        // Store the child process
        {
            let mut guard = self.daemon_process.lock().unwrap();
            *guard = Some(child);
        }

        // Wait for daemon to be ready (up to 10 seconds)
        for i in 0..100 {
            std::thread::sleep(Duration::from_millis(100));
            if self.socket_path.exists() {
                if let Ok(mut stream) = self.connect_to_daemon() {
                    // Send a check command to verify it's responding
                    let request = AsrRequest {
                        command: "check".to_string(),
                        ..Default::default()
                    };
                    if self.send_request(&mut stream, &request).is_ok() {
                        info!("ASR daemon started successfully");
                        return Ok(());
                    }
                }
            }
            if i % 20 == 0 {
                debug!("Waiting for ASR daemon to start... ({}/10s)", i / 10);
            }
        }

        Err(Error::InferenceError(
            "ASR daemon failed to start within 10 seconds".to_string(),
        ))
    }

    /// Stop the daemon
    pub fn stop_daemon(&self) -> Result<()> {
        if !self.is_daemon_running() {
            return Ok(());
        }

        info!("Stopping ASR daemon...");

        // Send shutdown command
        let request = AsrRequest {
            command: "shutdown".to_string(),
            ..Default::default()
        };

        if let Ok(mut stream) = self.connect_to_daemon() {
            let _ = self.send_request(&mut stream, &request);
        }

        // Wait for socket to be removed
        for _ in 0..50 {
            std::thread::sleep(Duration::from_millis(100));
            if !self.socket_path.exists() {
                info!("ASR daemon stopped");
                return Ok(());
            }
        }

        // Force kill if still running
        let mut guard = self.daemon_process.lock().unwrap();
        if let Some(mut child) = guard.take() {
            let _ = child.kill();
            let _ = child.wait();
        }

        Ok(())
    }

    /// Get daemon status
    pub fn get_status(&self) -> Result<AsrResponse> {
        let request = AsrRequest {
            command: "status".to_string(),
            ..Default::default()
        };
        self.call_daemon(&request)
    }

    /// Transcribe audio to text
    pub fn transcribe(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
    ) -> Result<AsrResponse> {
        let request = AsrRequest {
            command: "transcribe".to_string(),
            audio_base64: Some(audio_base64.to_string()),
            model_id: model_id.map(String::from),
            language: language.map(String::from),
        };
        self.call_daemon(&request)
    }

    /// Connect to the daemon socket
    fn connect_to_daemon(&self) -> Result<UnixStream> {
        UnixStream::connect(&self.socket_path)
            .map_err(|e| Error::InferenceError(format!("Failed to connect to ASR daemon: {}", e)))
    }

    /// Send a request to the daemon and receive response
    fn send_request(&self, stream: &mut UnixStream, request: &AsrRequest) -> Result<AsrResponse> {
        // Set timeouts
        stream
            .set_read_timeout(Some(Duration::from_secs(120)))
            .ok();
        stream
            .set_write_timeout(Some(Duration::from_secs(30)))
            .ok();

        // Serialize request
        let request_json = serde_json::to_vec(request)
            .map_err(|e| Error::InferenceError(format!("Failed to serialize request: {}", e)))?;

        // Send length prefix (4 bytes, big-endian)
        let length = (request_json.len() as u32).to_be_bytes();
        stream
            .write_all(&length)
            .map_err(|e| Error::InferenceError(format!("Failed to write length: {}", e)))?;

        // Send request
        stream
            .write_all(&request_json)
            .map_err(|e| Error::InferenceError(format!("Failed to write request: {}", e)))?;

        // Read response length
        let mut length_buf = [0u8; 4];
        stream
            .read_exact(&mut length_buf)
            .map_err(|e| Error::InferenceError(format!("Failed to read response length: {}", e)))?;
        let response_length = u32::from_be_bytes(length_buf) as usize;

        // Read response
        let mut response_buf = vec![0u8; response_length];
        stream
            .read_exact(&mut response_buf)
            .map_err(|e| Error::InferenceError(format!("Failed to read response: {}", e)))?;

        // Deserialize response
        let response: AsrResponse = serde_json::from_slice(&response_buf)
            .map_err(|e| Error::InferenceError(format!("Failed to parse response: {}", e)))?;

        // Check for errors in response
        if let Some(error) = &response.error {
            return Err(Error::InferenceError(error.clone()));
        }

        Ok(response)
    }

    /// Call daemon with request
    fn call_daemon(&self, request: &AsrRequest) -> Result<AsrResponse> {
        // Ensure daemon is running
        self.ensure_daemon_running()?;

        // Connect and send request
        let mut stream = self.connect_to_daemon()?;
        self.send_request(&mut stream, request)
    }
}

impl Drop for AsrBridge {
    fn drop(&mut self) {
        // Note: We don't stop the daemon on drop anymore
        // The daemon persists for better performance across requests
        // Use stop_daemon() explicitly if needed
    }
}
