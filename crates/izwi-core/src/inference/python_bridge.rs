//! Python bridge for Qwen3-TTS inference
//! Connects to a persistent Python daemon for fast inference

use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;
use tracing::{debug, info, warn};

use crate::error::{Error, Result};

/// Default socket path for the TTS daemon
const DEFAULT_SOCKET_PATH: &str = "/tmp/izwi_tts_daemon.sock";

/// Request to Python inference script
#[derive(Debug, Serialize)]
pub struct PythonTTSRequest {
    pub command: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub model_path: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instruct: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_voice_clone: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ref_audio_base64: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ref_text: Option<String>,
}

impl Default for PythonTTSRequest {
    fn default() -> Self {
        Self {
            command: String::new(),
            model_path: String::new(),
            text: String::new(),
            speaker: None,
            language: None,
            instruct: None,
            use_voice_clone: None,
            ref_audio_base64: None,
            ref_text: None,
        }
    }
}

/// Response from Python inference script
#[derive(Debug, Deserialize)]
pub struct PythonTTSResponse {
    pub audio_base64: Option<String>,
    pub sample_rate: Option<u32>,
    pub format: Option<String>,
    pub error: Option<String>,
    pub status: Option<String>,
    pub device: Option<String>,
    pub cached_models: Option<Vec<String>>,
}

/// Python TTS bridge for calling qwen_tts
/// Now connects to a persistent daemon for better performance
pub struct PythonBridge {
    socket_path: PathBuf,
    daemon_script_path: PathBuf,
    fallback_script_path: PathBuf,
    python_cmd: String,
    daemon_process: Mutex<Option<Child>>,
}

impl PythonBridge {
    /// Create a new Python bridge
    pub fn new() -> Self {
        let base_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

        Self {
            socket_path: PathBuf::from(DEFAULT_SOCKET_PATH),
            daemon_script_path: base_dir.join("scripts/tts_daemon.py"),
            fallback_script_path: base_dir.join("scripts/tts_inference.py"),
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
            debug!("TTS daemon already running");
            return Ok(());
        }

        info!("Starting TTS daemon...");

        let child = Command::new(&self.python_cmd)
            .arg(&self.daemon_script_path)
            .arg("--socket")
            .arg(&self.socket_path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| Error::InferenceError(format!("Failed to start daemon: {}", e)))?;

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
                    let request = PythonTTSRequest {
                        command: "check".to_string(),
                        ..Default::default()
                    };
                    if self.send_request(&mut stream, &request).is_ok() {
                        info!("TTS daemon started successfully");
                        return Ok(());
                    }
                }
            }
            if i % 20 == 0 {
                debug!("Waiting for daemon to start... ({}/10s)", i / 10);
            }
        }

        Err(Error::InferenceError(
            "Daemon failed to start within 10 seconds".to_string(),
        ))
    }

    /// Stop the daemon
    pub fn stop_daemon(&self) -> Result<()> {
        if !self.is_daemon_running() {
            return Ok(());
        }

        info!("Stopping TTS daemon...");

        // Send shutdown command
        if let Ok(mut stream) = self.connect_to_daemon() {
            let request = PythonTTSRequest {
                command: "shutdown".to_string(),
                ..Default::default()
            };
            let _ = self.send_request(&mut stream, &request);
        }

        // Kill the process if we started it
        {
            let mut guard = self.daemon_process.lock().unwrap();
            if let Some(mut child) = guard.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }

        // Clean up socket file
        if self.socket_path.exists() {
            let _ = std::fs::remove_file(&self.socket_path);
        }

        Ok(())
    }

    /// Connect to the daemon socket
    fn connect_to_daemon(&self) -> Result<UnixStream> {
        let stream = UnixStream::connect(&self.socket_path)
            .map_err(|e| Error::InferenceError(format!("Failed to connect to daemon: {}", e)))?;

        // Set longer timeouts for voice cloning which can take minutes
        // Use 5 minutes for read (generation can be slow) and 60s for write
        stream.set_read_timeout(Some(Duration::from_secs(300))).ok();
        stream.set_write_timeout(Some(Duration::from_secs(60))).ok();

        // Ensure socket is in blocking mode
        stream.set_nonblocking(false).ok();

        Ok(stream)
    }

    /// Read exactly n bytes with retry on EAGAIN/WouldBlock
    fn read_exact_with_retry(
        stream: &mut UnixStream,
        buf: &mut [u8],
        max_retries: u32,
    ) -> std::io::Result<()> {
        let mut total_read = 0;
        let mut retries = 0;

        while total_read < buf.len() {
            match stream.read(&mut buf[total_read..]) {
                Ok(0) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "Connection closed",
                    ));
                }
                Ok(n) => {
                    total_read += n;
                    retries = 0; // Reset retries on successful read
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    retries += 1;
                    if retries > max_retries {
                        return Err(e);
                    }
                    // Wait a bit before retrying
                    std::thread::sleep(Duration::from_millis(100));
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {
                    // Retry immediately on interrupt
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// Send request to daemon and receive response
    fn send_request(
        &self,
        stream: &mut UnixStream,
        request: &PythonTTSRequest,
    ) -> Result<PythonTTSResponse> {
        let request_json = serde_json::to_string(request)
            .map_err(|e| Error::InferenceError(format!("Failed to serialize request: {}", e)))?;

        // Send length-prefixed message
        let data = request_json.as_bytes();
        let length = (data.len() as u32).to_be_bytes();

        stream
            .write_all(&length)
            .map_err(|e| Error::InferenceError(format!("Failed to write length: {}", e)))?;
        stream
            .write_all(data)
            .map_err(|e| Error::InferenceError(format!("Failed to write request: {}", e)))?;
        stream
            .flush()
            .map_err(|e| Error::InferenceError(format!("Failed to flush: {}", e)))?;

        // Read length-prefixed response with retry logic for EAGAIN
        // Allow up to 3000 retries (5 minutes at 100ms per retry)
        let mut length_buf = [0u8; 4];
        Self::read_exact_with_retry(stream, &mut length_buf, 3000)
            .map_err(|e| Error::InferenceError(format!("Failed to read response length: {}", e)))?;
        let response_len = u32::from_be_bytes(length_buf) as usize;

        let mut response_buf = vec![0u8; response_len];
        Self::read_exact_with_retry(stream, &mut response_buf, 3000)
            .map_err(|e| Error::InferenceError(format!("Failed to read response body: {}", e)))?;

        let response: PythonTTSResponse = serde_json::from_slice(&response_buf).map_err(|e| {
            Error::InferenceError(format!(
                "Failed to parse response: {} - {}",
                e,
                String::from_utf8_lossy(&response_buf)
            ))
        })?;

        Ok(response)
    }

    /// Call daemon with request, with fallback to direct Python call
    fn call_daemon(&self, request: &PythonTTSRequest) -> Result<PythonTTSResponse> {
        // Try to ensure daemon is running
        if let Err(e) = self.ensure_daemon_running() {
            warn!("Failed to start daemon, falling back to direct call: {}", e);
            return self.call_python_direct(request);
        }

        // Connect and send request
        match self.connect_to_daemon() {
            Ok(mut stream) => match self.send_request(&mut stream, request) {
                Ok(response) => Ok(response),
                Err(e) => {
                    warn!("Daemon request failed, falling back to direct call: {}", e);
                    self.call_python_direct(request)
                }
            },
            Err(e) => {
                warn!(
                    "Failed to connect to daemon, falling back to direct call: {}",
                    e
                );
                self.call_python_direct(request)
            }
        }
    }

    /// Fallback: Call Python script directly (old method)
    fn call_python_direct(&self, request: &PythonTTSRequest) -> Result<PythonTTSResponse> {
        let request_json = serde_json::to_string(request)
            .map_err(|e| Error::InferenceError(format!("Failed to serialize request: {}", e)))?;

        let mut child = Command::new(&self.python_cmd)
            .arg(&self.fallback_script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| Error::InferenceError(format!("Failed to start Python: {}", e)))?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(request_json.as_bytes())
                .map_err(|e| Error::InferenceError(format!("Failed to write to Python: {}", e)))?;
        }

        let output = child
            .wait_with_output()
            .map_err(|e| Error::InferenceError(format!("Python process failed: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::InferenceError(format!("Python error: {}", stderr)));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
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

    /// Check if Python dependencies are available
    pub fn check_dependencies(&self) -> Result<bool> {
        let request = PythonTTSRequest {
            command: "check".to_string(),
            ..Default::default()
        };

        match self.call_daemon(&request) {
            Ok(response) => {
                if response.status.as_deref() == Some("ok") {
                    if let Some(device) = response.device {
                        info!("TTS daemon ready on device: {}", device);
                    }
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

    /// Get daemon status
    pub fn get_status(&self) -> Result<PythonTTSResponse> {
        let request = PythonTTSRequest {
            command: "status".to_string(),
            ..Default::default()
        };
        self.call_daemon(&request)
    }

    /// Preload a model into the daemon cache
    pub fn preload_model(&self, model_path: &Path) -> Result<()> {
        let request = PythonTTSRequest {
            command: "preload".to_string(),
            model_path: model_path.to_string_lossy().to_string(),
            ..Default::default()
        };

        let response = self.call_daemon(&request)?;

        if let Some(err) = response.error {
            return Err(Error::InferenceError(format!(
                "Failed to preload model: {}",
                err
            )));
        }

        Ok(())
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
        self.generate_with_clone(model_path, text, speaker, language, instruct, None, None)
    }

    /// Generate TTS audio with voice cloning
    pub fn generate_with_clone(
        &self,
        model_path: &Path,
        text: &str,
        speaker: Option<&str>,
        language: Option<&str>,
        instruct: Option<&str>,
        ref_audio_base64: Option<String>,
        ref_text: Option<String>,
    ) -> Result<(Vec<f32>, u32)> {
        info!("Generating TTS for text: {}", text);
        info!(
            "Voice clone params - ref_audio: {}, ref_text: {}",
            ref_audio_base64.is_some(),
            ref_text.is_some()
        );

        let use_voice_clone = ref_audio_base64.is_some() && ref_text.is_some();
        info!("use_voice_clone: {}", use_voice_clone);

        let request = PythonTTSRequest {
            command: "generate".to_string(),
            model_path: model_path.to_string_lossy().to_string(),
            text: text.to_string(),
            speaker: speaker.map(|s| s.to_string()),
            language: language.map(|s| s.to_string()),
            instruct: instruct.map(|s| s.to_string()),
            use_voice_clone: Some(use_voice_clone),
            ref_audio_base64,
            ref_text,
        };

        let response = self.call_daemon(&request)?;

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
}

impl Default for PythonBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for PythonBridge {
    fn drop(&mut self) {
        // Note: We don't stop the daemon on drop anymore
        // The daemon persists for better performance across requests
        // Use stop_daemon() explicitly if needed
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
