//! Configuration types for the Izwi TTS engine

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Directory to store downloaded models
    #[serde(default = "default_models_dir")]
    pub models_dir: PathBuf,

    /// Maximum batch size for inference
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,

    /// Maximum sequence length (tokens)
    #[serde(default = "default_max_sequence_length")]
    pub max_sequence_length: usize,

    /// Chunk size for streaming (in audio tokens)
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,

    /// Data type for KV cache
    #[serde(default = "default_kv_cache_dtype")]
    pub kv_cache_dtype: String,

    /// Enable Metal GPU acceleration
    #[serde(default = "default_use_metal")]
    pub use_metal: bool,

    /// Number of threads for CPU operations
    #[serde(default = "default_num_threads")]
    pub num_threads: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            models_dir: default_models_dir(),
            max_batch_size: default_max_batch_size(),
            max_sequence_length: default_max_sequence_length(),
            chunk_size: default_chunk_size(),
            kv_cache_dtype: default_kv_cache_dtype(),
            use_metal: default_use_metal(),
            num_threads: default_num_threads(),
        }
    }
}

fn default_models_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("izwi")
        .join("models")
}

fn default_max_batch_size() -> usize {
    8
}

fn default_max_sequence_length() -> usize {
    4096
}

fn default_chunk_size() -> usize {
    128
}

fn default_kv_cache_dtype() -> String {
    "float16".to_string()
}

fn default_use_metal() -> bool {
    cfg!(target_os = "macos")
}

fn default_num_threads() -> usize {
    get_num_cpus().min(8)
}

/// Model-specific configuration from config.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub architectures: Vec<String>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,

    #[serde(default)]
    pub audio_vocab_size: Option<usize>,

    #[serde(default)]
    pub num_codebooks: Option<usize>,

    #[serde(default)]
    pub audio_sample_rate: Option<usize>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            architectures: vec!["Qwen3TTSForConditionalGeneration".to_string()],
            hidden_size: 1536,
            intermediate_size: 8960,
            num_attention_heads: 12,
            num_hidden_layers: 28,
            num_key_value_heads: 2,
            vocab_size: 152064,
            max_position_embeddings: 32768,
            rope_theta: 1000000.0,
            rms_norm_eps: 1e-6,
            audio_vocab_size: Some(4096),
            num_codebooks: Some(16),
            audio_sample_rate: Some(24000),
        }
    }
}

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,

    #[serde(default = "default_port")]
    pub port: u16,

    #[serde(default = "default_cors_enabled")]
    pub cors_enabled: bool,

    #[serde(default)]
    pub cors_origins: Vec<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            cors_enabled: default_cors_enabled(),
            cors_origins: vec!["*".to_string()],
        }
    }
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    8080
}

fn default_cors_enabled() -> bool {
    true
}

fn get_num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
}
