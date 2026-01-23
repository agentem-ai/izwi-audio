//! Model information and metadata

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Available Qwen3-TTS model variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelVariant {
    /// 0.6B parameter base model
    #[serde(rename = "Qwen3-TTS-12Hz-0.6B-Base")]
    Qwen3Tts12Hz06BBase,
    /// 0.6B parameter custom voice model
    #[serde(rename = "Qwen3-TTS-12Hz-0.6B-CustomVoice")]
    Qwen3Tts12Hz06BCustomVoice,
    /// 1.7B parameter base model
    #[serde(rename = "Qwen3-TTS-12Hz-1.7B-Base")]
    Qwen3Tts12Hz17BBase,
    /// 1.7B parameter custom voice model  
    #[serde(rename = "Qwen3-TTS-12Hz-1.7B-CustomVoice")]
    Qwen3Tts12Hz17BCustomVoice,
    /// 1.7B parameter voice design model
    #[serde(rename = "Qwen3-TTS-12Hz-1.7B-VoiceDesign")]
    Qwen3Tts12Hz17BVoiceDesign,
    /// Tokenizer for 12Hz codec
    #[serde(rename = "Qwen3-TTS-Tokenizer-12Hz")]
    Qwen3TtsTokenizer12Hz,
}

impl ModelVariant {
    /// Get HuggingFace repository ID
    pub fn repo_id(&self) -> &'static str {
        match self {
            Self::Qwen3Tts12Hz06BBase => "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            Self::Qwen3Tts12Hz06BCustomVoice => "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            Self::Qwen3Tts12Hz17BBase => "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            Self::Qwen3Tts12Hz17BCustomVoice => "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            Self::Qwen3Tts12Hz17BVoiceDesign => "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            Self::Qwen3TtsTokenizer12Hz => "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        }
    }

    /// Get human-readable name
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Qwen3Tts12Hz06BBase => "Qwen3-TTS 0.6B Base",
            Self::Qwen3Tts12Hz06BCustomVoice => "Qwen3-TTS 0.6B CustomVoice",
            Self::Qwen3Tts12Hz17BBase => "Qwen3-TTS 1.7B Base",
            Self::Qwen3Tts12Hz17BCustomVoice => "Qwen3-TTS 1.7B CustomVoice",
            Self::Qwen3Tts12Hz17BVoiceDesign => "Qwen3-TTS 1.7B VoiceDesign",
            Self::Qwen3TtsTokenizer12Hz => "Qwen3-TTS Tokenizer 12Hz",
        }
    }

    /// Get local directory name
    pub fn dir_name(&self) -> &'static str {
        match self {
            Self::Qwen3Tts12Hz06BBase => "Qwen3-TTS-12Hz-0.6B-Base",
            Self::Qwen3Tts12Hz06BCustomVoice => "Qwen3-TTS-12Hz-0.6B-CustomVoice",
            Self::Qwen3Tts12Hz17BBase => "Qwen3-TTS-12Hz-1.7B-Base",
            Self::Qwen3Tts12Hz17BCustomVoice => "Qwen3-TTS-12Hz-1.7B-CustomVoice",
            Self::Qwen3Tts12Hz17BVoiceDesign => "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            Self::Qwen3TtsTokenizer12Hz => "Qwen3-TTS-Tokenizer-12Hz",
        }
    }

    /// Estimated model size in bytes
    pub fn estimated_size(&self) -> u64 {
        match self {
            Self::Qwen3Tts12Hz06BBase => 1_200_000_000, // ~1.2GB
            Self::Qwen3Tts12Hz06BCustomVoice => 1_200_000_000,
            Self::Qwen3Tts12Hz17BBase => 3_400_000_000, // ~3.4GB
            Self::Qwen3Tts12Hz17BCustomVoice => 3_400_000_000,
            Self::Qwen3Tts12Hz17BVoiceDesign => 3_400_000_000,
            Self::Qwen3TtsTokenizer12Hz => 500_000_000, // ~500MB
        }
    }

    /// Memory required for inference
    pub fn memory_required_gb(&self) -> f32 {
        match self {
            Self::Qwen3Tts12Hz06BBase | Self::Qwen3Tts12Hz06BCustomVoice => 2.5,
            Self::Qwen3Tts12Hz17BBase
            | Self::Qwen3Tts12Hz17BCustomVoice
            | Self::Qwen3Tts12Hz17BVoiceDesign => 6.0,
            Self::Qwen3TtsTokenizer12Hz => 1.0,
        }
    }

    /// Whether this is a tokenizer/codec model
    pub fn is_tokenizer(&self) -> bool {
        matches!(self, Self::Qwen3TtsTokenizer12Hz)
    }

    /// Get all available variants
    pub fn all() -> &'static [ModelVariant] {
        &[
            Self::Qwen3Tts12Hz06BBase,
            Self::Qwen3Tts12Hz06BCustomVoice,
            Self::Qwen3Tts12Hz17BBase,
            Self::Qwen3Tts12Hz17BCustomVoice,
            Self::Qwen3Tts12Hz17BVoiceDesign,
            Self::Qwen3TtsTokenizer12Hz,
        ]
    }
}

impl std::fmt::Display for ModelVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// Model download/load status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelStatus {
    /// Not downloaded
    NotDownloaded,
    /// Currently downloading
    Downloading,
    /// Downloaded but not loaded
    Downloaded,
    /// Currently loading into memory
    Loading,
    /// Loaded and ready for inference
    Ready,
    /// Error state
    Error,
}

/// Complete model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub variant: ModelVariant,
    pub status: ModelStatus,
    pub local_path: Option<PathBuf>,
    pub size_bytes: Option<u64>,
    pub download_progress: Option<f32>,
    pub error_message: Option<String>,
}

impl ModelInfo {
    pub fn new(variant: ModelVariant) -> Self {
        Self {
            variant,
            status: ModelStatus::NotDownloaded,
            local_path: None,
            size_bytes: None,
            download_progress: None,
            error_message: None,
        }
    }

    pub fn with_path(mut self, path: PathBuf) -> Self {
        self.local_path = Some(path);
        self.status = ModelStatus::Downloaded;
        self
    }
}
