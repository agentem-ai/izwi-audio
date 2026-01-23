//! Izwi Core - TTS Inference Engine for Qwen3-TTS on Apple Silicon
//!
//! This crate provides the core functionality for loading and running
//! Qwen3-TTS models using MLX on Apple Silicon devices.

pub mod config;
pub mod error;
pub mod model;
pub mod audio;
pub mod inference;
pub mod tokenizer;

pub use config::EngineConfig;
pub use error::{Error, Result};
pub use model::{ModelManager, ModelInfo, ModelVariant};
pub use inference::{InferenceEngine, GenerationConfig, AudioChunk};
