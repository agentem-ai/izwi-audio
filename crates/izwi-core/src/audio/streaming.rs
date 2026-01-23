//! Streaming audio buffer and configuration

use std::collections::VecDeque;
use tracing::debug;

/// Configuration for streaming audio generation
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Minimum tokens before starting to stream
    pub min_tokens_before_stream: usize,
    /// Maximum buffer size in tokens
    pub max_buffer_tokens: usize,
    /// Target chunk duration in milliseconds
    pub chunk_duration_ms: u32,
    /// Enable crossfade between chunks
    pub crossfade_enabled: bool,
    /// Crossfade duration in samples
    pub crossfade_samples: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            min_tokens_before_stream: 4,
            max_buffer_tokens: 256,
            chunk_duration_ms: 100,
            crossfade_enabled: true,
            crossfade_samples: 256,
        }
    }
}

/// Buffer for managing streaming audio chunks
pub struct AudioChunkBuffer {
    config: StreamingConfig,
    token_buffer: VecDeque<Vec<u32>>,
    sample_buffer: VecDeque<f32>,
    total_tokens_processed: usize,
    sample_rate: u32,
}

impl AudioChunkBuffer {
    /// Create a new buffer with configuration
    pub fn new(config: StreamingConfig, sample_rate: u32) -> Self {
        Self {
            config,
            token_buffer: VecDeque::new(),
            sample_buffer: VecDeque::new(),
            total_tokens_processed: 0,
            sample_rate,
        }
    }

    /// Add audio tokens to the buffer
    pub fn push_tokens(&mut self, tokens: Vec<u32>) {
        self.token_buffer.push_back(tokens);
    }

    /// Add decoded samples to the buffer
    pub fn push_samples(&mut self, samples: &[f32]) {
        self.sample_buffer.extend(samples);
    }

    /// Check if we have enough data to emit a chunk
    pub fn can_emit_chunk(&self) -> bool {
        let min_samples =
            (self.sample_rate as f32 * self.config.chunk_duration_ms as f32 / 1000.0) as usize;
        self.sample_buffer.len() >= min_samples
    }

    /// Check if buffer has reached minimum tokens for streaming
    pub fn ready_to_stream(&self) -> bool {
        self.token_buffer.len() >= self.config.min_tokens_before_stream
    }

    /// Take a chunk of samples from the buffer
    pub fn take_chunk(&mut self) -> Option<Vec<f32>> {
        let chunk_samples =
            (self.sample_rate as f32 * self.config.chunk_duration_ms as f32 / 1000.0) as usize;

        if self.sample_buffer.len() < chunk_samples {
            return None;
        }

        let mut chunk: Vec<f32> = self.sample_buffer.drain(..chunk_samples).collect();

        // Apply crossfade if enabled and there's more data
        if self.config.crossfade_enabled && !self.sample_buffer.is_empty() {
            self.apply_crossfade(&mut chunk);
        }

        self.total_tokens_processed += 1;
        debug!("Emitting chunk of {} samples", chunk.len());
        Some(chunk)
    }

    /// Take all remaining samples
    pub fn take_remaining(&mut self) -> Vec<f32> {
        self.sample_buffer.drain(..).collect()
    }

    /// Apply crossfade to smooth chunk boundaries
    fn apply_crossfade(&mut self, chunk: &mut [f32]) {
        let fade_len = self.config.crossfade_samples.min(chunk.len());
        let start = chunk.len() - fade_len;

        // Fade out end of current chunk
        for (i, sample) in chunk[start..].iter_mut().enumerate() {
            let fade = 1.0 - (i as f32 / fade_len as f32);
            *sample *= fade;
        }

        // Fade in start of next chunk (peek at buffer)
        for (i, &next_sample) in self.sample_buffer.iter().take(fade_len).enumerate() {
            let fade = i as f32 / fade_len as f32;
            chunk[start + i] += next_sample * fade;
        }
    }

    /// Get current buffer statistics
    pub fn stats(&self) -> BufferStats {
        BufferStats {
            tokens_buffered: self.token_buffer.len(),
            samples_buffered: self.sample_buffer.len(),
            total_processed: self.total_tokens_processed,
            buffer_duration_ms: (self.sample_buffer.len() as f32 / self.sample_rate as f32)
                * 1000.0,
        }
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.token_buffer.clear();
        self.sample_buffer.clear();
    }
}

/// Buffer statistics
#[derive(Debug, Clone)]
pub struct BufferStats {
    pub tokens_buffered: usize,
    pub samples_buffered: usize,
    pub total_processed: usize,
    pub buffer_duration_ms: f32,
}
