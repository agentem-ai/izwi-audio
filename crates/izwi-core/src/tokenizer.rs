//! Text tokenization for Qwen3-TTS

use std::path::Path;
use tokenizers::models::bpe::BPE;
use tokenizers::Tokenizer as HfTokenizer;
use tracing::{debug, info};

use crate::error::{Error, Result};

#[derive(Debug, Clone, Default)]
pub struct SpecialTokens {
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
    pub pad_id: Option<u32>,
    pub audio_start_id: Option<u32>,
    pub audio_end_id: Option<u32>,
}

pub struct Tokenizer {
    inner: HfTokenizer,
    special_tokens: SpecialTokens,
}

impl Tokenizer {
    pub fn from_path(model_dir: &Path) -> Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        if tokenizer_path.exists() {
            return Self::from_tokenizer_json(&tokenizer_path);
        }

        let vocab_path = model_dir.join("vocab.json");
        let merges_path = model_dir.join("merges.txt");

        if vocab_path.exists() && merges_path.exists() {
            return Self::from_vocab_merges(&vocab_path, &merges_path);
        }

        Err(Error::TokenizationError(format!(
            "No tokenizer found in {:?}",
            model_dir
        )))
    }

    fn from_tokenizer_json(path: &Path) -> Result<Self> {
        let inner =
            HfTokenizer::from_file(path).map_err(|e| Error::TokenizationError(e.to_string()))?;
        debug!("Loaded tokenizer from {:?}", path);
        Self::new_with_tokenizer(inner)
    }

    fn from_vocab_merges(vocab_path: &Path, merges_path: &Path) -> Result<Self> {
        info!("Loading BPE tokenizer from vocab.json + merges.txt");
        let vocab_str = vocab_path
            .to_str()
            .ok_or_else(|| Error::TokenizationError("Invalid vocab path".to_string()))?;
        let merges_str = merges_path
            .to_str()
            .ok_or_else(|| Error::TokenizationError("Invalid merges path".to_string()))?;

        let bpe = BPE::from_file(vocab_str, merges_str)
            .build()
            .map_err(|e| Error::TokenizationError(format!("BPE build failed: {}", e)))?;

        let inner = HfTokenizer::new(bpe);
        debug!("Loaded BPE tokenizer");
        Self::new_with_tokenizer(inner)
    }

    fn new_with_tokenizer(inner: HfTokenizer) -> Result<Self> {
        let special_tokens = SpecialTokens::default();

        Ok(Self {
            inner,
            special_tokens,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| Error::TokenizationError(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| Error::TokenizationError(e.to_string()))
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    pub fn format_tts_prompt(&self, text: &str, speaker: Option<&str>) -> String {
        let speaker_tag = speaker.unwrap_or("default");
        format!("[speaker:{}] {}", speaker_tag, text)
    }
}
