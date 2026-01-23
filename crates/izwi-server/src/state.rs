//! Application state management

use izwi_core::InferenceEngine;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<RwLock<InferenceEngine>>,
}

impl AppState {
    pub fn new(engine: InferenceEngine) -> Self {
        Self {
            engine: Arc::new(RwLock::new(engine)),
        }
    }
}
