//! API routes and handlers

mod daemon;
mod health;
mod lfm2;
mod models;
mod tts;

use axum::{
    routing::{get, post},
    Router,
};
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

use crate::state::AppState;

/// Create the main API router
pub fn create_router(state: AppState) -> Router {
    let api_routes = Router::new()
        // Health check
        .route("/health", get(health::health_check))
        // Daemon management
        .route("/daemon/status", get(daemon::get_status))
        .route("/daemon/start", post(daemon::start_daemon))
        .route("/daemon/stop", post(daemon::stop_daemon))
        .route("/daemon/preload", post(daemon::preload_model))
        // Model management
        .route("/models", get(models::list_models))
        .route("/models/:variant/download", post(models::download_model))
        .route("/models/:variant/load", post(models::load_model))
        .route("/models/:variant/unload", post(models::unload_model))
        .route("/models/:variant", get(models::get_model_info))
        // TTS generation (Qwen3-TTS)
        .route("/tts/generate", post(tts::generate))
        .route("/tts/stream", post(tts::generate_stream))
        // LFM2-Audio endpoints
        .route("/lfm2/status", get(lfm2::status))
        .route("/lfm2/tts", post(lfm2::tts))
        .route("/lfm2/asr", post(lfm2::asr))
        .route("/lfm2/chat", post(lfm2::chat));

    Router::new()
        .nest("/api/v1", api_routes)
        // Serve static files for UI
        .fallback_service(
            tower_http::services::ServeDir::new("ui/dist")
                .fallback(tower_http::services::ServeFile::new("ui/dist/index.html")),
        )
        .layer(TraceLayer::new_for_http())
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(state)
}
