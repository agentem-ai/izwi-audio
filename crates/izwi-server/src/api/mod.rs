//! API routes and handlers

mod health;
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
        // Model management
        .route("/models", get(models::list_models))
        .route("/models/:variant/download", post(models::download_model))
        .route("/models/:variant/load", post(models::load_model))
        .route("/models/:variant/unload", post(models::unload_model))
        .route("/models/:variant", get(models::get_model_info))
        // TTS generation
        .route("/tts/generate", post(tts::generate))
        .route("/tts/stream", post(tts::generate_stream));

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
