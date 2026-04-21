//! Web front-end for the cw-dit skimmer.
//!
//! Loads a WAV file once, then per WebSocket connection spawns a fresh
//! decode pipeline (Goertzel → Threshold → `RunLengthEncoder` →
//! `cwdit_morse::Decoder`) and streams JSON events to the client as the
//! samples play through at real time (scaled by `pace_factor`).
//!
//! The `build_app` entry point returns an [`axum::Router`] so integration
//! tests can drive the full stack in-process without spawning a subprocess.

pub mod pipeline;
pub mod ws;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
use axum::extract::{State, WebSocketUpgrade};
use axum::response::{Html, IntoResponse};
use axum::routing::get;

pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Per-invocation configuration passed from the CLI into the server.
pub struct ServerConfig {
    pub input: PathBuf,
    pub tone: f32,
    pub wpm: f32,
    /// Playback speed multiplier. 1.0 = real-time, 10.0 = ten times faster.
    pub pace_factor: f32,
}

/// Shared per-request application state. Cheap to clone — the sample buffer
/// is held behind an `Arc`.
#[derive(Clone)]
pub struct AppState {
    meta: pipeline::Meta,
    samples: Arc<Vec<f32>>,
    sample_rate: f32,
    tone: f32,
    wpm: f32,
    pace_factor: f32,
}

impl AppState {
    #[must_use]
    pub fn meta(&self) -> &pipeline::Meta {
        &self.meta
    }
}

/// Build the `axum` router that the binary (or a test) will serve.
pub fn build_app(cfg: &ServerConfig) -> Result<Router, BoxError> {
    let (samples, sample_rate) = pipeline::load(&cfg.input)?;
    let meta = pipeline::Meta {
        input: cfg.input.display().to_string(),
        sample_rate: sample_rate as u32,
        tone: cfg.tone,
        wpm: cfg.wpm,
    };
    let state = AppState {
        meta,
        samples: Arc::new(samples),
        sample_rate,
        tone: cfg.tone,
        wpm: cfg.wpm,
        pace_factor: cfg.pace_factor,
    };
    Ok(Router::new()
        .route("/", get(index))
        .route("/ws", get(ws_handler))
        .with_state(state))
}

/// Bind a TCP listener and serve the app until ctrl-c.
pub async fn serve(bind: SocketAddr, cfg: &ServerConfig) -> Result<(), BoxError> {
    let app = build_app(cfg)?;
    let listener = tokio::net::TcpListener::bind(bind).await?;
    tracing::info!("listening on http://{}", listener.local_addr()?);
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutting down");
        })
        .await?;
    Ok(())
}

async fn index() -> impl IntoResponse {
    Html(include_str!("../static/index.html"))
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| ws::handle(socket, state))
}
