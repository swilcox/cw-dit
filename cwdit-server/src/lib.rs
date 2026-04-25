//! Web front-end for the cw-dit skimmer.
//!
//! Loads a WAV file once, then per WebSocket connection spawns a fresh
//! decode pipeline and streams JSON events to the client as the samples
//! play through at real time (scaled by `pace_factor`). Handles
//! single-tone, fixed multi-channel (`channels`), and auto-detection
//! (`scan`) inputs via the same [`pipeline::pump`] entry point.
//!
//! The HTTP surface mounts the `SvelteKit` SPA (built into `web/build/`) at
//! `/` via a `ServeDir` fallback, falling back to a static stub when the
//! build directory is missing. The WebSocket lives at `/ws`.
//!
//! `build_app` returns an [`axum::Router`] so integration tests can drive
//! the full stack in-process without spawning a subprocess.

pub mod pipeline;
pub mod ws;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
use axum::extract::{State, WebSocketUpgrade};
use axum::response::{Html, IntoResponse};
use axum::routing::get;
use tower_http::services::{ServeDir, ServeFile};

pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Embedded placeholder served at `/` when `web_dir` is missing, so a fresh
/// clone still responds with actionable instructions instead of a 404.
const MISSING_WEB_DIR_HTML: &str = include_str!("missing_web.html");

/// Per-invocation configuration passed from the CLI into the server.
/// Mirrors the subset of `cwdit-cli` flags that make sense for a web
/// front-end (file input only — no `--live` here yet).
pub struct ServerConfig {
    pub input: PathBuf,
    pub tone: f32,
    /// When `Some`, overrides `tone` and decodes each listed frequency as a
    /// separate channel.
    pub channels: Option<Vec<f32>>,
    pub wpm: f32,
    /// Use the FFT channelizer instead of a Goertzel bank. Required for
    /// `scan`.
    pub fft: bool,
    /// Auto-detect occupied bins during the first `scan_duration` seconds.
    pub scan: bool,
    pub scan_duration: f32,
    pub scan_snr_db: f32,
    pub scan_max_channels: usize,
    pub scan_nms_radius: usize,
    pub scan_min_freq: f32,
    pub scan_max_freq: f32,
    /// Directory containing the built `SvelteKit` assets (a SPA with an
    /// `index.html` at its root). `None` serves the embedded stub page.
    pub web_dir: Option<PathBuf>,
    /// Playback speed multiplier. 1.0 = real-time, 10.0 = ten times faster.
    pub pace_factor: f32,
}

impl ServerConfig {
    fn tones(&self) -> Vec<f32> {
        self.channels.clone().unwrap_or_else(|| vec![self.tone])
    }
}

/// Shared per-request application state. Cheap to clone — the sample buffer
/// is held behind an `Arc`.
#[derive(Clone)]
pub struct AppState {
    input: String,
    samples: Arc<Vec<f32>>,
    sample_rate: f32,
    cfg: Arc<pipeline::PipelineConfig>,
    pace_factor: f32,
}

/// Build the `axum` router that the binary (or a test) will serve.
///
/// # Errors
/// Returns an error if the input WAV cannot be opened or decoded, or if
/// the configuration rejects (e.g. `scan` without `fft`).
pub fn build_app(cfg: &ServerConfig) -> Result<Router, BoxError> {
    if cfg.scan && !cfg.fft {
        return Err("--scan requires --fft".into());
    }
    let (samples, sample_rate) = pipeline::load(&cfg.input)?;
    let pipeline_cfg = pipeline::PipelineConfig {
        tones: cfg.tones(),
        wpm: cfg.wpm,
        fft: cfg.fft,
        scan: cfg.scan,
        scan_duration: cfg.scan_duration,
        scan_snr_db: cfg.scan_snr_db,
        scan_max_channels: cfg.scan_max_channels,
        scan_nms_radius: cfg.scan_nms_radius,
        scan_min_freq: cfg.scan_min_freq,
        scan_max_freq: cfg.scan_max_freq,
    };
    let state = AppState {
        input: cfg.input.display().to_string(),
        samples: Arc::new(samples),
        sample_rate,
        cfg: Arc::new(pipeline_cfg),
        pace_factor: cfg.pace_factor,
    };

    let mut router = Router::new()
        .route("/ws", get(ws_handler))
        .with_state(state);

    router = match cfg.web_dir.as_ref() {
        Some(dir) if dir.join("index.html").is_file() => {
            tracing::info!("serving web assets from {}", dir.display());
            let index = dir.join("index.html");
            let serve_dir = ServeDir::new(dir).fallback(ServeFile::new(index));
            router.fallback_service(serve_dir)
        }
        Some(dir) => {
            tracing::warn!(
                "web_dir {} has no index.html — serving built-in stub",
                dir.display()
            );
            router.fallback(missing_web)
        }
        None => router.fallback(missing_web),
    };

    Ok(router)
}

/// Bind a TCP listener and serve the app until ctrl-c.
///
/// # Errors
/// Propagates errors from `build_app`, TCP bind, and `axum::serve`.
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

/// Default location for the `SvelteKit` build output, relative to the
/// workspace root. The binary uses this when the user didn't pass
/// `--web-dir` on the command line. Returns `None` if the directory
/// doesn't contain an `index.html`.
#[must_use]
pub fn default_web_build_dir() -> Option<PathBuf> {
    // `CARGO_MANIFEST_DIR` points at `cwdit-server/` at build time. The
    // `SvelteKit` project sits one level up in `web/`.
    let default = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map_or_else(|| PathBuf::from("web/build"), |root| root.join("web/build"));
    if default.join("index.html").is_file() {
        Some(default)
    } else {
        None
    }
}

async fn missing_web() -> impl IntoResponse {
    Html(MISSING_WEB_DIR_HTML)
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| ws::handle(socket, state))
}
