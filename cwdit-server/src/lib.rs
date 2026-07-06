//! Web front-end for the cw-dit skimmer.
//!
//! Per WebSocket connection, spawns a fresh decode pipeline and streams
//! JSON events to the client. Input is either a WAV file — loaded once,
//! replayed from the top for every connection at real time scaled by
//! `pace_factor` — or live audio from a system input device, where one
//! shared capture fans samples out and each connection decodes the
//! stream from the moment it joins. Single-tone, fixed multi-channel
//! (`channels`), and continuous skimming (`scan`) all run through the
//! same [`pipeline::pump`] entry point.
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
use cwdit_source::{AudioSource, Source, SourceError};
use tokio::sync::broadcast;
use tower_http::services::{ServeDir, ServeFile};

pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Embedded placeholder served at `/` when `web_dir` is missing, so a fresh
/// clone still responds with actionable instructions instead of a 404.
const MISSING_WEB_DIR_HTML: &str = include_str!("missing_web.html");

/// What the server decodes.
pub enum Input {
    /// A mono PCM WAV file. Each connection replays it from the top at
    /// `pace_factor` × real time.
    Wav(PathBuf),
    /// Live audio from a system input device (`None` = default input).
    /// One capture is shared by all connections; each decodes the stream
    /// from the moment it joins. `pace_factor` does not apply.
    LiveAudio { device: Option<String> },
}

/// Per-invocation configuration passed from the CLI into the server.
/// Mirrors the subset of `cwdit-cli` flags that make sense for a web
/// front-end.
pub struct ServerConfig {
    pub input: Input,
    pub tone: f32,
    /// When `Some`, overrides `tone` and decodes each listed frequency as a
    /// separate channel.
    pub channels: Option<Vec<f32>>,
    pub wpm: f32,
    /// Use the FFT channelizer instead of a Goertzel bank for fixed-tone
    /// decoding. Ignored in `scan` mode, which always decodes each
    /// detected station through its own Goertzel filter.
    pub fft: bool,
    /// Skim continuously: re-detect stations every `scan_duration`
    /// seconds, opening and closing channels as they come and go.
    pub scan: bool,
    pub scan_duration: f32,
    pub scan_snr_db: f32,
    pub scan_max_channels: usize,
    pub scan_nms_radius: usize,
    pub scan_min_freq: f32,
    pub scan_max_freq: f32,
    /// Seconds a skimmed channel may go undetected before it closes.
    pub channel_timeout: f32,
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

/// The sample store connections draw from: a replayable in-memory buffer
/// (WAV) or a shared live capture to subscribe to.
#[derive(Clone)]
enum SharedInput {
    Replay {
        samples: Arc<Vec<f32>>,
        pace_factor: f32,
    },
    Live {
        feed: broadcast::Sender<pipeline::Chunk>,
    },
}

/// Shared per-request application state. Cheap to clone — the sample buffer
/// is held behind an `Arc`.
#[derive(Clone)]
pub struct AppState {
    input: String,
    shared: SharedInput,
    sample_rate: f32,
    cfg: Arc<pipeline::PipelineConfig>,
}

impl AppState {
    /// A fresh [`pipeline::Feed`] for one connection.
    fn feed(&self) -> pipeline::Feed {
        match &self.shared {
            SharedInput::Replay {
                samples,
                pace_factor,
            } => pipeline::Feed::Replay {
                samples: Arc::clone(samples),
                pace_factor: *pace_factor,
            },
            SharedInput::Live { feed } => pipeline::Feed::Live {
                rx: feed.subscribe(),
            },
        }
    }
}

/// Build the `axum` router that the binary (or a test) will serve.
///
/// # Errors
/// Returns an error if the input WAV cannot be opened or decoded, or the
/// live audio device cannot be opened.
pub fn build_app(cfg: &ServerConfig) -> Result<Router, BoxError> {
    match &cfg.input {
        Input::Wav(path) => {
            let (samples, sample_rate) = pipeline::load(path)?;
            let state = AppState {
                input: path.display().to_string(),
                shared: SharedInput::Replay {
                    samples: Arc::new(samples),
                    pace_factor: cfg.pace_factor,
                },
                sample_rate,
                cfg: Arc::new(pipeline_config(cfg)),
            };
            Ok(router_for(state, cfg))
        }
        Input::LiveAudio { device } => {
            let label = device
                .as_deref()
                .map_or_else(|| "live audio".to_owned(), |d| format!("live: {d}"));
            let device = device.clone();
            build_app_from_source(cfg, label, move || {
                AudioSource::with_device(device.as_deref())
            })
        }
    }
}

/// [`build_app`] over an arbitrary live sample source instead of a
/// hardware device — the seam tests and embedders use to feed the server
/// synthetic "live" input. `open` runs on the capture thread (sources
/// need not be `Send`). `cfg.input` and `pace_factor` are ignored; the
/// source streams at whatever rate its `read` delivers.
///
/// # Errors
/// Propagates the error from `open`.
pub fn build_app_from_source<S, F>(
    cfg: &ServerConfig,
    input_label: String,
    open: F,
) -> Result<Router, BoxError>
where
    S: Source<Sample = f32>,
    F: FnOnce() -> Result<S, SourceError> + Send + 'static,
{
    let (feed, sample_rate) = pipeline::spawn_capture(open)?;
    let state = AppState {
        input: input_label,
        shared: SharedInput::Live { feed },
        sample_rate,
        cfg: Arc::new(pipeline_config(cfg)),
    };
    Ok(router_for(state, cfg))
}

fn pipeline_config(cfg: &ServerConfig) -> pipeline::PipelineConfig {
    pipeline::PipelineConfig {
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
        channel_timeout: cfg.channel_timeout,
    }
}

fn router_for(state: AppState, cfg: &ServerConfig) -> Router {
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

    router
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
