//! Web front-end for the cw-dit skimmer.
//!
//! Per WebSocket connection, spawns a fresh decode pipeline and streams
//! JSON events to the client. Input is a WAV file — loaded once, replayed
//! from the top for every connection at real time scaled by `pace_factor`
//! — or a shared live capture (system audio, or IQ from a `SoapySDR`
//! radio with the `soapy` feature) that fans samples out so each
//! connection decodes the stream from the moment it joins. Audio input
//! runs single-tone, fixed multi-channel (`channels`), or continuous
//! skimming (`scan`) through [`pipeline::pump`]; SDR input always skims,
//! through [`pipeline::pump_iq`], with frequencies in absolute RF Hz.
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
use rustfft::num_complex::Complex32;
use tokio::sync::broadcast;
use tower_http::services::{ServeDir, ServeFile};

/// Default audio-path scan bounds, in Hz. SDR input defaults instead to
/// the sampled passband minus a 5 % guard at each edge.
const DEFAULT_AUDIO_SCAN_MIN_HZ: f32 = 300.0;
const DEFAULT_AUDIO_SCAN_MAX_HZ: f32 = 3_000.0;
/// Fraction of each half-passband an SDR scan covers by default — drives
/// over DC spurs near the carrier and folding noise at the band edges.
const SDR_SCAN_EDGE_GUARD: f32 = 0.95;

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
    /// Live IQ from a `SoapySDR` radio. Requires the `soapy` cargo feature
    /// at build time and scan mode — the server always skims an SDR.
    /// One capture is shared by all connections, like `LiveAudio`.
    Sdr {
        /// Soapy device-args string (e.g. `"driver=rtlsdr"`); empty uses
        /// the default driver.
        args: String,
        /// RF centre frequency in actual-RF Hz. All reported frequencies
        /// stay in actual-RF terms regardless of `lo_offset`.
        freq_hz: f32,
        /// SDR sample rate in Hz.
        rf_rate: f32,
        /// Gain in dB; `None` enables hardware AGC.
        rf_gain: Option<f32>,
        /// Local-oscillator offset in Hz for up/downconverters; the radio
        /// is tuned to `freq_hz + lo_offset`.
        lo_offset: f32,
    },
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
    /// Scan bounds in Hz — audio Hz for audio input, absolute RF Hz for
    /// SDR input. `None` picks the per-input default: 300–3000 Hz for
    /// audio, the sampled passband minus a 5 % edge guard for SDR.
    pub scan_min_freq: Option<f32>,
    pub scan_max_freq: Option<f32>,
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
/// (WAV) or a shared live capture (audio or IQ) to subscribe to.
#[derive(Clone)]
enum SharedInput {
    Replay {
        samples: Arc<Vec<f32>>,
        pace_factor: f32,
    },
    Live {
        feed: broadcast::Sender<pipeline::Chunk>,
    },
    LiveIq {
        feed: broadcast::Sender<pipeline::Chunk<Complex32>>,
        /// RF centre frequency of the IQ stream, in actual-RF Hz.
        center_freq: f32,
    },
}

/// A fresh per-connection feed, dispatched by input domain so the
/// WebSocket handler can spawn the matching pipeline.
pub enum ConnectionFeed {
    Audio(pipeline::Feed),
    Iq {
        feed: pipeline::Feed<Complex32>,
        center_freq: f32,
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
    /// A fresh feed for one connection.
    fn feed(&self) -> ConnectionFeed {
        match &self.shared {
            SharedInput::Replay {
                samples,
                pace_factor,
            } => ConnectionFeed::Audio(pipeline::Feed::Replay {
                samples: Arc::clone(samples),
                pace_factor: *pace_factor,
            }),
            SharedInput::Live { feed } => ConnectionFeed::Audio(pipeline::Feed::Live {
                rx: feed.subscribe(),
            }),
            SharedInput::LiveIq { feed, center_freq } => ConnectionFeed::Iq {
                feed: pipeline::Feed::Live {
                    rx: feed.subscribe(),
                },
                center_freq: *center_freq,
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
                cfg: Arc::new(pipeline_config(cfg, audio_scan_range(cfg))),
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
        Input::Sdr {
            args,
            freq_hz,
            rf_rate,
            rf_gain,
            lo_offset,
        } => build_sdr_app(cfg, args, *freq_hz, *rf_rate, *rf_gain, *lo_offset),
    }
}

#[cfg(not(feature = "soapy"))]
fn build_sdr_app(
    _cfg: &ServerConfig,
    _args: &str,
    _freq_hz: f32,
    _rf_rate: f32,
    _rf_gain: Option<f32>,
    _lo_offset: f32,
) -> Result<Router, BoxError> {
    Err("--sdr requires the cwdit-server `soapy` feature; rebuild with \
         `cargo build -p cwdit-server --features soapy`"
        .into())
}

#[cfg(feature = "soapy")]
fn build_sdr_app(
    cfg: &ServerConfig,
    args: &str,
    freq_hz: f32,
    rf_rate: f32,
    rf_gain: Option<f32>,
    lo_offset: f32,
) -> Result<Router, BoxError> {
    use cwdit_source::SoapySource;

    let driver = if args.trim().is_empty() {
        cwdit_source::sdr::DEFAULT_DRIVER_ARGS
    } else {
        args
    };
    if cfg!(debug_assertions) {
        tracing::warn!(
            "this is an unoptimized debug build — at SDR sample rates the DSP \
             runs slower than real time and drops most of the signal; rebuild \
             with `cargo run --release -p cwdit-server --features soapy`"
        );
    }
    let label = format!("sdr {driver} @ {:.4} MHz", freq_hz / 1_000_000.0);
    let tune_freq = freq_hz + lo_offset;
    let device_args = args.to_owned();
    build_app_from_iq_source(cfg, label, freq_hz, move || {
        SoapySource::open(&device_args, tune_freq, rf_rate, rf_gain)
    })
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
        cfg: Arc::new(pipeline_config(cfg, audio_scan_range(cfg))),
    };
    Ok(router_for(state, cfg))
}

/// [`build_app`] over an arbitrary live IQ source instead of SDR hardware
/// — the seam tests and embedders use to feed the server synthetic IQ.
/// `center_freq_hz` is the RF centre of the stream; scan bounds default to
/// the sampled passband minus a 5 % guard at each edge. IQ input always
/// scans, so `cfg.scan` must be set.
///
/// # Errors
/// Propagates the error from `open`; rejects `cfg.scan == false`.
pub fn build_app_from_iq_source<S, F>(
    cfg: &ServerConfig,
    input_label: String,
    center_freq_hz: f32,
    open: F,
) -> Result<Router, BoxError>
where
    S: Source<Sample = Complex32>,
    F: FnOnce() -> Result<S, SourceError> + Send + 'static,
{
    if !cfg.scan {
        return Err("SDR/IQ input always skims: pass --scan (fixed-tone IQ decoding \
                    is not supported in the server)"
            .into());
    }
    let (feed, sample_rate) = pipeline::spawn_capture(open)?;
    let half = sample_rate * 0.5;
    let scan_range = (
        cfg.scan_min_freq
            .unwrap_or(center_freq_hz - half * SDR_SCAN_EDGE_GUARD),
        cfg.scan_max_freq
            .unwrap_or(center_freq_hz + half * SDR_SCAN_EDGE_GUARD),
    );
    let state = AppState {
        input: input_label,
        shared: SharedInput::LiveIq {
            feed,
            center_freq: center_freq_hz,
        },
        sample_rate,
        cfg: Arc::new(pipeline_config(cfg, scan_range)),
    };
    Ok(router_for(state, cfg))
}

/// Audio-path scan bounds: the configured values, or 300–3000 Hz.
fn audio_scan_range(cfg: &ServerConfig) -> (f32, f32) {
    (
        cfg.scan_min_freq.unwrap_or(DEFAULT_AUDIO_SCAN_MIN_HZ),
        cfg.scan_max_freq.unwrap_or(DEFAULT_AUDIO_SCAN_MAX_HZ),
    )
}

fn pipeline_config(cfg: &ServerConfig, scan_range: (f32, f32)) -> pipeline::PipelineConfig {
    pipeline::PipelineConfig {
        tones: cfg.tones(),
        wpm: cfg.wpm,
        fft: cfg.fft,
        scan: cfg.scan,
        scan_duration: cfg.scan_duration,
        scan_snr_db: cfg.scan_snr_db,
        scan_max_channels: cfg.scan_max_channels,
        scan_nms_radius: cfg.scan_nms_radius,
        scan_min_freq: scan_range.0,
        scan_max_freq: scan_range.1,
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
