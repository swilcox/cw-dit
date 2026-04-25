//! `cwdit-server` — Axum front-end for the cw-dit skimmer.
//!
//! Serves a minimal web UI at `/` and a decode-event WebSocket at `/ws`
//! for the configured WAV input. Mirrors the subset of `cwdit-cli` flags
//! that apply to file-based decoding.

use std::error::Error;
use std::net::SocketAddr;
use std::path::PathBuf;

use clap::Parser;
use cwdit_server::{ServerConfig, default_web_build_dir, serve};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    name = "cwdit-server",
    version,
    about = "Web front-end for the cw-dit skimmer",
)]
// Several orthogonal boolean flags (fft, scan) — mirrors cwdit-cli.
#[allow(clippy::struct_excessive_bools)]
struct Args {
    /// Path to a mono PCM WAV file containing CW.
    input: PathBuf,

    /// Target tone frequency in Hz. Ignored when --channels is given.
    #[arg(short = 't', long, default_value_t = 700.0)]
    tone: f32,

    /// Comma-separated list of tone frequencies in Hz, one per channel.
    /// When supplied, decodes each channel in parallel and streams
    /// labelled per-channel events.
    #[arg(short = 'c', long, value_delimiter = ',')]
    channels: Option<Vec<f32>>,

    /// Keying rate in words per minute (PARIS convention).
    #[arg(short = 'w', long, default_value_t = 20.0)]
    wpm: f32,

    /// Use the FFT channelizer under the hood instead of a bank of
    /// Goertzels. Required for --scan.
    #[arg(long, default_value_t = false)]
    fft: bool,

    /// Scan the band for occupied bins instead of decoding fixed tones.
    /// Requires --fft and conflicts with --channels.
    #[arg(long, default_value_t = false, requires = "fft", conflicts_with = "channels")]
    scan: bool,

    /// Calibration window for --scan, in seconds.
    #[arg(long, default_value_t = 3.0, requires = "scan")]
    scan_duration: f32,

    /// Minimum peak-to-noise-floor SNR (dB) required to flag a bin as
    /// occupied during --scan.
    #[arg(long, default_value_t = 12.0, requires = "scan")]
    scan_snr_db: f32,

    /// Cap on the number of signals returned by --scan.
    #[arg(long, default_value_t = 32, requires = "scan")]
    scan_max_channels: usize,

    /// Hard non-max-suppression radius in bins for --scan.
    #[arg(long, default_value_t = 3, requires = "scan")]
    scan_nms_radius: usize,

    /// Lower frequency bound (Hz) for --scan.
    #[arg(long, default_value_t = 300.0, requires = "scan")]
    scan_min_freq: f32,

    /// Upper frequency bound (Hz) for --scan.
    #[arg(long, default_value_t = 3000.0, requires = "scan")]
    scan_max_freq: f32,

    /// Listen address (host:port).
    #[arg(long, default_value = "127.0.0.1:3000")]
    bind: SocketAddr,

    /// Directory containing the built `SvelteKit` assets. Defaults to
    /// `web/build/` relative to the workspace root; pass a path here for
    /// deployments where the assets live elsewhere.
    #[arg(long)]
    web_dir: Option<PathBuf>,

    /// Playback speed multiplier. 1.0 = real-time, 10.0 = ten times faster.
    #[arg(long, default_value_t = 1.0)]
    pace_factor: f32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let args = Args::parse();
    let web_dir = args.web_dir.or_else(default_web_build_dir);
    serve(
        args.bind,
        &ServerConfig {
            input: args.input,
            tone: args.tone,
            channels: args.channels,
            wpm: args.wpm,
            fft: args.fft,
            scan: args.scan,
            scan_duration: args.scan_duration,
            scan_snr_db: args.scan_snr_db,
            scan_max_channels: args.scan_max_channels,
            scan_nms_radius: args.scan_nms_radius,
            scan_min_freq: args.scan_min_freq,
            scan_max_freq: args.scan_max_freq,
            web_dir,
            pace_factor: args.pace_factor,
        },
    )
    .await
}
