//! `cwdit-server` — Axum front-end for the cw-dit skimmer.
//!
//! Serves a minimal web UI at `/` and a decode-event WebSocket at `/ws`
//! for the configured input — a WAV file or, with `--live`, a system
//! audio input device. Mirrors the matching subset of `cwdit-cli` flags.

use std::error::Error;
use std::net::SocketAddr;
use std::path::PathBuf;

use clap::Parser;
use cwdit_server::{Input, ServerConfig, default_web_build_dir, serve};
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
    /// Path to a mono PCM WAV file containing CW. Omit when using --live.
    #[arg(required_unless_present = "live", conflicts_with = "live")]
    input: Option<PathBuf>,

    /// Decode live audio from the default system input device. Clients
    /// join the stream in progress instead of replaying a file.
    #[arg(long, default_value_t = false)]
    live: bool,

    /// Audio input device for --live (defaults to the system default).
    #[arg(long, requires = "live")]
    device: Option<String>,

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
    /// Goertzels for fixed-tone decoding. Ignored with --scan.
    #[arg(long, default_value_t = false)]
    fft: bool,

    /// Skim continuously: re-detect stations every --scan-duration
    /// seconds, opening and closing channels as they come and go.
    /// Conflicts with --channels.
    #[arg(long, default_value_t = false, conflicts_with = "channels")]
    scan: bool,

    /// Calibration interval for --scan, in seconds. Detection re-runs
    /// every interval.
    #[arg(long, default_value_t = 3.0, requires = "scan")]
    scan_duration: f32,

    /// Minimum peak SNR (dB) against the local noise floor required to
    /// flag a bin as occupied during --scan.
    #[arg(long, default_value_t = 8.0, requires = "scan")]
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

    /// Seconds a skimmed channel may go undetected before it is closed.
    #[arg(long, default_value_t = 30.0, requires = "scan")]
    channel_timeout: f32,

    /// Listen address (host:port).
    #[arg(long, default_value = "127.0.0.1:3000")]
    bind: SocketAddr,

    /// Directory containing the built `SvelteKit` assets. Defaults to
    /// `web/build/` relative to the workspace root; pass a path here for
    /// deployments where the assets live elsewhere.
    #[arg(long)]
    web_dir: Option<PathBuf>,

    /// Playback speed multiplier for file input. 1.0 = real-time, 10.0 =
    /// ten times faster.
    #[arg(long, default_value_t = 1.0, conflicts_with = "live")]
    pace_factor: f32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let args = Args::parse();
    let web_dir = args.web_dir.or_else(default_web_build_dir);
    let input = if args.live {
        Input::LiveAudio {
            device: args.device,
        }
    } else {
        // clap guarantees `input` is present when --live is absent.
        Input::Wav(args.input.expect("input path"))
    };
    serve(
        args.bind,
        &ServerConfig {
            input,
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
            channel_timeout: args.channel_timeout,
            web_dir,
            pace_factor: args.pace_factor,
        },
    )
    .await
}
