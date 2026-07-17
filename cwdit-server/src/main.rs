//! `cwdit-server` — Axum front-end for the cw-dit skimmer.
//!
//! Serves a minimal web UI at `/` and a decode-event WebSocket at `/ws`
//! for the configured input — a WAV file, a system audio input device
//! (`--live`), or a `SoapySDR` radio (`--sdr`, built with `--features
//! soapy`). Mirrors the matching subset of `cwdit-cli` flags.

use std::error::Error;
use std::net::SocketAddr;
use std::path::PathBuf;

use clap::Parser;
use cwdit_server::{Input, ServerConfig, default_web_build_dir, serve};
use tracing_subscriber::EnvFilter;

/// Default RF sample rate when `--rf-rate` is omitted. 1.024 Msps is the
/// most common rate that all supported drivers (`RTL-SDR`, `SDRplay`)
/// accept. Mirrors `cwdit-cli`.
const DEFAULT_RF_SAMPLE_RATE: f32 = 1_024_000.0;

#[derive(Debug, Parser)]
#[command(
    name = "cwdit-server",
    version,
    about = "Web front-end for the cw-dit skimmer",
)]
// Several orthogonal boolean flags (fft, scan) — mirrors cwdit-cli.
#[allow(clippy::struct_excessive_bools)]
struct Args {
    /// Path to a mono PCM WAV file containing CW. Omit when using --live
    /// or --sdr.
    #[arg(
        required_unless_present_any = ["live", "sdr"],
        conflicts_with_all = ["live", "sdr"],
    )]
    input: Option<PathBuf>,

    /// Decode live audio from the default system input device. Clients
    /// join the stream in progress instead of replaying a file.
    #[arg(long, default_value_t = false, conflicts_with = "sdr")]
    live: bool,

    /// Audio input device for --live (defaults to the system default).
    #[arg(long, requires = "live")]
    device: Option<String>,

    /// Skim IQ from a `SoapySDR` device. Optional value is the Soapy
    /// device-args string (default: "driver=sdrplay"). Requires --scan and
    /// `--features soapy` at build time.
    #[arg(long, num_args = 0..=1, default_missing_value = "", requires = "scan")]
    sdr: Option<String>,

    /// RF centre frequency in Hz. Required with --sdr.
    #[arg(long, requires = "sdr")]
    freq: Option<f32>,

    /// SDR sample rate in Hz. Defaults to 1024000 (1.024 Msps).
    #[arg(long, requires = "sdr")]
    rf_rate: Option<f32>,

    /// SDR gain in dB. Omit for hardware AGC.
    #[arg(long, requires = "sdr")]
    rf_gain: Option<f32>,

    /// Local-oscillator offset in Hz for upconverters/downconverters (e.g.
    /// Ham It Up: 125000000). The radio is tuned to `--freq + --lo-offset`,
    /// while `--freq`, `--scan-*-freq`, and all reported frequencies stay
    /// in actual-RF terms. Defaults to 0.
    #[arg(long, requires = "sdr", allow_hyphen_values = true)]
    lo_offset: Option<f32>,

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

    /// Lower frequency bound (Hz) for --scan. Defaults to 300 Hz on the
    /// audio path; defaults to the bottom of the SDR passband with --sdr
    /// (absolute RF Hz).
    #[arg(long, requires = "scan")]
    scan_min_freq: Option<f32>,

    /// Upper frequency bound (Hz) for --scan. Defaults to 3000 Hz on the
    /// audio path; defaults to the top of the SDR passband with --sdr
    /// (absolute RF Hz).
    #[arg(long, requires = "scan")]
    scan_max_freq: Option<f32>,

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
    let input = if let Some(sdr_args) = args.sdr {
        Input::Sdr {
            args: sdr_args,
            freq_hz: args.freq.ok_or("--sdr requires --freq <RF Hz>")?,
            rf_rate: args.rf_rate.unwrap_or(DEFAULT_RF_SAMPLE_RATE),
            rf_gain: args.rf_gain,
            lo_offset: args.lo_offset.unwrap_or(0.0),
        }
    } else if args.live {
        Input::LiveAudio {
            device: args.device,
        }
    } else {
        // clap guarantees `input` is present when --live and --sdr are absent.
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
