//! `cwdit-server` — Axum front-end for the cw-dit skimmer.
//!
//! Serves a minimal web UI at `/` and a decode-event WebSocket at `/ws`
//! for the configured WAV input.

use std::error::Error;
use std::net::SocketAddr;
use std::path::PathBuf;

use clap::Parser;
use cwdit_server::{ServerConfig, serve};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    name = "cwdit-server",
    version,
    about = "Web front-end for the cw-dit skimmer",
)]
struct Args {
    /// Path to a mono PCM WAV file containing CW.
    input: PathBuf,

    /// Target tone frequency in Hz.
    #[arg(short = 't', long, default_value_t = 700.0)]
    tone: f32,

    /// Keying rate in words per minute (PARIS convention).
    #[arg(short = 'w', long, default_value_t = 20.0)]
    wpm: f32,

    /// Listen address (host:port).
    #[arg(long, default_value = "127.0.0.1:3000")]
    bind: SocketAddr,

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
    serve(
        args.bind,
        &ServerConfig {
            input: args.input,
            tone: args.tone,
            wpm: args.wpm,
            pace_factor: args.pace_factor,
        },
    )
    .await
}
