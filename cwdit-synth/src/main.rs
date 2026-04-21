//! `cwdit-synth` — generate a mono CW WAV file from text.
//!
//! One shot: `cwdit-synth -o cq.wav -t "CQ DE W1AW" -f 700 -w 20`.
//! Multi-channel: `cwdit-synth -o two.wav -c "CQ DE W1AW:18:600" -c "QRZ:20:1400"`.

use std::error::Error;
use std::path::PathBuf;
use std::process;

use clap::Parser;
use cwdit_synth::{SynthOptions, Track, synth_to_path};

#[derive(Debug, Parser)]
#[command(
    name = "cwdit-synth",
    version,
    about = "Synthesise a mono CW WAV file from text",
)]
struct Args {
    /// Output WAV path.
    #[arg(short = 'o', long)]
    output: PathBuf,

    /// Message to encode. Ignored when one or more --channel flags are
    /// given.
    #[arg(short = 't', long, default_value = "CQ DE W1AW")]
    text: String,

    /// Tone frequency in Hz for the single-track form.
    #[arg(short = 'f', long, default_value_t = 700.0)]
    tone: f32,

    /// Keying speed in words per minute for the single-track form.
    #[arg(short = 'w', long, default_value_t = 20.0)]
    wpm: f32,

    /// Output sample rate in Hz.
    #[arg(short = 's', long, default_value_t = 8_000)]
    sample_rate: u32,

    /// Add a keyed channel specified as `TEXT:WPM:TONE`. Repeatable.
    /// When one or more are given, `--text/--wpm/--tone` are ignored.
    #[arg(short = 'c', long = "channel")]
    channels: Vec<String>,

    /// Lead silence before keying, in seconds.
    #[arg(long, default_value_t = 0.2)]
    lead: f32,

    /// Tail silence after keying, in seconds.
    #[arg(long, default_value_t = 0.2)]
    tail: f32,

    /// On/off ramp length in milliseconds.
    #[arg(long, default_value_t = 10.0)]
    ramp_ms: f32,

    /// Peak post-mix amplitude (0.0–1.0).
    #[arg(long, default_value_t = 0.8)]
    amplitude: f32,
}

fn main() {
    if let Err(e) = run(&Args::parse()) {
        eprintln!("cwdit-synth: {e}");
        process::exit(1);
    }
}

fn run(args: &Args) -> Result<(), Box<dyn Error>> {
    let tracks = if args.channels.is_empty() {
        vec![Track::new(args.text.clone(), args.wpm, args.tone)]
    } else {
        args.channels
            .iter()
            .map(|s| parse_channel(s))
            .collect::<Result<Vec<_>, _>>()?
    };
    let options = SynthOptions {
        sample_rate: args.sample_rate,
        lead_silence_s: args.lead,
        tail_silence_s: args.tail,
        ramp_ms: args.ramp_ms,
        amplitude: args.amplitude,
    };
    synth_to_path(&args.output, &tracks, &options)?;
    Ok(())
}

/// Parse a `TEXT:WPM:TONE` channel spec. Splits from the right so that
/// `TEXT` may contain colons (unusual in CW, but free to support).
fn parse_channel(s: &str) -> Result<Track, String> {
    let parts: Vec<&str> = s.rsplitn(3, ':').collect();
    if parts.len() != 3 {
        return Err(format!("expected TEXT:WPM:TONE, got {s:?}"));
    }
    // rsplitn yields rightmost-first.
    let tone: f32 = parts[0]
        .parse()
        .map_err(|e| format!("bad tone in {s:?}: {e}"))?;
    let wpm: f32 = parts[1]
        .parse()
        .map_err(|e| format!("bad wpm in {s:?}: {e}"))?;
    let text = parts[2].to_string();
    Ok(Track::new(text, wpm, tone))
}
