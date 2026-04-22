//! `cwdit` — headless command-line decoder for narrow-band CW audio.
//!
//! Wires a `cwdit-source::Source` → `cwdit-dsp` (Goertzel bank → Threshold →
//! `RunLengthEncoder`) → `cwdit-morse::Decoder` and streams the decoded text
//! to stdout. The input is either a mono PCM WAV file (`INPUT`) or the
//! default system audio input (`--live`). In single-channel mode the
//! decoded characters stream live; in multi-channel mode each channel's
//! text is printed on its own line once the file has been fully processed
//! (file input only — live audio requires single-channel for now).

use std::error::Error;
use std::io::{self, Write};
use std::path::PathBuf;

use clap::Parser;
use cwdit_dsp::{GoertzelBank, RunLengthEncoder, Threshold};
use cwdit_morse::{Decoded, Decoder, TimingEstimator};
use cwdit_source::{AudioSource, Source, WavSource};

/// Minimum Goertzel block length, regardless of sample rate / tone.
const MIN_BLOCK_LEN: u32 = 16;

/// Default Goertzel block length multiplier: block spans roughly this many
/// cycles of the target tone. More cycles = narrower filter but lower
/// envelope time resolution.
const DEFAULT_BLOCK_CYCLES: f32 = 4.0;

/// Default absolute-envelope floor for the slicer in multi-channel mode.
/// Prevents quiet channels from triggering on sidelobe leakage from strong
/// nearby signals.
const DEFAULT_MULTI_ON_FLOOR: f32 = 0.08;

#[derive(Debug, Parser)]
#[command(
    name = "cwdit",
    version,
    about = "Decode narrow-band CW from a WAV file or live audio input",
    long_about = None,
)]
struct Args {
    /// Path to a mono PCM WAV file containing CW. Omit when using --live.
    #[arg(required_unless_present = "live", conflicts_with = "live")]
    input: Option<PathBuf>,

    /// Decode live audio from the default system input device.
    #[arg(long, default_value_t = false)]
    live: bool,

    /// Name of the audio input device to use with --live. Defaults to the
    /// system default input.
    #[arg(long, requires = "live")]
    device: Option<String>,

    /// Target tone frequency in Hz. Ignored when --channels is given.
    #[arg(short = 't', long, default_value_t = 700.0)]
    tone: f32,

    /// Comma-separated list of tone frequencies in Hz, one per channel.
    /// When supplied, decodes each channel in parallel and prints labelled
    /// per-channel output.
    #[arg(short = 'c', long, value_delimiter = ',')]
    channels: Option<Vec<f32>>,

    /// Keying rate in words per minute (PARIS convention).
    #[arg(short = 'w', long, default_value_t = 20.0)]
    wpm: f32,

    /// Goertzel block length, in input samples. Omit to auto-select based on
    /// the lowest configured tone and the sample rate.
    #[arg(short = 'b', long)]
    block_len: Option<u32>,

    /// Peak-detector half-life for the envelope slicer, in seconds.
    #[arg(long, default_value_t = 1.0)]
    peak_half_life: f32,

    /// Minimum envelope peak used as a noise-floor guard (0.0–1.0).
    #[arg(long, default_value_t = 0.005)]
    min_peak: f32,

    /// Absolute-envelope floor below which the slicer will never turn on.
    /// Defaults to 0.0 in single-channel mode and 0.08 in multi-channel
    /// mode so quiet channels reject sidelobe leakage.
    #[arg(long)]
    on_floor: Option<f32>,

    /// Let the decoder adapt its dot-unit estimate from observed dits.
    /// Off by default — a fixed estimate from --wpm is usually the right
    /// thing for clean recordings.
    #[arg(long, default_value_t = false)]
    adapt: bool,
}

impl Args {
    fn tones(&self) -> Vec<f32> {
        self.channels.clone().unwrap_or_else(|| vec![self.tone])
    }

    fn resolved_block_len(&self, sample_rate: f32, tones: &[f32]) -> u32 {
        self.block_len.unwrap_or_else(|| {
            let lowest_tone = tones.iter().copied().fold(f32::INFINITY, f32::min);
            let raw = (DEFAULT_BLOCK_CYCLES * sample_rate / lowest_tone).round() as u32;
            raw.max(MIN_BLOCK_LEN)
        })
    }

    fn resolved_on_floor(&self, multi_channel: bool) -> f32 {
        self.on_floor.unwrap_or(if multi_channel {
            DEFAULT_MULTI_ON_FLOOR
        } else {
            0.0
        })
    }
}

fn main() {
    if let Err(e) = run(&Args::parse()) {
        eprintln!("cwdit: {e}");
        std::process::exit(1);
    }
}

fn run(args: &Args) -> Result<(), Box<dyn Error>> {
    if args.live {
        if args.channels.as_ref().is_some_and(|c| c.len() > 1) {
            return Err("multi-channel live audio is not yet supported".into());
        }
        let source = AudioSource::with_device(args.device.as_deref())?;
        eprintln!(
            "cwdit: live audio at {:.0} Hz — press Ctrl+C to stop",
            source.sample_rate()
        );
        decode(args, source)
    } else {
        let path = args
            .input
            .as_ref()
            .expect("clap requires input when --live is absent");
        let source = WavSource::from_path(path)?;
        decode(args, source)
    }
}

fn decode<S: Source<Sample = f32>>(args: &Args, mut source: S) -> Result<(), Box<dyn Error>> {
    let sample_rate = source.sample_rate();
    let tones = args.tones();
    let multi = tones.len() > 1;
    let block_len = args.resolved_block_len(sample_rate, &tones);
    let env_rate = sample_rate / block_len as f32;
    let on_floor = args.resolved_on_floor(multi);

    let mut bank = GoertzelBank::new(&tones, sample_rate, block_len);
    let mut chains: Vec<ChannelChain> = tones
        .iter()
        .map(|_| {
            ChannelChain::new(
                env_rate,
                args.wpm,
                args.peak_half_life,
                args.min_peak,
                on_floor,
                args.adapt,
            )
        })
        .collect();

    let mut buf = vec![0.0_f32; 4_096];
    let stdout = io::stdout();
    let mut out = stdout.lock();

    loop {
        let n = source.read(&mut buf)?;
        if n == 0 {
            break;
        }
        for &sample in &buf[..n] {
            if let Some(envs) = bank.push(sample) {
                for (i, chain) in chains.iter_mut().enumerate() {
                    let events = chain.feed_envelope(envs[i]);
                    if !multi {
                        for ev in events {
                            write_event(&mut out, ev)?;
                        }
                    }
                }
            }
        }
    }
    for chain in &mut chains {
        let events = chain.finish();
        if !multi {
            for ev in events {
                write_event(&mut out, ev)?;
            }
        }
    }

    if multi {
        for (tone, chain) in tones.iter().zip(&chains) {
            writeln!(out, "[{tone:>6.0} Hz] {}", chain.text)?;
        }
    } else {
        writeln!(out)?;
    }
    Ok(())
}

/// Per-channel decode pipeline state.
struct ChannelChain {
    threshold: Threshold,
    rle: RunLengthEncoder,
    decoder: Decoder,
    text: String,
}

impl ChannelChain {
    fn new(
        env_rate: f32,
        wpm: f32,
        peak_half_life_s: f32,
        min_peak: f32,
        on_floor: f32,
        adapt: bool,
    ) -> Self {
        let mut threshold = Threshold::new(env_rate, peak_half_life_s, min_peak);
        if on_floor > 0.0 {
            threshold = threshold.with_absolute_on_floor(on_floor);
        }
        Self {
            threshold,
            rle: RunLengthEncoder::new(),
            decoder: Decoder::new(TimingEstimator::from_wpm(wpm, env_rate)).with_adapt(adapt),
            text: String::new(),
        }
    }

    fn feed_envelope(&mut self, env: f32) -> Vec<Decoded> {
        let mark = self.threshold.push(env);
        let mut out = Vec::new();
        if let Some(run) = self.rle.push(mark) {
            for ev in self.decoder.push(run.mark, run.duration) {
                self.accumulate(ev);
                out.push(ev);
            }
        }
        out
    }

    fn finish(&mut self) -> Vec<Decoded> {
        let mut out = Vec::new();
        if let Some(run) = self.rle.finish() {
            for ev in self.decoder.push(run.mark, run.duration) {
                self.accumulate(ev);
                out.push(ev);
            }
        }
        for ev in self.decoder.finish() {
            self.accumulate(ev);
            out.push(ev);
        }
        out
    }

    fn accumulate(&mut self, ev: Decoded) {
        match ev {
            Decoded::Char(c) => self.text.push(c),
            Decoded::WordBreak => self.text.push(' '),
            Decoded::Unknown => self.text.push('?'),
        }
    }
}

fn write_event<W: Write>(w: &mut W, ev: Decoded) -> io::Result<()> {
    match ev {
        Decoded::Char(c) => {
            write!(w, "{c}")?;
            w.flush()
        }
        Decoded::WordBreak => {
            write!(w, " ")?;
            w.flush()
        }
        Decoded::Unknown => {
            write!(w, "?")?;
            w.flush()
        }
    }
}
