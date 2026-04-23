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
use cwdit_dsp::{BinStats, FftChannelizer, GoertzelBank, RunLengthEncoder, ScanConfig, Threshold};
use cwdit_morse::{BootstrapDecoder, Decoded, TimingEstimator};
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

/// Auto-hop target: this many envelope samples per CW dit. Keeps the
/// envelope rate comfortably above the dit rate up through contest-speed CW
/// (30–40 WPM).
const TARGET_SAMPLES_PER_DIT: f32 = 10.0;

/// Minimum auto-selected FFT size. Below this, bin spacing gets too wide
/// for CW (>60 Hz at 8 kHz).
const MIN_AUTO_FFT_SIZE: usize = 128;

/// Maximum auto-selected FFT size.
const MAX_AUTO_FFT_SIZE: usize = 4096;

#[derive(Debug, Parser)]
#[command(
    name = "cwdit",
    version,
    about = "Decode narrow-band CW from a WAV file or live audio input",
    long_about = None,
)]
// Several orthogonal boolean flags (live, adapt, fft, scan) — converting to
// an enum would hurt CLI ergonomics without gaining clarity.
#[allow(clippy::struct_excessive_bools)]
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

    /// Disable per-channel adaptive timing. Keeps the dot-unit pinned to
    /// the --wpm seed instead of deriving it from the input and nudging
    /// it with observed dits. Useful for tests and clean machine-generated
    /// input at a known rate.
    #[arg(long, default_value_t = false)]
    fixed_timing: bool,

    /// Use the FFT channelizer under the hood instead of a bank of
    /// Goertzels. Tones in --channels / --tone are mapped to the nearest
    /// FFT bin. Better for many simultaneous signals; for 1–3 channels the
    /// Goertzel path is cheaper.
    #[arg(long, default_value_t = false)]
    fft: bool,

    /// FFT length (input samples per frame) when --fft is set. Omit to
    /// auto-select from --wpm so the analysis window is no longer than one
    /// dit — avoids smearing at contest speeds.
    #[arg(long, requires = "fft")]
    fft_size: Option<usize>,

    /// Hop size (input samples between FFT frames) when --fft is set. Omit
    /// to auto-select from --wpm so the envelope rate stays above the dit
    /// rate for CW up through contest speeds.
    #[arg(long, requires = "fft")]
    hop: Option<usize>,

    /// Scan the band for occupied bins instead of decoding fixed tones.
    /// Requires --fft and conflicts with --channels: the first
    /// --scan-duration seconds of input are analysed for CW-keyed bins,
    /// then every detected signal is decoded. File input only.
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

    /// Hard non-max-suppression radius in bins for --scan. Raise if
    /// legitimate CW signals closer than a few bin-spacings are routinely
    /// registering as multiple signals.
    #[arg(long, default_value_t = 3, requires = "scan")]
    scan_nms_radius: usize,

    /// Lower frequency bound (Hz) for --scan. Skips DC / sub-audible hum.
    #[arg(long, default_value_t = 300.0, requires = "scan")]
    scan_min_freq: f32,

    /// Upper frequency bound (Hz) for --scan.
    #[arg(long, default_value_t = 3000.0, requires = "scan")]
    scan_max_freq: f32,
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

    fn resolved_fft_size(&self, sample_rate: f32) -> usize {
        self.fft_size
            .unwrap_or_else(|| auto_fft_size(sample_rate, self.wpm))
    }

    fn resolved_hop(&self, sample_rate: f32, fft_size: usize) -> usize {
        let auto = auto_hop(sample_rate, self.wpm, fft_size);
        let raw = self.hop.unwrap_or(auto);
        raw.clamp(1, fft_size / 2)
    }
}

/// Largest power-of-two FFT size whose analysis window is no longer than a
/// single dit at `wpm`, clamped to `[MIN_AUTO_FFT_SIZE, MAX_AUTO_FFT_SIZE]`.
/// Keeping window ≤ dit avoids smearing adjacent elements together at
/// contest speeds.
fn auto_fft_size(sample_rate: f32, wpm: f32) -> usize {
    let dit_s = 1.2 / wpm;
    let raw = sample_rate * dit_s;
    let cap = if raw >= 1.0 { raw as usize } else { 1 };
    let pow2 = prev_pow2(cap).max(MIN_AUTO_FFT_SIZE);
    pow2.min(MAX_AUTO_FFT_SIZE)
}

/// Pick a hop size that keeps the envelope rate at
/// `TARGET_SAMPLES_PER_DIT` per dit at `wpm`, clamped to the FFT size.
fn auto_hop(sample_rate: f32, wpm: f32, fft_size: usize) -> usize {
    let dit_s = 1.2 / wpm;
    let raw = (sample_rate * dit_s / TARGET_SAMPLES_PER_DIT).floor();
    let hop = if raw >= 1.0 { raw as usize } else { 1 };
    hop.clamp(1, fft_size / 2)
}

fn prev_pow2(n: usize) -> usize {
    if n < 2 {
        1
    } else {
        1usize << (usize::BITS - 1 - n.leading_zeros())
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
        if args.scan {
            return Err("--scan requires file input; not yet supported with --live".into());
        }
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

    let (tones, prefetch) = if args.scan {
        scan_for_tones(args, &mut source)?
    } else {
        (args.tones(), Vec::new())
    };

    if tones.is_empty() {
        eprintln!("cwdit: no signals detected in scan window");
        return Ok(());
    }

    let multi = tones.len() > 1;
    let on_floor = args.resolved_on_floor(multi);

    let mut backend: Box<dyn EnvelopeProducer> = if args.fft {
        let fft_size = args.resolved_fft_size(sample_rate);
        let hop = args.resolved_hop(sample_rate, fft_size);
        Box::new(FftBackend::new(fft_size, hop, sample_rate, &tones))
    } else {
        let block_len = args.resolved_block_len(sample_rate, &tones);
        Box::new(GoertzelBackend::new(sample_rate, &tones, block_len))
    };
    let env_rate = backend.envelope_sample_rate();
    let labels = backend.labels();

    let mut chains: Vec<ChannelChain> = tones
        .iter()
        .map(|_| {
            ChannelChain::new(
                env_rate,
                args.wpm,
                args.peak_half_life,
                args.min_peak,
                on_floor,
                !args.fixed_timing,
            )
        })
        .collect();

    let mut buf = vec![0.0_f32; 4_096];
    let stdout = io::stdout();
    let mut out = stdout.lock();

    let mut env_scratch = vec![0.0_f32; tones.len()];

    let feed_sample =
        |sample: f32,
         backend: &mut Box<dyn EnvelopeProducer>,
         chains: &mut [ChannelChain],
         env_scratch: &mut [f32],
         out: &mut dyn Write|
         -> io::Result<()> {
            if backend.push(sample, env_scratch) {
                for (i, chain) in chains.iter_mut().enumerate() {
                    let events = chain.feed_envelope(env_scratch[i]);
                    if !multi {
                        for ev in events {
                            write_event(out, ev)?;
                        }
                    }
                }
            }
            Ok(())
        };

    for &sample in &prefetch {
        feed_sample(sample, &mut backend, &mut chains, &mut env_scratch, &mut out)?;
    }
    loop {
        let n = source.read(&mut buf)?;
        if n == 0 {
            break;
        }
        for &sample in &buf[..n] {
            feed_sample(sample, &mut backend, &mut chains, &mut env_scratch, &mut out)?;
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
        for (label, chain) in labels.iter().zip(&chains) {
            let wpm = chain.decoder.timing().wpm(env_rate);
            writeln!(out, "[{label}, {wpm:>4.1} WPM] {}", chain.text)?;
        }
    } else {
        writeln!(out)?;
    }
    Ok(())
}

/// Run the first `--scan-duration` seconds of `source` through an FFT
/// channelizer purely to collect per-bin statistics, detect occupied bins,
/// and return (detected-bin-centre-frequencies, buffered samples). The
/// buffered samples are replayed through the real decode pipeline by the
/// caller so the calibration window isn't lost.
fn scan_for_tones<S: Source<Sample = f32>>(
    args: &Args,
    source: &mut S,
) -> Result<(Vec<f32>, Vec<f32>), Box<dyn Error>> {
    let sample_rate = source.sample_rate();
    let fft_size = args.resolved_fft_size(sample_rate);
    let hop = args.resolved_hop(sample_rate, fft_size);
    let mut channelizer = FftChannelizer::new(fft_size, hop, sample_rate);

    let target_samples = (args.scan_duration * sample_rate).max(fft_size as f32) as usize;
    let mut prefetch: Vec<f32> = Vec::with_capacity(target_samples);
    let mut stats = BinStats::new(channelizer.channel_count());
    let mut mag_frame = vec![0.0_f32; channelizer.channel_count()];

    let mut buf = vec![0.0_f32; 4_096];
    while prefetch.len() < target_samples {
        let n = source.read(&mut buf)?;
        if n == 0 {
            break;
        }
        for &sample in &buf[..n] {
            prefetch.push(sample);
            if let Some(bins) = channelizer.push(sample) {
                for (dst, c) in mag_frame.iter_mut().zip(bins) {
                    *dst = c.norm();
                }
                stats.observe(&mag_frame);
            }
        }
    }

    let cfg = ScanConfig {
        peak_snr_db: args.scan_snr_db,
        max_channels: args.scan_max_channels,
        nms_radius: args.scan_nms_radius,
        min_bin: channelizer.bin_index_for(args.scan_min_freq).max(1),
        max_bin: Some(
            (channelizer.bin_index_for(args.scan_max_freq) + 1).min(channelizer.channel_count()),
        ),
        ..ScanConfig::default()
    };
    let bins = stats.detect(&cfg);
    let tones: Vec<f32> = bins
        .iter()
        .map(|&b| channelizer.bin_frequency(b))
        .collect();

    if tones.is_empty() {
        eprintln!(
            "cwdit: scan found no occupied bins in {:.1} s between {:.0}–{:.0} Hz",
            args.scan_duration, args.scan_min_freq, args.scan_max_freq,
        );
    } else {
        let list = tones
            .iter()
            .map(|f| format!("{f:.1} Hz"))
            .collect::<Vec<_>>()
            .join(", ");
        eprintln!(
            "cwdit: scan detected {} signal(s): {list}",
            tones.len()
        );
    }

    Ok((tones, prefetch))
}

/// Narrow trait over whatever DSP front end produces one envelope sample
/// per channel. Lets `decode` drive either a `GoertzelBank` or an
/// `FftChannelizer` through the same loop.
trait EnvelopeProducer {
    /// Feed one input sample. When a new envelope frame is ready, fill
    /// `envs` (length equal to the channel count) and return `true`.
    fn push(&mut self, sample: f32, envs: &mut [f32]) -> bool;
    fn envelope_sample_rate(&self) -> f32;
    fn labels(&self) -> Vec<String>;
}

struct GoertzelBackend {
    bank: GoertzelBank,
    tones: Vec<f32>,
    env_rate: f32,
}

impl GoertzelBackend {
    fn new(sample_rate: f32, tones: &[f32], block_len: u32) -> Self {
        Self {
            bank: GoertzelBank::new(tones, sample_rate, block_len),
            tones: tones.to_vec(),
            env_rate: sample_rate / block_len as f32,
        }
    }
}

impl EnvelopeProducer for GoertzelBackend {
    fn push(&mut self, sample: f32, envs: &mut [f32]) -> bool {
        if let Some(out) = self.bank.push(sample) {
            envs.copy_from_slice(out);
            true
        } else {
            false
        }
    }

    fn envelope_sample_rate(&self) -> f32 {
        self.env_rate
    }

    fn labels(&self) -> Vec<String> {
        self.tones.iter().map(|t| format!("{t:>6.0} Hz")).collect()
    }
}

struct FftBackend {
    channelizer: FftChannelizer,
    bins: Vec<usize>,
    actual_freqs: Vec<f32>,
}

impl FftBackend {
    fn new(fft_size: usize, hop: usize, sample_rate: f32, tones: &[f32]) -> Self {
        let channelizer = FftChannelizer::new(fft_size, hop, sample_rate);
        let bins: Vec<usize> = tones
            .iter()
            .map(|&t| channelizer.bin_index_for(t))
            .collect();
        let actual_freqs: Vec<f32> = bins.iter().map(|&b| channelizer.bin_frequency(b)).collect();
        Self {
            channelizer,
            bins,
            actual_freqs,
        }
    }
}

impl EnvelopeProducer for FftBackend {
    fn push(&mut self, sample: f32, envs: &mut [f32]) -> bool {
        if let Some(frame) = self.channelizer.push(sample) {
            for (slot, &bin) in envs.iter_mut().zip(&self.bins) {
                *slot = frame[bin].norm();
            }
            true
        } else {
            false
        }
    }

    fn envelope_sample_rate(&self) -> f32 {
        self.channelizer.output_sample_rate()
    }

    fn labels(&self) -> Vec<String> {
        self.actual_freqs
            .iter()
            .map(|f| format!("{f:>6.1} Hz"))
            .collect()
    }
}

/// Per-channel decode pipeline state.
struct ChannelChain {
    threshold: Threshold,
    rle: RunLengthEncoder,
    decoder: BootstrapDecoder,
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
        let decoder = BootstrapDecoder::new(TimingEstimator::from_wpm(wpm, env_rate))
            .with_adapt(adapt);
        Self {
            threshold,
            rle: RunLengthEncoder::new(),
            decoder,
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

fn write_event<W: Write + ?Sized>(w: &mut W, ev: Decoded) -> io::Result<()> {
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
