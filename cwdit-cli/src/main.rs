//! `cwdit` — headless command-line decoder for narrow-band CW audio.
//!
//! Wires a `cwdit-source::Source` → `cwdit-dsp` (Goertzel bank or FFT
//! channelizer → Threshold → `RunLengthEncoder`) → `cwdit-morse::Decoder`
//! and streams the decoded text to stdout. Input is a mono PCM WAV file
//! (`INPUT`), the default system audio input (`--live`), or — with
//! `--features soapy` — a `SoapySDR`-supported radio (`--sdr`). Real-audio
//! inputs flow through the original Goertzel/FFT path; IQ from `--sdr`
//! flows through a complex-input FFT channelizer that covers the radio's
//! full sampled bandwidth in one pass.

use std::error::Error;
use std::io::{self, Write};
use std::path::PathBuf;

use clap::Parser;
use cwdit_dsp::{
    BinStats, ChannelTracker, Channelizer, Debouncer, Detector, DetectorConfig, FftChannelizer,
    Goertzel, GoertzelBank, MovingAverage, RunLengthEncoder, ScanConfig, Threshold, ToneFilter,
    TrackerConfig, skim, suppress_correlated_ghosts,
};
use cwdit_morse::{BootstrapDecoder, Decoded, TimingEstimator};
use cwdit_source::{AudioSource, Source, WavSource};

#[cfg(feature = "soapy")]
use cwdit_dsp::{IqChannelizer, IqDetector, IqTone};
#[cfg(feature = "soapy")]
use cwdit_source::SoapySource;
#[cfg(feature = "soapy")]
use rustfft::num_complex::Complex32;

// Window sizing, detection thresholds, and ghost-filter policy live in
// `cwdit_dsp::skim`, shared with the server front-end.

/// A detection this close to a live channel refreshes it instead of
/// spawning a duplicate. Covers bin quantisation and fade-induced jitter
/// while staying below any station spacing worth separating.
const SKIM_MATCH_RADIUS_HZ: f32 = 25.0;

/// Default absolute-envelope floor for the slicer in multi-channel mode.
const DEFAULT_MULTI_ON_FLOOR: f32 = 0.08;

/// Default slicer noise-floor guard (`--min-peak`) for audio input, where
/// soundcard levels run near full scale.
const AUDIO_MIN_PEAK: f32 = 0.005;

/// The same guard for SDR IQ input, whose levels run ~60 dB lower: with
/// AGC targeting the whole passband, a single CW carrier's [`IqTone`]
/// envelope measures ~1e-4 keyed over ~3e-5 noise (`RSPdx`, 40 m) — the
/// audio-scale
/// guard sits far above the signal and mutes every channel. 1e-6 stays
/// below real noise but above numerical dust.
const IQ_MIN_PEAK: f32 = 1e-6;

/// IQ-rate FFT auto-size targets bin spacing rather than dit width: at SDR
/// rates the dit-width rule would pick FFTs too coarse to resolve adjacent
/// CW signals (e.g. 245 Hz spacing at 1 Msps with N = 4096).
#[cfg(feature = "soapy")]
const TARGET_IQ_BIN_SPACING_HZ: f32 = 100.0;
#[cfg(feature = "soapy")]
const MIN_AUTO_IQ_FFT_SIZE: usize = 4096;
#[cfg(feature = "soapy")]
const MAX_AUTO_IQ_FFT_SIZE: usize = 65_536;
#[cfg(feature = "soapy")]
const TARGET_IQ_ENVELOPE_RATE_HZ: f32 = 250.0;

/// Default RF sample rate when `--rf-rate` is omitted. 1.024 Msps is the
/// most common rate that all supported drivers (`RTL-SDR`, `SDRplay`) accept.
#[cfg(feature = "soapy")]
const DEFAULT_RF_SAMPLE_RATE: f32 = 1_024_000.0;

#[derive(Debug, Parser)]
#[command(
    name = "cwdit",
    version,
    about = "Decode narrow-band CW from a WAV file, live audio, or an SDR",
    long_about = None,
)]
#[allow(clippy::struct_excessive_bools)]
struct Args {
    /// Path to a mono PCM WAV file containing CW. Omit when using --live or
    /// --sdr.
    #[arg(
        required_unless_present_any = ["live", "sdr"],
        conflicts_with_all = ["live", "sdr"],
    )]
    input: Option<PathBuf>,

    /// Decode live audio from the default system input device.
    #[arg(long, default_value_t = false, conflicts_with = "sdr")]
    live: bool,

    /// Audio input device for --live (defaults to the system default).
    #[arg(long, requires = "live")]
    device: Option<String>,

    /// Stream IQ from a `SoapySDR` device. Optional value is the Soapy
    /// device-args string (default: "driver=sdrplay"). Requires
    /// `--features soapy` at build time and forces the FFT path.
    #[arg(long, num_args = 0..=1, default_missing_value = "")]
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
    /// while `--freq`, `--channels`, and `--scan-*-freq` and all reported
    /// labels stay in actual-RF terms. Defaults to 0.
    #[arg(long, requires = "sdr", allow_hyphen_values = true)]
    lo_offset: Option<f32>,

    /// Target tone frequency in Hz (audio path) or RF Hz (--sdr path).
    /// Ignored when --channels or --scan is given.
    #[arg(short = 't', long, default_value_t = 700.0)]
    tone: f32,

    /// Comma-separated list of tone frequencies in Hz (audio path) or RF Hz
    /// (--sdr path), one per channel.
    #[arg(short = 'c', long, value_delimiter = ',')]
    channels: Option<Vec<f32>>,

    /// Keying rate in words per minute (PARIS convention).
    #[arg(short = 'w', long, default_value_t = 20.0)]
    wpm: f32,

    /// Goertzel block length, in input samples. Audio path only.
    #[arg(short = 'b', long)]
    block_len: Option<u32>,

    /// Peak-detector half-life for the envelope slicer, in seconds.
    #[arg(long, default_value_t = 1.0)]
    peak_half_life: f32,

    /// Minimum envelope peak used as a noise-floor guard (0.0–1.0).
    /// Defaults to 0.005 for audio input and 1e-6 for SDR IQ, whose
    /// sample levels run far lower.
    #[arg(long)]
    min_peak: Option<f32>,

    /// Absolute-envelope floor below which the slicer will never turn on.
    #[arg(long)]
    on_floor: Option<f32>,

    /// SNR gate: minimum peak/noise-floor ratio (linear) before the slicer
    /// reports key-down. 1.0 disables the gate. Default ~6 dB.
    #[arg(long)]
    snr_gate: Option<f32>,

    /// Disable per-channel adaptive timing.
    #[arg(long, default_value_t = false)]
    fixed_timing: bool,

    /// Use the FFT channelizer under the hood instead of a Goertzel bank.
    /// Implied by --sdr. With --scan it selects the one-shot
    /// calibrate-then-decode flow instead of continuous skimming.
    #[arg(long, default_value_t = false)]
    fft: bool,

    /// FFT length when --fft is set. Auto-selected when omitted.
    #[arg(long)]
    fft_size: Option<usize>,

    /// Hop size when --fft is set. Auto-selected when omitted.
    #[arg(long)]
    hop: Option<usize>,

    /// Scan the band for occupied bins instead of decoding fixed tones.
    /// Detection runs on a long-window FFT; the detected tones are then
    /// decoded by a Goertzel bank tuned to each signal (pass --fft to
    /// decode via the FFT channelizer instead). With --sdr the scan covers
    /// the full sampled passband; with audio it covers
    /// --scan-min-freq..--scan-max-freq.
    #[arg(long, default_value_t = false, conflicts_with = "channels")]
    scan: bool,

    /// Calibration interval for --scan, in seconds. On the audio path
    /// detection re-runs every interval, spawning and retiring decode
    /// channels as stations come and go; with --sdr or --fft it is a
    /// one-shot window.
    #[arg(long, default_value_t = 3.0, requires = "scan")]
    scan_duration: f32,

    /// Seconds a skimmed channel may go undetected before it is closed
    /// (audio --scan only). Generous by default: the quiet side of a QSO
    /// stays silent for its partner's whole over.
    #[arg(long, default_value_t = 30.0, requires = "scan")]
    channel_timeout: f32,

    /// Minimum peak-to-noise-floor SNR (dB) required to flag a bin as
    /// occupied during --scan. Measured against a *local* (sliding-median)
    /// noise floor, which tracks passband shaping honestly — so real
    /// signals clear it with smaller margins than against the old global
    /// floor.
    #[arg(long, default_value_t = 8.0, requires = "scan")]
    scan_snr_db: f32,

    /// Cap on the number of signals returned by --scan.
    #[arg(long, default_value_t = 32, requires = "scan")]
    scan_max_channels: usize,

    /// Hard non-max-suppression radius in bins for --scan.
    #[arg(long, default_value_t = 3, requires = "scan")]
    scan_nms_radius: usize,

    /// Lower frequency bound (Hz) for --scan. Defaults to 300 Hz on the
    /// audio path; defaults to the bottom of the SDR passband on --sdr.
    #[arg(long, requires = "scan")]
    scan_min_freq: Option<f32>,

    /// Upper frequency bound (Hz) for --scan. Defaults to 3000 Hz on the
    /// audio path; defaults to the top of the SDR passband on --sdr.
    #[arg(long, requires = "scan")]
    scan_max_freq: Option<f32>,
}

impl Args {
    fn tones(&self) -> Vec<f32> {
        self.channels.clone().unwrap_or_else(|| vec![self.tone])
    }

    fn resolved_block_len(&self, sample_rate: f32, tones: &[f32]) -> u32 {
        self.block_len.unwrap_or_else(|| {
            let lowest_tone = tones.iter().copied().fold(f32::INFINITY, f32::min);
            skim::decode_block_len(sample_rate, self.wpm, lowest_tone)
        })
    }

    /// `guard` is true for blind multi-channel lists (`--channels` with
    /// more than one entry): those get the absolute floor so a channel
    /// parked on an empty frequency doesn't decode noise. Single-tone and
    /// scan-created channels run un-floored.
    fn resolved_on_floor(&self, guard: bool) -> f32 {
        self.on_floor
            .unwrap_or(if guard { DEFAULT_MULTI_ON_FLOOR } else { 0.0 })
    }

    /// `min_peak` scaled to the input domain's envelope levels; see
    /// [`AUDIO_MIN_PEAK`] / [`IQ_MIN_PEAK`].
    fn resolved_min_peak(&self, iq: bool) -> f32 {
        self.min_peak
            .unwrap_or(if iq { IQ_MIN_PEAK } else { AUDIO_MIN_PEAK })
    }

    fn resolved_fft_size(&self, sample_rate: f32) -> usize {
        self.fft_size
            .unwrap_or_else(|| skim::decode_fft_size(sample_rate, self.wpm))
    }

    /// FFT size for the `--scan` calibration channelizer. An explicit
    /// `--fft-size` overrides both this and the decode window.
    fn resolved_scan_fft_size(&self, sample_rate: f32) -> usize {
        self.fft_size
            .unwrap_or_else(|| skim::detect_fft_size(sample_rate, self.wpm))
    }

    fn resolved_hop(&self, sample_rate: f32, fft_size: usize) -> usize {
        let auto = skim::auto_hop(sample_rate, self.wpm, fft_size);
        let raw = self.hop.unwrap_or(auto);
        raw.clamp(1, fft_size / 2)
    }

    #[cfg(feature = "soapy")]
    fn resolved_iq_fft_size(&self, sample_rate: f32) -> usize {
        self.fft_size
            .unwrap_or_else(|| auto_iq_fft_size(sample_rate))
    }

    #[cfg(feature = "soapy")]
    fn resolved_iq_hop(&self, sample_rate: f32, fft_size: usize) -> usize {
        let auto = auto_iq_hop(sample_rate);
        let raw = self.hop.unwrap_or(auto);
        raw.clamp(1, fft_size / 2)
    }

    /// Whether the *decode* backend must be the FFT channelizer. `--scan`
    /// no longer forces it: scan uses an FFT internally for detection but
    /// hands the detected tones to a Goertzel bank, which sits exactly on
    /// each signal instead of the bin grid.
    fn force_fft(&self) -> bool {
        self.fft || self.sdr.is_some()
    }
}

/// Pick an FFT size whose bin spacing is `<= TARGET_IQ_BIN_SPACING_HZ`,
/// clamped to a sane envelope-rate-aware range.
#[cfg(feature = "soapy")]
fn auto_iq_fft_size(sample_rate: f32) -> usize {
    let raw_min = (sample_rate / TARGET_IQ_BIN_SPACING_HZ).ceil() as usize;
    let pow2 = next_pow2(raw_min).max(MIN_AUTO_IQ_FFT_SIZE);
    pow2.min(MAX_AUTO_IQ_FFT_SIZE)
}

/// Pick an IQ hop sized to keep envelope rate near `TARGET_IQ_ENVELOPE_RATE_HZ`.
#[cfg(feature = "soapy")]
fn auto_iq_hop(sample_rate: f32) -> usize {
    let raw = (sample_rate / TARGET_IQ_ENVELOPE_RATE_HZ).round() as usize;
    raw.max(1)
}

#[cfg(feature = "soapy")]
fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        1usize << (usize::BITS - (n - 1).leading_zeros())
    }
}

fn main() {
    if let Err(e) = run(&Args::parse()) {
        eprintln!("cwdit: {e}");
        std::process::exit(1);
    }
}

fn run(args: &Args) -> Result<(), Box<dyn Error>> {
    if let Some(sdr_args) = args.sdr.as_deref() {
        return run_sdr(args, sdr_args);
    }
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
        decode_real(args, source)
    } else {
        let path = args
            .input
            .as_ref()
            .expect("clap requires input when --live and --sdr are absent");
        let source = WavSource::from_path(path)?;
        decode_real(args, source)
    }
}

#[cfg(not(feature = "soapy"))]
fn run_sdr(_args: &Args, _sdr_args: &str) -> Result<(), Box<dyn Error>> {
    Err("--sdr requires the cwdit-cli `soapy` feature; rebuild with `cargo build -p cwdit-cli --features soapy`".into())
}

#[cfg(feature = "soapy")]
fn run_sdr(args: &Args, sdr_args: &str) -> Result<(), Box<dyn Error>> {
    let center = args
        .freq
        .ok_or("--sdr requires --freq <RF Hz>")?;
    let rate = args.rf_rate.unwrap_or(DEFAULT_RF_SAMPLE_RATE);
    let lo_offset = args.lo_offset.unwrap_or(0.0);
    let tune_freq = center + lo_offset;
    if cfg!(debug_assertions) {
        eprintln!(
            "cwdit: warning: unoptimized debug build — at SDR sample rates the \
             DSP runs slower than real time and silently drops samples; rebuild \
             with `cargo run --release -p cwdit-cli --features soapy`"
        );
    }
    let source = SoapySource::open(sdr_args, tune_freq, rate, args.rf_gain)?;
    if lo_offset == 0.0 {
        eprintln!(
            "cwdit: SDR streaming at {:.3} Msps centred on {:.3} MHz — press Ctrl+C to stop",
            rate / 1_000_000.0,
            center / 1_000_000.0,
        );
    } else {
        eprintln!(
            "cwdit: SDR streaming at {:.3} Msps centred on {:.4} MHz (radio tuned to {:.4} MHz, LO offset {:+.3} MHz) — press Ctrl+C to stop",
            rate / 1_000_000.0,
            center / 1_000_000.0,
            tune_freq / 1_000_000.0,
            lo_offset / 1_000_000.0,
        );
    }
    decode_iq(args, source, center)
}

fn decode_real<S: Source<Sample = f32>>(args: &Args, mut source: S) -> Result<(), Box<dyn Error>> {
    // Audio-path scanning skims continuously: detection re-runs every
    // interval and channels come and go. With --fft the one-shot
    // calibrate-then-decode flow below is kept (fixed FFT decode bins
    // can't follow a changing channel list).
    if args.scan && !args.fft {
        return skim_audio(args, source);
    }
    let sample_rate = source.sample_rate();

    let (tones, prefetch) = if args.scan {
        let fft_size = args.resolved_scan_fft_size(sample_rate);
        let hop = args.resolved_hop(sample_rate, fft_size);
        let mut channelizer = FftChannelizer::new(fft_size, hop, sample_rate);
        let min_freq = args.scan_min_freq.unwrap_or(300.0);
        let max_freq = args.scan_max_freq.unwrap_or(3_000.0);
        scan_to_tones(args, &mut source, &mut channelizer, min_freq, max_freq)?
    } else {
        (args.tones(), Vec::new())
    };

    if tones.is_empty() {
        eprintln!("cwdit: no signals detected in scan window");
        return Ok(());
    }

    let backend: Box<dyn Backend<Input = f32>> = if args.force_fft() {
        let fft_size = args.resolved_fft_size(sample_rate);
        let hop = args.resolved_hop(sample_rate, fft_size);
        Box::new(FftBackend::new(fft_size, hop, sample_rate, &tones))
    } else {
        let block_len = args.resolved_block_len(sample_rate, &tones);
        Box::new(GoertzelBackend::new(sample_rate, &tones, block_len))
    };

    decode_pipeline(args, source, backend, &prefetch, &tones, args.resolved_min_peak(false))
}

#[cfg(feature = "soapy")]
fn decode_iq<S: Source<Sample = Complex32>>(
    args: &Args,
    mut source: S,
    center_freq: f32,
) -> Result<(), Box<dyn Error>> {
    // Like the audio path, --scan without --fft skims continuously:
    // detection re-runs every interval and IqTone decode channels come and
    // go. --scan --fft keeps the one-shot calibrate-then-decode flow below.
    if args.scan && !args.fft {
        return skim_iq(args, source, center_freq);
    }
    let sample_rate = source.sample_rate();
    let fft_size = args.resolved_iq_fft_size(sample_rate);
    let hop = args.resolved_iq_hop(sample_rate, fft_size);

    let (tones, prefetch) = if args.scan {
        let mut channelizer = IqChannelizer::new(fft_size, hop, sample_rate, center_freq);
        // Default scan range to the full passband minus a 5% guard at each
        // edge — drives over DC spurs near the carrier and folding noise at
        // the band edges.
        let half = sample_rate * 0.5;
        let min_freq = args.scan_min_freq.unwrap_or(center_freq - half * 0.95);
        let max_freq = args.scan_max_freq.unwrap_or(center_freq + half * 0.95);
        scan_to_tones(args, &mut source, &mut channelizer, min_freq, max_freq)?
    } else if let Some(channels) = &args.channels {
        (channels.clone(), Vec::new())
    } else if args.tone > 0.0 && args.channels.is_none() {
        // The default --tone of 700 Hz is meaningless on the SDR path;
        // require explicit --channels or --scan when no channels are given.
        return Err(
            "--sdr decoding requires --scan or --channels with explicit RF Hz frequencies".into(),
        );
    } else {
        (Vec::new(), Vec::new())
    };

    if tones.is_empty() {
        eprintln!("cwdit: no signals detected in scan window");
        return Ok(());
    }

    let backend = IqFftBackend::new(fft_size, hop, sample_rate, center_freq, &tones);
    decode_pipeline(args, source, Box::new(backend), &prefetch, &tones, args.resolved_min_peak(true))
}

/// Drive `source` through `backend`, post-process per-channel envelopes
/// through the slicer / RLE / morse decoder, and stream output.
fn decode_pipeline<T, S>(
    args: &Args,
    mut source: S,
    mut backend: Box<dyn Backend<Input = T>>,
    prefetch: &[T],
    tones: &[f32],
    min_peak: f32,
) -> Result<(), Box<dyn Error>>
where
    T: Copy + Default,
    S: Source<Sample = T>,
{
    let env_rate = backend.envelope_sample_rate();
    let labels = backend.labels();
    let multi = tones.len() > 1;
    // The absolute floor guards *blind* channel lists (--channels) against
    // decoding raw noise. Scan-created channels already cleared the scan's
    // SNR gate, so they run un-floored — a fixed floor is exactly what
    // silences weak-but-real signals.
    let on_floor = args.resolved_on_floor(multi && !args.scan);

    let mut chains: Vec<ChannelChain> = tones
        .iter()
        .map(|_| {
            ChannelChain::new(
                env_rate,
                args.wpm,
                args.peak_half_life,
                min_peak,
                on_floor,
                args.snr_gate,
                !args.fixed_timing,
            )
        })
        .collect();

    let mut buf: Vec<T> = vec![T::default(); 4_096];
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut env_scratch = vec![0.0_f32; tones.len()];
    let mut pending: Vec<String> = vec![String::new(); tones.len()];

    if multi {
        eprintln!("cwdit: decoding {} channels — output flushes per word break", tones.len());
    }

    for &sample in prefetch {
        feed_sample(
            sample,
            backend.as_mut(),
            &mut chains,
            &mut env_scratch,
            multi,
            &labels,
            &mut pending,
            &mut out,
        )?;
    }
    loop {
        let n = source.read(&mut buf)?;
        if n == 0 {
            break;
        }
        for &sample in &buf[..n] {
            feed_sample(
                sample,
                backend.as_mut(),
                &mut chains,
                &mut env_scratch,
                multi,
                &labels,
                &mut pending,
                &mut out,
            )?;
        }
    }
    for (i, chain) in chains.iter_mut().enumerate() {
        let events = chain.finish();
        for ev in events {
            if multi {
                write_multi_event(&mut out, &labels[i], &mut pending[i], ev)?;
            } else {
                write_event(&mut out, ev)?;
            }
        }
    }

    if multi {
        // Flush any tail text that didn't reach a word break, and print a
        // per-channel WPM summary so the user sees what the timing settled
        // on.
        for (i, chain) in chains.iter().enumerate() {
            if !pending[i].is_empty() {
                writeln!(out, "[{}] {}", labels[i], pending[i])?;
            }
            let wpm = chain.decoder.timing().wpm(env_rate);
            writeln!(out, "[{}, {wpm:>4.1} WPM] (full) {}", labels[i], chain.text)?;
        }
    } else {
        writeln!(out)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn feed_sample<T: Copy + Default>(
    sample: T,
    backend: &mut dyn Backend<Input = T>,
    chains: &mut [ChannelChain],
    env_scratch: &mut [f32],
    multi: bool,
    labels: &[String],
    pending: &mut [String],
    out: &mut dyn Write,
) -> io::Result<()> {
    if backend.push(sample, env_scratch) {
        for (i, chain) in chains.iter_mut().enumerate() {
            let events = chain.feed_envelope(env_scratch[i]);
            for ev in events {
                if multi {
                    write_multi_event(out, &labels[i], &mut pending[i], ev)?;
                } else {
                    write_event(out, ev)?;
                }
            }
        }
    }
    Ok(())
}

/// One live skimmer channel: a [`ToneFilter`] on the detected tone
/// feeding its own decode chain, with per-word buffered output.
struct LiveChannel<F: ToneFilter> {
    label: String,
    filter: F,
    chain: ChannelChain,
    pending: String,
}

impl<F: ToneFilter> LiveChannel<F> {
    fn new(label: String, filter: F, env_rate: f32, args: &Args, min_peak: f32) -> Self {
        Self {
            label,
            filter,
            chain: ChannelChain::new(
                env_rate,
                args.wpm,
                args.peak_half_life,
                min_peak,
                // Skim channels cleared the scan's SNR gate; see
                // resolved_on_floor.
                args.resolved_on_floor(false),
                args.snr_gate,
                !args.fixed_timing,
            ),
            pending: String::new(),
        }
    }

    fn feed<W: Write + ?Sized>(&mut self, sample: F::Input, out: &mut W) -> io::Result<()> {
        if let Some(env) = self.filter.push(sample) {
            for ev in self.chain.feed_envelope(env) {
                write_multi_event(out, &self.label, &mut self.pending, ev)?;
            }
        }
        Ok(())
    }

    /// Flush everything and print the channel summary line; `marker` is
    /// `(closed)` for a reaped channel, `(full)` at end of input.
    fn close<W: Write + ?Sized>(
        &mut self,
        out: &mut W,
        env_rate: f32,
        marker: &str,
    ) -> io::Result<()> {
        for ev in self.chain.finish() {
            write_multi_event(out, &self.label, &mut self.pending, ev)?;
        }
        if !self.pending.is_empty() {
            writeln!(out, "[{}] {}", self.label, self.pending)?;
            self.pending.clear();
        }
        let wpm = self.chain.decoder.timing().wpm(env_rate);
        writeln!(out, "[{}, {wpm:>4.1} WPM] {marker} {}", self.label, self.chain.text)
    }
}

/// The channelizer-specific policy a [`Skimmer`] is built around: how to
/// size and place the detector, decode filters, and labels. `AudioSkim`
/// works in audio Hz over an [`FftChannelizer`]; `IqSkim` works in RF Hz
/// over an [`IqChannelizer`], so the same skim lifecycle serves both.
trait SkimMode {
    type Chan: Channelizer;
    type Filter: ToneFilter<Input = <Self::Chan as Channelizer>::Input>;

    /// Default slicer noise-floor guard, scaled to this mode's envelope
    /// levels ([`AUDIO_MIN_PEAK`] / [`IQ_MIN_PEAK`]).
    const MIN_PEAK: f32;

    /// Scan bounds `(min, max)` in this mode's frequency units.
    fn scan_range(&self, args: &Args, sample_rate: f32) -> (f32, f32);
    /// Per-interval detector over the scan range.
    fn make_detector(
        &self,
        args: &Args,
        sample_rate: f32,
        range: (f32, f32),
    ) -> Detector<Self::Chan>;
    /// Decode integration block length, in input samples.
    fn block_len(&self, args: &Args, sample_rate: f32, min_freq: f32) -> u32;
    /// Decode filter tuned to `tone` (in this mode's frequency units).
    fn make_filter(&self, tone: f32, sample_rate: f32, block_len: u32) -> Self::Filter;
    /// Human-readable label for a detected tone.
    fn label(&self, tone: f32) -> String;
}

/// Real-audio skim: Goertzel decode filters on an [`FftChannelizer`] grid,
/// frequencies in audio Hz.
struct AudioSkim;

impl SkimMode for AudioSkim {
    type Chan = FftChannelizer;
    type Filter = Goertzel;

    const MIN_PEAK: f32 = AUDIO_MIN_PEAK;

    fn scan_range(&self, args: &Args, _sample_rate: f32) -> (f32, f32) {
        (
            args.scan_min_freq.unwrap_or(300.0),
            args.scan_max_freq.unwrap_or(3_000.0),
        )
    }

    fn make_detector(
        &self,
        args: &Args,
        sample_rate: f32,
        range: (f32, f32),
    ) -> Detector<FftChannelizer> {
        let fft_size = args.resolved_scan_fft_size(sample_rate);
        let cfg = DetectorConfig {
            fft_size,
            hop: args.resolved_hop(sample_rate, fft_size),
            min_freq_hz: range.0,
            max_freq_hz: range.1,
            snr_db: args.scan_snr_db,
            nms_radius: args.scan_nms_radius,
            max_channels: args.scan_max_channels,
            interval_s: args.scan_duration,
        };
        Detector::new(&cfg, sample_rate)
    }

    fn block_len(&self, args: &Args, sample_rate: f32, min_freq: f32) -> u32 {
        args.resolved_block_len(sample_rate, &[min_freq])
    }

    fn make_filter(&self, tone: f32, sample_rate: f32, block_len: u32) -> Goertzel {
        Goertzel::new(tone, sample_rate, block_len)
    }

    fn label(&self, tone: f32) -> String {
        format!("{tone:>6.0} Hz")
    }
}

/// IQ skim: [`IqTone`] decode filters on an [`IqChannelizer`] grid,
/// frequencies in absolute RF Hz around the tuned carrier.
#[cfg(feature = "soapy")]
struct IqSkim {
    center_freq: f32,
}

#[cfg(feature = "soapy")]
impl SkimMode for IqSkim {
    type Chan = IqChannelizer;
    type Filter = IqTone;

    const MIN_PEAK: f32 = IQ_MIN_PEAK;

    fn scan_range(&self, args: &Args, sample_rate: f32) -> (f32, f32) {
        // Default to the full passband minus a 5% guard at each edge, past
        // the DC spur near the carrier and the folding noise at the edges.
        let half = sample_rate * 0.5;
        (
            args.scan_min_freq
                .unwrap_or(self.center_freq - half * 0.95),
            args.scan_max_freq
                .unwrap_or(self.center_freq + half * 0.95),
        )
    }

    fn make_detector(
        &self,
        args: &Args,
        sample_rate: f32,
        range: (f32, f32),
    ) -> Detector<IqChannelizer> {
        // Detection sizes by bin spacing (25 Hz target), not dit width, so
        // stations working each other tens of Hz apart land in separate
        // bins; the decode filters are IqTones, not the coarse one-shot
        // decode channelizer.
        let fft_size = args.fft_size.unwrap_or_else(|| skim::detect_iq_fft_size(sample_rate));
        let hop = {
            let auto = skim::auto_hop(sample_rate, args.wpm, fft_size);
            args.hop.unwrap_or(auto).clamp(1, fft_size / 2)
        };
        let cfg = DetectorConfig {
            fft_size,
            hop,
            min_freq_hz: range.0,
            max_freq_hz: range.1,
            snr_db: args.scan_snr_db,
            nms_radius: args.scan_nms_radius,
            max_channels: args.scan_max_channels,
            interval_s: args.scan_duration,
        };
        IqDetector::new_iq(&cfg, sample_rate, self.center_freq)
    }

    fn block_len(&self, args: &Args, sample_rate: f32, _min_freq: f32) -> u32 {
        skim::iq_decode_block_len(sample_rate, args.wpm)
    }

    fn make_filter(&self, tone: f32, sample_rate: f32, block_len: u32) -> IqTone {
        IqTone::new(tone - self.center_freq, sample_rate, block_len)
    }

    fn label(&self, tone: f32) -> String {
        format!("{:>10.4} MHz", tone / 1_000_000.0)
    }
}

/// Continuous skimmer: a [`Detector`] runs per-interval detection, a
/// [`ChannelTracker`] decides lifecycle, and [`LiveChannel`]s decode. A
/// newly spawned channel replays the interval that discovered it, so its
/// first transmission is decoded from the top. The [`SkimMode`] `M`
/// selects the audio or IQ front-end.
struct Skimmer<'a, M: SkimMode> {
    args: &'a Args,
    mode: M,
    detector: Detector<M::Chan>,
    tracker: ChannelTracker,
    channels: Vec<LiveChannel<M::Filter>>,
    range: (f32, f32),
    sample_rate: f32,
    block_len: u32,
    env_rate: f32,
    total_samples: u64,
}

impl<'a, M: SkimMode> Skimmer<'a, M> {
    fn new(args: &'a Args, mode: M, sample_rate: f32) -> Self {
        let range = mode.scan_range(args, sample_rate);
        let detector = mode.make_detector(args, sample_rate, range);
        let spacing = detector.bin_spacing_hz();
        let block_len = mode.block_len(args, sample_rate, range.0);
        Skimmer {
            args,
            mode,
            detector,
            tracker: ChannelTracker::new(TrackerConfig {
                match_radius_hz: SKIM_MATCH_RADIUS_HZ.max(spacing),
                timeout_s: args.channel_timeout,
                max_channels: args.scan_max_channels,
            }),
            channels: Vec::new(),
            range,
            sample_rate,
            block_len,
            env_rate: sample_rate / block_len as f32,
            total_samples: 0,
        }
    }

    fn banner(&self) {
        eprintln!(
            "cwdit: skimming {} – {} — rescan every {:.1} s, channel timeout {:.0} s",
            self.mode.label(self.range.0).trim(),
            self.mode.label(self.range.1).trim(),
            self.args.scan_duration,
            self.args.channel_timeout,
        );
    }

    fn push_sample<W: Write + ?Sized>(
        &mut self,
        sample: <M::Chan as Channelizer>::Input,
        out: &mut W,
    ) -> io::Result<()> {
        self.total_samples += 1;
        for ch in &mut self.channels {
            ch.feed(sample, out)?;
        }
        self.detector.push(sample);
        if self.detector.interval_complete() {
            self.detection_round(out)?;
        }
        Ok(())
    }

    fn detection_round<W: Write + ?Sized>(&mut self, out: &mut W) -> io::Result<()> {
        let tones = self.detector.detect();
        let now_s = self.total_samples as f32 / self.sample_rate;
        let update = self.tracker.observe(now_s, &tones);
        for &idx in &update.reaped {
            let mut ch = self.channels.remove(idx);
            ch.close(out, self.env_rate, "(closed)")?;
            eprintln!(
                "cwdit: channel - {} after {:.0} s idle",
                ch.label.trim(),
                self.args.channel_timeout
            );
        }
        for &tone in &update.spawned {
            eprintln!("cwdit: channel + {}", self.mode.label(tone).trim());
            let filter = self.mode.make_filter(tone, self.sample_rate, self.block_len);
            let mut ch = LiveChannel::new(
                self.mode.label(tone),
                filter,
                self.env_rate,
                self.args,
                self.args.min_peak.unwrap_or(M::MIN_PEAK),
            );
            // Replay the discovery interval so the transmission that
            // triggered detection is decoded from its start.
            for &s in self.detector.interval_audio() {
                ch.feed(s, out)?;
            }
            self.channels.push(ch);
        }

        self.detector.reset_interval();
        Ok(())
    }

    fn finish<W: Write + ?Sized>(&mut self, out: &mut W) -> io::Result<()> {
        for ch in &mut self.channels {
            ch.close(out, self.env_rate, "(full)")?;
        }
        Ok(())
    }
}

/// Drive `source` through a continuous [`Skimmer`] in mode `mode`.
fn run_skimmer<M, S>(args: &Args, mode: M, mut source: S) -> Result<(), Box<dyn Error>>
where
    M: SkimMode,
    S: Source<Sample = <M::Chan as Channelizer>::Input>,
{
    let sample_rate = source.sample_rate();
    let mut skimmer = Skimmer::new(args, mode, sample_rate);
    skimmer.banner();

    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut buf: Vec<<M::Chan as Channelizer>::Input> = vec![Default::default(); 4_096];
    loop {
        let n = source.read(&mut buf)?;
        if n == 0 {
            break;
        }
        for &sample in &buf[..n] {
            skimmer.push_sample(sample, &mut out)?;
        }
    }
    skimmer.finish(&mut out)?;
    Ok(())
}

/// Continuous skim of an audio source; see [`Skimmer`].
fn skim_audio<S: Source<Sample = f32>>(args: &Args, source: S) -> Result<(), Box<dyn Error>> {
    run_skimmer(args, AudioSkim, source)
}

/// Continuous skim of an IQ source centred on `center_freq`; see [`Skimmer`].
#[cfg(feature = "soapy")]
fn skim_iq<S: Source<Sample = Complex32>>(
    args: &Args,
    source: S,
    center_freq: f32,
) -> Result<(), Box<dyn Error>> {
    run_skimmer(args, IqSkim { center_freq }, source)
}

/// Detected tone centres (Hz) plus the buffered prefetch samples that fed
/// the calibration window — replayed through the real decoder so they
/// aren't lost.
type ScanResult<T> = (Vec<f32>, Vec<T>);

/// Run the first `--scan-duration` seconds of `source` through `channelizer`
/// to gather per-bin statistics, detect occupied bins, and return the
/// detected centre frequencies plus the buffered samples.
fn scan_to_tones<C, S>(
    args: &Args,
    source: &mut S,
    channelizer: &mut C,
    min_freq_hz: f32,
    max_freq_hz: f32,
) -> Result<ScanResult<C::Input>, Box<dyn Error>>
where
    C: Channelizer,
    C::Input: Copy + Default,
    S: Source<Sample = C::Input>,
{
    let sample_rate = source.sample_rate();
    let target_samples =
        (args.scan_duration * sample_rate).max(channelizer.fft_size() as f32) as usize;
    let mut prefetch: Vec<C::Input> = Vec::with_capacity(target_samples);
    let mut stats = BinStats::new(channelizer.channel_count());
    let mut mag_frame = vec![0.0_f32; channelizer.channel_count()];

    let mut buf: Vec<C::Input> = vec![C::Input::default(); 4_096];
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

    let spacing = channelizer.bin_spacing_hz();
    let cfg = ScanConfig {
        peak_snr_db: args.scan_snr_db,
        max_channels: args.scan_max_channels,
        nms_radius: args.scan_nms_radius,
        // Peak-ratio dominance suppression can't tell a strong signal's
        // keying sidebands from a genuinely weaker neighbour (the other
        // side of a QSO sits just as close and just as far down). Disable
        // it and let the envelope-correlation ghost filter below decide.
        dominance_db: f32::INFINITY,
        floor_radius: Some(((skim::FLOOR_RADIUS_HZ / spacing).round() as usize).max(8)),
        min_bin: channelizer.bin_index_for(min_freq_hz).max(1),
        max_bin: Some(
            (channelizer.bin_index_for(max_freq_hz) + 1).min(channelizer.channel_count()),
        ),
        ..ScanConfig::default()
    };
    let candidates = stats.detect(&cfg);

    // Second pass: replay the buffered calibration audio and record each
    // candidate bin's envelope, then drop candidates that key in lockstep
    // with a much stronger neighbour — click sidebands, not stations.
    // Recording only candidate bins keeps memory flat on wide IQ scans.
    let mut history: Vec<Vec<f32>> = vec![Vec::new(); candidates.len()];
    for &sample in &prefetch {
        if let Some(bins) = channelizer.push(sample) {
            for (h, &b) in history.iter_mut().zip(&candidates) {
                h.push(bins[b].norm());
            }
        }
    }
    let ghost_bins = ((skim::GHOST_RADIUS_HZ / spacing).round() as usize).max(1);
    let (bins, ghosts) = suppress_correlated_ghosts(
        &candidates,
        &history,
        &stats,
        ghost_bins,
        skim::GHOST_MIN_DB,
        skim::GHOST_CORR,
    );
    if ghosts > 0 {
        eprintln!("cwdit: scan suppressed {ghosts} correlated sideband(s)");
    }

    // Refine each detection off the bin grid: the decoder tunes a
    // Goertzel filter to the reported frequency, so fractional-bin
    // accuracy directly improves its SNR.
    let tones: Vec<f32> = bins
        .iter()
        .map(|&b| channelizer.bin_frequency(b) + stats.peak_offset(b) * spacing)
        .collect();

    if tones.is_empty() {
        eprintln!(
            "cwdit: scan found no occupied bins in {:.1} s between {:.0}–{:.0} Hz",
            args.scan_duration, min_freq_hz, max_freq_hz,
        );
    } else {
        let list = tones
            .iter()
            .map(|f| format!("{f:.1} Hz"))
            .collect::<Vec<_>>()
            .join(", ");
        eprintln!("cwdit: scan detected {} signal(s): {list}", tones.len());
    }

    Ok((tones, prefetch))
}

/// Narrow trait over whatever DSP front end produces one envelope sample
/// per channel for one input sample.
trait Backend {
    type Input: Copy + Default;
    fn push(&mut self, sample: Self::Input, envs: &mut [f32]) -> bool;
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

impl Backend for GoertzelBackend {
    type Input = f32;
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

impl Backend for FftBackend {
    type Input = f32;
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

#[cfg(feature = "soapy")]
struct IqFftBackend {
    channelizer: IqChannelizer,
    bins: Vec<usize>,
    actual_freqs: Vec<f32>,
}

#[cfg(feature = "soapy")]
impl IqFftBackend {
    fn new(fft_size: usize, hop: usize, sample_rate: f32, center_freq: f32, tones: &[f32]) -> Self {
        let channelizer = IqChannelizer::new(fft_size, hop, sample_rate, center_freq);
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

#[cfg(feature = "soapy")]
impl Backend for IqFftBackend {
    type Input = Complex32;
    fn push(&mut self, sample: Complex32, envs: &mut [f32]) -> bool {
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
        // RF Hz can run up to many MHz — render in MHz for readability.
        self.actual_freqs
            .iter()
            .map(|f| format!("{:>10.4} MHz", f / 1_000_000.0))
            .collect()
    }
}

/// Per-channel decode pipeline state.
struct ChannelChain {
    smoother: MovingAverage,
    threshold: Threshold,
    rle: RunLengthEncoder,
    debouncer: Debouncer,
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
        snr_gate: Option<f32>,
        adapt: bool,
    ) -> Self {
        // Smooth over ~1/4 dit and absorb runs under ~1/5 dit, both from
        // the seed WPM so a channel keying up to ~3x faster than the seed
        // still keeps its dits intact. Floors of 2: below that the smoother
        // passes raw Rayleigh noise excursions and the debouncer passes
        // 1-tick glitches, and the slicer's SNR gate alone can't hold.
        let dit_ticks = 1.2 * env_rate / wpm;
        let smooth_len = ((dit_ticks / 4.0).round() as usize).clamp(2, 16);
        let min_run = ((dit_ticks / 5.0) as u32).max(2);
        let mut threshold = Threshold::new(env_rate, peak_half_life_s, min_peak);
        if on_floor > 0.0 {
            threshold = threshold.with_absolute_on_floor(on_floor);
        }
        if let Some(gate) = snr_gate {
            threshold = threshold.with_snr_gate(gate);
        }
        let decoder =
            BootstrapDecoder::new(TimingEstimator::from_wpm(wpm, env_rate)).with_adapt(adapt);
        Self {
            smoother: MovingAverage::new(smooth_len),
            threshold,
            rle: RunLengthEncoder::new(),
            debouncer: Debouncer::new(min_run),
            decoder,
            text: String::new(),
        }
    }

    fn feed_envelope(&mut self, env: f32) -> Vec<Decoded> {
        let mark = self.threshold.push(self.smoother.push(env));
        let mut out = Vec::new();
        if let Some(run) = self.rle.push(mark).and_then(|r| self.debouncer.push(r)) {
            for ev in self.decoder.push(run.mark, run.duration) {
                self.accumulate(ev);
                out.push(ev);
            }
        }
        out
    }

    fn finish(&mut self) -> Vec<Decoded> {
        let mut out = Vec::new();
        let tail = [
            self.rle.finish().and_then(|r| self.debouncer.push(r)),
            self.debouncer.finish(),
        ];
        for run in tail.into_iter().flatten() {
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

/// Multi-channel live output. Buffers characters per channel, flushes
/// `[label] word\n` at each word break so per-channel lines stay
/// uninterleaved on stdout.
fn write_multi_event<W: Write + ?Sized>(
    w: &mut W,
    label: &str,
    pending: &mut String,
    ev: Decoded,
) -> io::Result<()> {
    match ev {
        Decoded::Char(c) => pending.push(c),
        Decoded::Unknown => pending.push('?'),
        Decoded::WordBreak => {
            if !pending.is_empty() {
                writeln!(w, "[{label}] {pending}")?;
                w.flush()?;
                pending.clear();
            }
        }
    }
    Ok(())
}
