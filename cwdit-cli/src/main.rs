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
    BinStats, FftChannelizer, GoertzelBank, IqChannelizer, RunLengthEncoder, ScanConfig, Threshold,
};
use cwdit_morse::{BootstrapDecoder, Decoded, TimingEstimator};
use cwdit_source::{AudioSource, Source, WavSource};
use rustfft::num_complex::Complex32;

#[cfg(feature = "soapy")]
use cwdit_source::SoapySource;

/// Minimum Goertzel block length, regardless of sample rate / tone.
const MIN_BLOCK_LEN: u32 = 16;

/// Default Goertzel block length multiplier: block spans roughly this many
/// cycles of the target tone.
const DEFAULT_BLOCK_CYCLES: f32 = 4.0;

/// Default absolute-envelope floor for the slicer in multi-channel mode.
const DEFAULT_MULTI_ON_FLOOR: f32 = 0.08;

/// Auto-hop target: this many envelope samples per CW dit.
const TARGET_SAMPLES_PER_DIT: f32 = 10.0;

/// Audio-rate FFT auto-size limits.
const MIN_AUTO_FFT_SIZE: usize = 128;
const MAX_AUTO_FFT_SIZE: usize = 4096;

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
    #[arg(long, default_value_t = 0.005)]
    min_peak: f32,

    /// Absolute-envelope floor below which the slicer will never turn on.
    #[arg(long)]
    on_floor: Option<f32>,

    /// Disable per-channel adaptive timing.
    #[arg(long, default_value_t = false)]
    fixed_timing: bool,

    /// Use the FFT channelizer under the hood instead of a Goertzel bank.
    /// Implied by --sdr and --scan.
    #[arg(long, default_value_t = false)]
    fft: bool,

    /// FFT length when --fft is set. Auto-selected when omitted.
    #[arg(long)]
    fft_size: Option<usize>,

    /// Hop size when --fft is set. Auto-selected when omitted.
    #[arg(long)]
    hop: Option<usize>,

    /// Scan the band for occupied bins instead of decoding fixed tones.
    /// Requires the FFT path. With --sdr the scan covers the full sampled
    /// passband; with audio it covers --scan-min-freq..--scan-max-freq.
    #[arg(long, default_value_t = false, conflicts_with = "channels")]
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

    fn force_fft(&self) -> bool {
        self.fft || self.scan || self.sdr.is_some()
    }
}

fn auto_fft_size(sample_rate: f32, wpm: f32) -> usize {
    let dit_s = 1.2 / wpm;
    let raw = sample_rate * dit_s;
    let cap = if raw >= 1.0 { raw as usize } else { 1 };
    let pow2 = prev_pow2(cap).max(MIN_AUTO_FFT_SIZE);
    pow2.min(MAX_AUTO_FFT_SIZE)
}

fn auto_hop(sample_rate: f32, wpm: f32, fft_size: usize) -> usize {
    let dit_s = 1.2 / wpm;
    let raw = (sample_rate * dit_s / TARGET_SAMPLES_PER_DIT).floor();
    let hop = if raw >= 1.0 { raw as usize } else { 1 };
    hop.clamp(1, fft_size / 2)
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

fn prev_pow2(n: usize) -> usize {
    if n < 2 {
        1
    } else {
        1usize << (usize::BITS - 1 - n.leading_zeros())
    }
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
    let sample_rate = source.sample_rate();

    let (tones, prefetch) = if args.scan {
        let fft_size = args.resolved_fft_size(sample_rate);
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

    decode_pipeline(args, source, backend, &prefetch, &tones)
}

#[cfg(feature = "soapy")]
fn decode_iq<S: Source<Sample = Complex32>>(
    args: &Args,
    mut source: S,
    center_freq: f32,
) -> Result<(), Box<dyn Error>> {
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
    decode_pipeline(args, source, Box::new(backend), &prefetch, &tones)
}

/// Drive `source` through `backend`, post-process per-channel envelopes
/// through the slicer / RLE / morse decoder, and stream output.
fn decode_pipeline<T, S>(
    args: &Args,
    mut source: S,
    mut backend: Box<dyn Backend<Input = T>>,
    prefetch: &[T],
    tones: &[f32],
) -> Result<(), Box<dyn Error>>
where
    T: Copy + Default,
    S: Source<Sample = T>,
{
    let env_rate = backend.envelope_sample_rate();
    let labels = backend.labels();
    let multi = tones.len() > 1;
    let on_floor = args.resolved_on_floor(multi);

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
    C: ChannelizerLike,
    C::Input: Copy + Default,
    S: Source<Sample = C::Input>,
{
    let sample_rate = source.sample_rate();
    let target_samples =
        (args.scan_duration * sample_rate).max(channelizer.fft_size_hint() as f32) as usize;
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

    let cfg = ScanConfig {
        peak_snr_db: args.scan_snr_db,
        max_channels: args.scan_max_channels,
        nms_radius: args.scan_nms_radius,
        min_bin: channelizer.bin_index_for(min_freq_hz).max(1),
        max_bin: Some(
            (channelizer.bin_index_for(max_freq_hz) + 1).min(channelizer.channel_count()),
        ),
        ..ScanConfig::default()
    };
    let bins = stats.detect(&cfg);
    let tones: Vec<f32> = bins.iter().map(|&b| channelizer.bin_frequency(b)).collect();

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

/// Common shape of an FFT channelizer for both real and IQ inputs.
trait ChannelizerLike {
    type Input: Copy + Default;
    fn push(&mut self, sample: Self::Input) -> Option<&[Complex32]>;
    fn channel_count(&self) -> usize;
    fn bin_frequency(&self, idx: usize) -> f32;
    fn bin_index_for(&self, freq_hz: f32) -> usize;
    fn fft_size_hint(&self) -> usize;
}

impl ChannelizerLike for FftChannelizer {
    type Input = f32;
    fn push(&mut self, sample: f32) -> Option<&[Complex32]> {
        FftChannelizer::push(self, sample)
    }
    fn channel_count(&self) -> usize {
        FftChannelizer::channel_count(self)
    }
    fn bin_frequency(&self, idx: usize) -> f32 {
        FftChannelizer::bin_frequency(self, idx)
    }
    fn bin_index_for(&self, freq_hz: f32) -> usize {
        FftChannelizer::bin_index_for(self, freq_hz)
    }
    fn fft_size_hint(&self) -> usize {
        FftChannelizer::fft_size(self)
    }
}

impl ChannelizerLike for IqChannelizer {
    type Input = Complex32;
    fn push(&mut self, sample: Complex32) -> Option<&[Complex32]> {
        IqChannelizer::push(self, sample)
    }
    fn channel_count(&self) -> usize {
        IqChannelizer::channel_count(self)
    }
    fn bin_frequency(&self, idx: usize) -> f32 {
        IqChannelizer::bin_frequency(self, idx)
    }
    fn bin_index_for(&self, freq_hz: f32) -> usize {
        IqChannelizer::bin_index_for(self, freq_hz)
    }
    fn fft_size_hint(&self) -> usize {
        IqChannelizer::fft_size(self)
    }
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
        let decoder =
            BootstrapDecoder::new(TimingEstimator::from_wpm(wpm, env_rate)).with_adapt(adapt);
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
