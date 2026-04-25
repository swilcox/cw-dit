//! Per-connection decode pipeline task.
//!
//! `pump` runs one of these per WebSocket, driving either a Goertzel bank
//! or an FFT channelizer into per-channel Threshold → `RunLengthEncoder` →
//! `BootstrapDecoder` chains and emitting JSON-friendly [`Event`]s. Handles
//! the three input modes of the CLI: fixed single tone, fixed multi-channel
//! (`tones.len() > 1`), and auto-detect (`scan`).

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use cwdit_dsp::{
    BinStats, FftChannelizer, GoertzelBank, RunLengthEncoder, ScanConfig, Threshold,
};
use cwdit_morse::{BootstrapDecoder, Decoded, TimingEstimator};
use cwdit_source::{Source, SourceError, WavSource};
use serde::Serialize;
use tokio::sync::mpsc;

/// Minimum Goertzel block length regardless of sample rate / tone.
const MIN_BLOCK_LEN: u32 = 16;
/// Goertzel block-length multiplier (≈ this many cycles of the lowest tone).
const DEFAULT_BLOCK_CYCLES: f32 = 4.0;
/// Envelope floor used in multi-channel mode to reject sidelobe leakage.
const DEFAULT_MULTI_ON_FLOOR: f32 = 0.08;
/// Peak-detector half-life for the envelope slicer, in seconds.
const PEAK_HALF_LIFE_S: f32 = 1.0;
/// Minimum envelope peak used as a noise-floor guard (0.0–1.0).
const MIN_PEAK: f32 = 0.005;
/// Auto-hop target: envelope samples per dit.
const TARGET_SAMPLES_PER_DIT: f32 = 10.0;
/// Auto-selected FFT size bounds.
const MIN_AUTO_FFT_SIZE: usize = 128;
const MAX_AUTO_FFT_SIZE: usize = 4096;
/// Emit a `wpm` event only when a channel's estimate has moved this much
/// since the last emission. Keeps the stream quiet during steady-state.
const WPM_EVENT_THRESHOLD: f32 = 0.5;
/// Target waterfall frame rate (frames per second). The pump decimates
/// the channelizer's native frame rate to land near this.
const SPECTRUM_TARGET_FPS: f32 = 25.0;
/// Lower edge of the dB range mapped to `u8` 0 in spectrum frames.
const SPECTRUM_DB_FLOOR: f32 = -80.0;
/// Upper edge of the dB range mapped to `u8` 255 in spectrum frames.
const SPECTRUM_DB_CEILING: f32 = 0.0;

/// Per-connection processing configuration. Clone-friendly (small, no IO).
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub tones: Vec<f32>,
    pub wpm: f32,
    pub fft: bool,
    pub scan: bool,
    pub scan_duration: f32,
    pub scan_snr_db: f32,
    pub scan_max_channels: usize,
    pub scan_nms_radius: usize,
    pub scan_min_freq: f32,
    pub scan_max_freq: f32,
}

/// Events sent to a connected WebSocket client. Serialised as
/// `{"type": "...", ...}` with `snake_case` type tags.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    /// Always the first event. Announces the stream's global parameters
    /// and which mode the UI should expect.
    Session {
        input: String,
        sample_rate: u32,
        mode: SessionMode,
    },
    /// Scan progress. Only emitted when `mode == scan`.
    ScanStatus {
        state: ScanState,
        #[serde(skip_serializing_if = "Option::is_none")]
        detected: Option<usize>,
    },
    /// Announces one channel that will subsequently produce decode
    /// events. Emitted up-front in `fixed` mode and after scan completion
    /// in `scan` mode. `id` is stable for the life of the connection.
    ChannelOpen {
        id: u32,
        freq_hz: f32,
        wpm: f32,
    },
    Char {
        channel: u32,
        ch: char,
    },
    WordBreak {
        channel: u32,
    },
    Unknown {
        channel: u32,
    },
    /// Adaptive-timing update. Emitted when a channel's estimated WPM
    /// moves more than [`WPM_EVENT_THRESHOLD`] from its last value.
    Wpm {
        channel: u32,
        wpm: f32,
    },
    /// One row of waterfall data — magnitudes for the positive half of the
    /// FFT, dB-scaled into 0..=255 and base64-encoded. Only emitted when
    /// the FFT backend is in use.
    Spectrum {
        /// Base64-encoded `Vec<u8>`, one byte per FFT bin (DC … Nyquist).
        bins: String,
        /// Centre frequency of bin 0 (always 0 Hz today).
        f_min: f32,
        /// Centre frequency of the last bin (`sample_rate / 2`).
        f_max: f32,
        /// dB value mapped to byte 0.
        db_floor: f32,
        /// dB value mapped to byte 255.
        db_ceiling: f32,
    },
    /// End of stream — no further events will be sent.
    Done,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionMode {
    Fixed,
    Scan,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScanState {
    Calibrating,
    Ready,
}

/// Read an entire mono WAV file into memory and return the sample buffer
/// alongside the file's sample rate.
///
/// # Errors
/// Propagates any error from the underlying [`WavSource`].
pub fn load(path: &Path) -> Result<(Vec<f32>, f32), SourceError> {
    let mut source = WavSource::from_path(path)?;
    let sr = source.sample_rate();
    let mut samples = Vec::new();
    let mut buf = vec![0.0_f32; 4_096];
    loop {
        let n = source.read(&mut buf)?;
        if n == 0 {
            break;
        }
        samples.extend_from_slice(&buf[..n]);
    }
    Ok((samples, sr))
}

/// Stream `samples` through a fresh decode pipeline, publishing [`Event`]s.
/// Returns when the sample buffer is exhausted or the receiver is dropped.
#[allow(clippy::too_many_lines)]
pub async fn pump(
    input: String,
    samples: Arc<Vec<f32>>,
    sample_rate: f32,
    cfg: Arc<PipelineConfig>,
    pace_factor: f32,
    tx: mpsc::Sender<Event>,
) {
    let mode = if cfg.scan {
        SessionMode::Scan
    } else {
        SessionMode::Fixed
    };
    if tx
        .send(Event::Session {
            input,
            sample_rate: sample_rate as u32,
            mode,
        })
        .await
        .is_err()
    {
        return;
    }

    // Pacing: one tick per ~20 ms of source audio, scaled by pace_factor.
    let chunk_samples = ((sample_rate * 0.020) as usize).max(64);
    let effective_rate = (sample_rate * pace_factor.max(0.01)).max(1.0);
    let chunk_period =
        Duration::from_secs_f64(f64::from(chunk_samples as u32) / f64::from(effective_rate));
    let mut interval = tokio::time::interval(chunk_period);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    // Decide tones and which samples still need to be decoded.
    let mut cursor = 0_usize;
    let tones: Vec<f32> = if cfg.scan {
        if tx
            .send(Event::ScanStatus {
                state: ScanState::Calibrating,
                detected: None,
            })
            .await
            .is_err()
        {
            return;
        }
        let (detected, scan_end) =
            run_scan(&cfg, &samples, sample_rate, chunk_samples, &mut interval).await;
        cursor = scan_end;
        if tx
            .send(Event::ScanStatus {
                state: ScanState::Ready,
                detected: Some(detected.len()),
            })
            .await
            .is_err()
        {
            return;
        }
        detected
    } else {
        cfg.tones.clone()
    };

    if tones.is_empty() {
        let _ = tx.send(Event::Done).await;
        return;
    }

    let multi = tones.len() > 1;
    let on_floor = if multi { DEFAULT_MULTI_ON_FLOOR } else { 0.0 };

    let mut backend: Box<dyn EnvelopeProducer + Send + Sync> = if cfg.fft {
        let fft_size = auto_fft_size(sample_rate, cfg.wpm);
        let hop = auto_hop(sample_rate, cfg.wpm, fft_size);
        Box::new(FftBackend::new(fft_size, hop, sample_rate, &tones))
    } else {
        let block_len = resolved_block_len(sample_rate, &tones);
        Box::new(GoertzelBackend::new(sample_rate, &tones, block_len))
    };
    let env_rate = backend.envelope_sample_rate();
    let actual_freqs = backend.frequencies();

    let mut chains: Vec<ChannelChain> = tones
        .iter()
        .map(|_| ChannelChain::new(env_rate, cfg.wpm, on_floor))
        .collect();

    for (idx, (freq, chain)) in actual_freqs.iter().zip(&chains).enumerate() {
        let open = Event::ChannelOpen {
            id: idx as u32,
            freq_hz: *freq,
            wpm: chain.decoder.timing().wpm(env_rate),
        };
        if tx.send(open).await.is_err() {
            return;
        }
    }

    let mut env_scratch = vec![0.0_f32; tones.len()];
    let mut spectrum = cfg
        .fft
        .then(|| SpectrumEmitter::new(env_rate, sample_rate));

    // Scan prefetch: blast the buffered calibration samples through the
    // real decode pipeline as fast as possible so any characters sent
    // during calibration land on the UI right after channel_open.
    for &sample in &samples[..cursor] {
        match feed_sample(sample, &mut backend, &mut chains, &mut env_scratch, env_rate, &tx).await
        {
            FeedOutcome::Break => return,
            FeedOutcome::NoFrame => {}
            FeedOutcome::Frame => {
                if !emit_spectrum(spectrum.as_mut(), backend.as_ref(), &tx).await {
                    return;
                }
            }
        }
    }

    // Remaining samples run at the paced rate.
    for chunk in samples[cursor..].chunks(chunk_samples) {
        interval.tick().await;
        for &sample in chunk {
            match feed_sample(
                sample,
                &mut backend,
                &mut chains,
                &mut env_scratch,
                env_rate,
                &tx,
            )
            .await
            {
                FeedOutcome::Break => return,
                FeedOutcome::NoFrame => {}
                FeedOutcome::Frame => {
                    if !emit_spectrum(spectrum.as_mut(), backend.as_ref(), &tx).await {
                        return;
                    }
                }
            }
        }
    }

    for (idx, chain) in chains.iter_mut().enumerate() {
        let events = chain.finish();
        for ev in events {
            if !send_decoded(&tx, idx as u32, ev).await {
                return;
            }
        }
    }
    let _ = tx.send(Event::Done).await;
}

/// If the FFT backend produced a frame this push and the cadence counter
/// is due, send a `Spectrum` event. Returns `false` only when the receiver
/// has gone away.
async fn emit_spectrum(
    emitter: Option<&mut SpectrumEmitter>,
    backend: &(dyn EnvelopeProducer + Send + Sync),
    tx: &mpsc::Sender<Event>,
) -> bool {
    let Some(emitter) = emitter else { return true };
    let Some(mag) = backend.latest_spectrum() else {
        return true;
    };
    let Some(ev) = emitter.maybe_emit(mag) else {
        return true;
    };
    tx.send(ev).await.is_ok()
}

/// Outcome of a single-sample push. `Frame` means the backend produced a
/// new envelope frame (so the caller may also want to emit a spectrum
/// event); `Break` signals the downstream receiver is gone.
enum FeedOutcome {
    NoFrame,
    Frame,
    Break,
}

async fn feed_sample(
    sample: f32,
    backend: &mut Box<dyn EnvelopeProducer + Send + Sync>,
    chains: &mut [ChannelChain],
    env_scratch: &mut [f32],
    env_rate: f32,
    tx: &mpsc::Sender<Event>,
) -> FeedOutcome {
    if !backend.push(sample, env_scratch) {
        return FeedOutcome::NoFrame;
    }
    for (idx, chain) in chains.iter_mut().enumerate() {
        let events = chain.feed_envelope(env_scratch[idx]);
        let channel = idx as u32;
        for ev in events {
            if !send_decoded(tx, channel, ev).await {
                return FeedOutcome::Break;
            }
        }
        if let Some(wpm) = chain.take_wpm_update(env_rate)
            && tx.send(Event::Wpm { channel, wpm }).await.is_err()
        {
            return FeedOutcome::Break;
        }
    }
    FeedOutcome::Frame
}

async fn send_decoded(tx: &mpsc::Sender<Event>, channel: u32, ev: Decoded) -> bool {
    let event = match ev {
        Decoded::Char(c) => Event::Char { channel, ch: c },
        Decoded::WordBreak => Event::WordBreak { channel },
        Decoded::Unknown => Event::Unknown { channel },
    };
    tx.send(event).await.is_ok()
}

/// Drive the first `scan_duration` seconds of `samples` through an FFT
/// channelizer purely to collect per-bin statistics, then detect occupied
/// bins. Returns the detected centre frequencies alongside the index into
/// `samples` where the calibration window ended (so the caller can replay
/// those same samples through the real decode pipeline).
async fn run_scan(
    cfg: &PipelineConfig,
    samples: &[f32],
    sample_rate: f32,
    chunk_samples: usize,
    interval: &mut tokio::time::Interval,
) -> (Vec<f32>, usize) {
    let fft_size = auto_fft_size(sample_rate, cfg.wpm);
    let hop = auto_hop(sample_rate, cfg.wpm, fft_size);
    let mut channelizer = FftChannelizer::new(fft_size, hop, sample_rate);
    let mut stats = BinStats::new(channelizer.channel_count());
    let mut mag_frame = vec![0.0_f32; channelizer.channel_count()];

    let target_samples =
        ((cfg.scan_duration * sample_rate) as usize).max(fft_size).min(samples.len());
    let mut consumed = 0_usize;

    for chunk in samples[..target_samples].chunks(chunk_samples) {
        interval.tick().await;
        for &sample in chunk {
            if let Some(bins) = channelizer.push(sample) {
                for (dst, c) in mag_frame.iter_mut().zip(bins) {
                    *dst = c.norm();
                }
                stats.observe(&mag_frame);
            }
        }
        consumed += chunk.len();
    }

    let scan_cfg = ScanConfig {
        peak_snr_db: cfg.scan_snr_db,
        max_channels: cfg.scan_max_channels,
        nms_radius: cfg.scan_nms_radius,
        min_bin: channelizer.bin_index_for(cfg.scan_min_freq).max(1),
        max_bin: Some(
            (channelizer.bin_index_for(cfg.scan_max_freq) + 1).min(channelizer.channel_count()),
        ),
        ..ScanConfig::default()
    };
    let tones: Vec<f32> = stats
        .detect(&scan_cfg)
        .iter()
        .map(|&b| channelizer.bin_frequency(b))
        .collect();
    (tones, consumed)
}

fn resolved_block_len(sample_rate: f32, tones: &[f32]) -> u32 {
    let lowest = tones.iter().copied().fold(f32::INFINITY, f32::min);
    let raw = (DEFAULT_BLOCK_CYCLES * sample_rate / lowest).round() as u32;
    raw.max(MIN_BLOCK_LEN)
}

fn auto_fft_size(sample_rate: f32, wpm: f32) -> usize {
    let dit_s = 1.2 / wpm;
    let raw = sample_rate * dit_s;
    let cap = if raw >= 1.0 { raw as usize } else { 1 };
    prev_pow2(cap).clamp(MIN_AUTO_FFT_SIZE, MAX_AUTO_FFT_SIZE)
}

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

/// Narrow trait over whatever DSP front end produces one envelope sample
/// per channel. Lets `pump` drive either a `GoertzelBank` or an
/// `FftChannelizer` through the same loop.
trait EnvelopeProducer {
    /// Feed one input sample. When a new envelope frame is ready, fill
    /// `envs` (length equal to the channel count) and return `true`.
    fn push(&mut self, sample: f32, envs: &mut [f32]) -> bool;
    fn envelope_sample_rate(&self) -> f32;
    /// Centre frequency actually being measured on each channel, in the
    /// same order as [`push`]'s output slots. For FFT channels this is
    /// the bin centre, which may differ from the requested tone.
    fn frequencies(&self) -> Vec<f32>;
    /// Most recent full-spectrum magnitudes, when this backend produces
    /// one (FFT only). Returns `None` for backends that don't have a
    /// full-band view (Goertzel) or when no frame has been emitted yet.
    fn latest_spectrum(&self) -> Option<&[f32]>;
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

    fn frequencies(&self) -> Vec<f32> {
        self.tones.clone()
    }

    fn latest_spectrum(&self) -> Option<&[f32]> {
        None
    }
}

struct FftBackend {
    channelizer: FftChannelizer,
    bins: Vec<usize>,
    actual_freqs: Vec<f32>,
    /// Magnitudes of the most recent FFT frame, kept around so the pump
    /// can read them after `push` returns. Sized to `channel_count()`.
    mag_frame: Vec<f32>,
    has_frame: bool,
}

impl FftBackend {
    fn new(fft_size: usize, hop: usize, sample_rate: f32, tones: &[f32]) -> Self {
        let channelizer = FftChannelizer::new(fft_size, hop, sample_rate);
        let bins: Vec<usize> = tones.iter().map(|&t| channelizer.bin_index_for(t)).collect();
        let actual_freqs: Vec<f32> =
            bins.iter().map(|&b| channelizer.bin_frequency(b)).collect();
        let mag_frame = vec![0.0_f32; channelizer.channel_count()];
        Self {
            channelizer,
            bins,
            actual_freqs,
            mag_frame,
            has_frame: false,
        }
    }
}

impl EnvelopeProducer for FftBackend {
    fn push(&mut self, sample: f32, envs: &mut [f32]) -> bool {
        if let Some(frame) = self.channelizer.push(sample) {
            for (slot, c) in self.mag_frame.iter_mut().zip(frame) {
                *slot = c.norm();
            }
            for (slot, &bin) in envs.iter_mut().zip(&self.bins) {
                *slot = self.mag_frame[bin];
            }
            self.has_frame = true;
            true
        } else {
            false
        }
    }

    fn envelope_sample_rate(&self) -> f32 {
        self.channelizer.output_sample_rate()
    }

    fn frequencies(&self) -> Vec<f32> {
        self.actual_freqs.clone()
    }

    fn latest_spectrum(&self) -> Option<&[f32]> {
        self.has_frame.then_some(self.mag_frame.as_slice())
    }
}

/// Per-channel decode pipeline state.
struct ChannelChain {
    threshold: Threshold,
    rle: RunLengthEncoder,
    decoder: BootstrapDecoder,
    last_reported_wpm: f32,
}

impl ChannelChain {
    fn new(env_rate: f32, wpm: f32, on_floor: f32) -> Self {
        let mut threshold = Threshold::new(env_rate, PEAK_HALF_LIFE_S, MIN_PEAK);
        if on_floor > 0.0 {
            threshold = threshold.with_absolute_on_floor(on_floor);
        }
        let decoder = BootstrapDecoder::new(TimingEstimator::from_wpm(wpm, env_rate));
        Self {
            threshold,
            rle: RunLengthEncoder::new(),
            decoder,
            last_reported_wpm: wpm,
        }
    }

    fn feed_envelope(&mut self, env: f32) -> Vec<Decoded> {
        let mark = self.threshold.push(env);
        let mut out = Vec::new();
        if let Some(run) = self.rle.push(mark) {
            for ev in self.decoder.push(run.mark, run.duration) {
                out.push(ev);
            }
        }
        out
    }

    fn finish(&mut self) -> Vec<Decoded> {
        let mut out = Vec::new();
        if let Some(run) = self.rle.finish() {
            for ev in self.decoder.push(run.mark, run.duration) {
                out.push(ev);
            }
        }
        for ev in self.decoder.finish() {
            out.push(ev);
        }
        out
    }

    /// If the decoder's current WPM has drifted meaningfully from the last
    /// value we reported, return the new value and update the threshold.
    fn take_wpm_update(&mut self, env_rate: f32) -> Option<f32> {
        let current = self.decoder.timing().wpm(env_rate);
        if (current - self.last_reported_wpm).abs() >= WPM_EVENT_THRESHOLD {
            self.last_reported_wpm = current;
            Some(current)
        } else {
            None
        }
    }
}

/// Cadence-controlled encoder for `Spectrum` events. Lives for the
/// duration of one connection.
struct SpectrumEmitter {
    /// Number of envelope frames to skip between emits. Computed once
    /// from the backend's frame rate to target [`SPECTRUM_TARGET_FPS`].
    decimate: u32,
    counter: u32,
    /// Reusable byte scratch sized to the FFT bin count on first emit.
    bytes: Vec<u8>,
    /// Frequency span the bins cover. Sent with each event so the UI is
    /// self-describing without holding session state.
    f_min: f32,
    f_max: f32,
}

impl SpectrumEmitter {
    fn new(env_rate: f32, sample_rate: f32) -> Self {
        // env_rate is the channelizer's frame rate. We want roughly
        // SPECTRUM_TARGET_FPS frames per second going to the wire; emit
        // once every Nth frame.
        let decimate = (env_rate / SPECTRUM_TARGET_FPS).round().max(1.0) as u32;
        Self {
            decimate,
            // Start at decimate-1 so the very first frame produces an
            // immediate emit — gives the UI something to draw without a
            // ~40 ms warmup pause.
            counter: decimate.saturating_sub(1),
            bytes: Vec::new(),
            f_min: 0.0,
            f_max: sample_rate / 2.0,
        }
    }

    fn maybe_emit(&mut self, mag: &[f32]) -> Option<Event> {
        self.counter += 1;
        if self.counter < self.decimate {
            return None;
        }
        self.counter = 0;
        if self.bytes.len() != mag.len() {
            self.bytes.resize(mag.len(), 0);
        }
        for (b, &m) in self.bytes.iter_mut().zip(mag) {
            *b = mag_to_u8(m);
        }
        Some(Event::Spectrum {
            bins: BASE64.encode(&self.bytes),
            f_min: self.f_min,
            f_max: self.f_max,
            db_floor: SPECTRUM_DB_FLOOR,
            db_ceiling: SPECTRUM_DB_CEILING,
        })
    }
}

/// Map a linear FFT magnitude to a 0..=255 brightness using a fixed dB
/// window. Floor / ceiling come from [`SPECTRUM_DB_FLOOR`] /
/// [`SPECTRUM_DB_CEILING`].
fn mag_to_u8(mag: f32) -> u8 {
    let db = 20.0 * mag.max(1e-10).log10();
    let span = SPECTRUM_DB_CEILING - SPECTRUM_DB_FLOOR;
    let t = (db - SPECTRUM_DB_FLOOR) / span;
    (t.clamp(0.0, 1.0) * 255.0).round() as u8
}
