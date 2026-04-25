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

    let mut backend: Box<dyn EnvelopeProducer + Send> = if cfg.fft {
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

    // Scan prefetch: blast the buffered calibration samples through the
    // real decode pipeline as fast as possible so any characters sent
    // during calibration land on the UI right after channel_open.
    for &sample in &samples[..cursor] {
        if feed_sample(
            sample,
            &mut backend,
            &mut chains,
            &mut env_scratch,
            env_rate,
            &tx,
        )
        .await
        .is_break()
        {
            return;
        }
    }

    // Remaining samples run at the paced rate.
    for chunk in samples[cursor..].chunks(chunk_samples) {
        interval.tick().await;
        for &sample in chunk {
            if feed_sample(
                sample,
                &mut backend,
                &mut chains,
                &mut env_scratch,
                env_rate,
                &tx,
            )
            .await
            .is_break()
            {
                return;
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

/// Outcome of a single-sample push. `Break` signals that the downstream
/// receiver is gone and we should tear down the pump.
enum FeedOutcome {
    Continue,
    Break,
}

impl FeedOutcome {
    fn is_break(&self) -> bool {
        matches!(self, FeedOutcome::Break)
    }
}

async fn feed_sample(
    sample: f32,
    backend: &mut Box<dyn EnvelopeProducer + Send>,
    chains: &mut [ChannelChain],
    env_scratch: &mut [f32],
    env_rate: f32,
    tx: &mpsc::Sender<Event>,
) -> FeedOutcome {
    if !backend.push(sample, env_scratch) {
        return FeedOutcome::Continue;
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
    FeedOutcome::Continue
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
}

struct FftBackend {
    channelizer: FftChannelizer,
    bins: Vec<usize>,
    actual_freqs: Vec<f32>,
}

impl FftBackend {
    fn new(fft_size: usize, hop: usize, sample_rate: f32, tones: &[f32]) -> Self {
        let channelizer = FftChannelizer::new(fft_size, hop, sample_rate);
        let bins: Vec<usize> = tones.iter().map(|&t| channelizer.bin_index_for(t)).collect();
        let actual_freqs: Vec<f32> =
            bins.iter().map(|&b| channelizer.bin_frequency(b)).collect();
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

    fn frequencies(&self) -> Vec<f32> {
        self.actual_freqs.clone()
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
