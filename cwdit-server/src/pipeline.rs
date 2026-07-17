//! Per-connection decode pipeline task.
//!
//! `pump` runs one of these per WebSocket, driving either a Goertzel bank
//! or an FFT channelizer into per-channel Threshold → `RunLengthEncoder` →
//! `BootstrapDecoder` chains and emitting JSON-friendly [`Event`]s.
//!
//! Two modes: **fixed** (single tone or `--channels` list, channels
//! announced up-front) and **scan**, which skims continuously — a
//! [`Detector`] re-runs detection every calibration interval, a
//! [`ChannelTracker`] spawns and retires per-tone decode channels
//! (`channel_open` / `channel_close` events), and the detection
//! channelizer's frames feed the waterfall, cropped to the scanned band.
//!
//! Scan mode is generic over the input domain: [`pump`] skims real audio
//! through Goertzel decode channels, [`pump_iq`] skims complex IQ from an
//! SDR through [`IqTone`] channels on an RF bin grid, with every reported
//! frequency in absolute RF Hz.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use cwdit_dsp::{
    ChannelTracker, Channelizer, Debouncer, Detector, DetectorConfig, FftChannelizer, Goertzel,
    GoertzelBank, IqDetector, IqTone, MovingAverage, RunLengthEncoder, Threshold, ToneFilter,
    TrackerConfig, skim,
};
use rustfft::num_complex::Complex32;
use cwdit_morse::{BootstrapDecoder, Decoded, TimingEstimator};
use cwdit_source::{Source, SourceError, WavSource};
use serde::Serialize;
use tokio::sync::{broadcast, mpsc};

/// Envelope floor used in fixed multi-channel mode to reject sidelobe
/// leakage on blind channel lists. Scan-spawned channels run un-floored —
/// they already cleared the detector's SNR gate.
const DEFAULT_MULTI_ON_FLOOR: f32 = 0.08;
/// Peak-detector half-life for the envelope slicer, in seconds.
const PEAK_HALF_LIFE_S: f32 = 1.0;
/// Minimum envelope peak used as a noise-floor guard (0.0–1.0).
const MIN_PEAK: f32 = 0.005;
/// A detection this close to a live channel refreshes it instead of
/// spawning a duplicate.
const SKIM_MATCH_RADIUS_HZ: f32 = 25.0;
/// Emit a `wpm` event only when a channel's estimate has moved this much
/// since the last emission. Keeps the stream quiet during steady-state.
const WPM_EVENT_THRESHOLD: f32 = 0.5;
/// Target waterfall frame rate (frames per second). The pump decimates
/// the channelizer's native frame rate to land near this.
const SPECTRUM_TARGET_FPS: f32 = 25.0;
/// Cap on bins per `Spectrum` event. An IQ detection FFT can span tens of
/// thousands of bins — far more than a canvas can show — so frames wider
/// than this are max-pooled down (max, not mean, so a narrow CW peak
/// survives pooling).
const MAX_SPECTRUM_BINS: usize = 2_048;
/// Lower edge of the dB range mapped to `u8` 0 in spectrum frames.
const SPECTRUM_DB_FLOOR: f32 = -80.0;
/// Upper edge of the dB range mapped to `u8` 255 in spectrum frames.
const SPECTRUM_DB_CEILING: f32 = 0.0;
/// Live-capture chunk length, in seconds of audio. Small enough that a
/// freshly connected client starts decoding promptly; large enough that
/// per-chunk overhead stays trivial.
const CAPTURE_CHUNK_S: f32 = 0.05;
/// Broadcast depth, in chunks (~13 s at [`CAPTURE_CHUNK_S`]), before a
/// slow pipeline starts losing samples.
const FEED_CAPACITY: usize = 256;

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
    /// Seconds a skimmed channel may go undetected before `channel_close`.
    pub channel_timeout: f32,
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
    /// events. Emitted up-front in `fixed` mode; in `scan` mode channels
    /// open whenever a new station is detected. `id` is stable for the
    /// life of the connection and never reused.
    ChannelOpen {
        id: u32,
        freq_hz: f32,
        wpm: f32,
    },
    /// A skimmed channel went undetected past the timeout and was
    /// retired. Its decoded text stays valid; no further events carry
    /// this id. Scan mode only.
    ChannelClose {
        id: u32,
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

/// One batch of input samples, shared between live-feed subscribers.
/// `f32` for audio, `Complex32` for IQ.
pub type Chunk<T = f32> = Arc<Vec<T>>;

/// Where a connection's samples come from. Generic over the sample type:
/// `f32` for audio, `Complex32` for IQ.
pub enum Feed<T = f32> {
    /// The whole input is in memory; replay it at `pace_factor` × real
    /// time. Every connection starts from the beginning.
    Replay {
        samples: Arc<Vec<T>>,
        pace_factor: f32,
    },
    /// A shared live capture (see [`spawn_capture`]); the connection
    /// decodes the stream from "now".
    Live { rx: broadcast::Receiver<Chunk<T>> },
}

/// Open a live source via `open` on a dedicated capture thread and fan
/// its samples out to every subscriber of the returned sender. The source
/// is *created on* the thread that reads it because some sources (cpal
/// audio streams) are not `Send`. Returns the feed alongside the source's
/// sample rate.
///
/// An empty sentinel chunk marks end-of-stream (source exhausted or
/// errored) — the sender side outlives the capture inside `AppState`, so
/// channel closure can't signal EOS.
///
/// # Errors
/// Propagates the error from `open` (e.g. no such audio device).
pub fn spawn_capture<S, F>(open: F) -> Result<CaptureFeed<S::Sample>, SourceError>
where
    S: Source,
    S::Sample: Default + Send + Sync + 'static,
    F: FnOnce() -> Result<S, SourceError> + Send + 'static,
{
    let (tx, _) = broadcast::channel(FEED_CAPACITY);
    let feed = tx.clone();
    let (rate_tx, rate_rx) = std::sync::mpsc::sync_channel::<Result<f32, SourceError>>(1);
    std::thread::spawn(move || {
        let mut source = match open() {
            Ok(source) => {
                let _ = rate_tx.send(Ok(source.sample_rate()));
                source
            }
            Err(e) => {
                let _ = rate_tx.send(Err(e));
                return;
            }
        };
        let chunk_len = ((source.sample_rate() * CAPTURE_CHUNK_S) as usize).max(256);
        loop {
            let mut buf = vec![S::Sample::default(); chunk_len];
            match source.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    buf.truncate(n);
                    // No subscribers is fine — the chunk falls on the floor.
                    let _ = feed.send(Arc::new(buf));
                }
                Err(e) => {
                    tracing::warn!("live capture ended: {e}");
                    break;
                }
            }
        }
        let _ = feed.send(Arc::new(Vec::new()));
    });
    let sample_rate = rate_rx
        .recv()
        .map_err(|_| SourceError::Decode("capture thread died during open".into()))??;
    Ok((tx, sample_rate))
}

/// What [`spawn_capture`] hands back: the broadcast feed to subscribe
/// connections to, plus the source's sample rate.
pub type CaptureFeed<T> = (broadcast::Sender<Chunk<T>>, f32);

/// Pull-based iteration over a [`Feed`]. Replay paces itself against the
/// wall clock; live waits on the broadcast.
enum FeedIter<T = f32> {
    Replay {
        samples: Arc<Vec<T>>,
        pos: usize,
        chunk_samples: usize,
        interval: tokio::time::Interval,
    },
    Live {
        rx: broadcast::Receiver<Chunk<T>>,
    },
}

impl<T: Copy + Send + 'static> FeedIter<T> {
    fn new(feed: Feed<T>, sample_rate: f32) -> Self {
        match feed {
            Feed::Replay {
                samples,
                pace_factor,
            } => {
                // Pacing: one tick per ~20 ms of source audio, scaled by
                // pace_factor.
                let chunk_samples = ((sample_rate * 0.020) as usize).max(64);
                let effective_rate = (sample_rate * pace_factor.max(0.01)).max(1.0);
                let chunk_period = Duration::from_secs_f64(
                    f64::from(chunk_samples as u32) / f64::from(effective_rate),
                );
                let mut interval = tokio::time::interval(chunk_period);
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                Self::Replay {
                    samples,
                    pos: 0,
                    chunk_samples,
                    interval,
                }
            }
            Feed::Live { rx } => Self::Live { rx },
        }
    }

    /// Next chunk of input, or `None` at end of stream. A lagged live
    /// subscriber (pipeline slower than capture for ~13 s) loses the
    /// dropped samples but keeps streaming.
    async fn next(&mut self) -> Option<Chunk<T>> {
        match self {
            Self::Replay {
                samples,
                pos,
                chunk_samples,
                interval,
            } => {
                if *pos >= samples.len() {
                    return None;
                }
                interval.tick().await;
                let end = (*pos + *chunk_samples).min(samples.len());
                let chunk = Arc::new(samples[*pos..end].to_vec());
                *pos = end;
                Some(chunk)
            }
            Self::Live { rx } => loop {
                match rx.recv().await {
                    Ok(chunk) if chunk.is_empty() => return None,
                    Ok(chunk) => return Some(chunk),
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!("live feed lagged: {n} chunks dropped");
                    }
                    Err(broadcast::error::RecvError::Closed) => return None,
                }
            },
        }
    }
}

/// Stream a [`Feed`] through a fresh decode pipeline, publishing
/// [`Event`]s. Returns at end of feed or when the receiver is dropped.
#[allow(clippy::too_many_lines)]
pub async fn pump(
    input: String,
    sample_rate: f32,
    feed: Feed,
    cfg: Arc<PipelineConfig>,
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

    let mut feed = FeedIter::new(feed, sample_rate);

    if cfg.scan {
        let state = SkimState::new_audio(&cfg, sample_rate);
        run_skim(feed, state, &tx).await;
        return;
    }

    let tones = cfg.tones.clone();
    if tones.is_empty() {
        let _ = tx.send(Event::Done).await;
        return;
    }

    let multi = tones.len() > 1;
    let on_floor = if multi { DEFAULT_MULTI_ON_FLOOR } else { 0.0 };

    let mut backend: Box<dyn EnvelopeProducer + Send + Sync> = if cfg.fft {
        let fft_size = skim::decode_fft_size(sample_rate, cfg.wpm);
        let hop = skim::auto_hop(sample_rate, cfg.wpm, fft_size);
        Box::new(FftBackend::new(fft_size, hop, sample_rate, &tones))
    } else {
        let lowest = tones.iter().copied().fold(f32::INFINITY, f32::min);
        let block_len = skim::decode_block_len(sample_rate, cfg.wpm, lowest);
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
        .then(|| SpectrumEmitter::new(env_rate, 0.0, sample_rate / 2.0));

    while let Some(chunk) = feed.next().await {
        for &sample in chunk.iter() {
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

/// Stream an IQ [`Feed`] through a continuous skim pipeline centred on
/// `center_freq` (RF Hz), publishing [`Event`]s whose frequencies —
/// channel opens, waterfall span — are all absolute RF Hz. SDR input
/// always scans; there is no fixed-tone IQ mode in the server.
pub async fn pump_iq(
    input: String,
    sample_rate: f32,
    center_freq: f32,
    feed: Feed<Complex32>,
    cfg: Arc<PipelineConfig>,
    tx: mpsc::Sender<Event>,
) {
    if tx
        .send(Event::Session {
            input,
            sample_rate: sample_rate as u32,
            mode: SessionMode::Scan,
        })
        .await
        .is_err()
    {
        return;
    }
    let state = SkimState::new_iq(&cfg, sample_rate, center_freq);
    run_skim(FeedIter::new(feed, sample_rate), state, &tx).await;
}

/// Drive a feed through a [`SkimState`] until end of input or the
/// receiver goes away. Shared by the audio and IQ scan paths.
async fn run_skim<C, F>(mut feed: FeedIter<C::Input>, mut state: SkimState<C, F>, tx: &mpsc::Sender<Event>)
where
    C: Channelizer,
    C::Input: Send + Sync + 'static,
    F: ToneFilter<Input = C::Input>,
{
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
    while let Some(chunk) = feed.next().await {
        for &sample in chunk.iter() {
            if !state.push(sample, tx).await {
                return;
            }
        }
    }
    if state.finish(tx).await {
        let _ = tx.send(Event::Done).await;
    }
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

/// One live skim channel: a [`ToneFilter`] on the detected tone feeding
/// its own decode chain. `id` is the wire-protocol channel id.
struct SkimChannel<F: ToneFilter> {
    id: u32,
    filter: F,
    chain: ChannelChain,
}

/// Continuous scan-mode state: a [`Detector`] finds stations per
/// calibration interval, a [`ChannelTracker`] decides lifecycle, and each
/// live station decodes through its own [`SkimChannel`]. Waterfall frames
/// come from the detection channelizer, cropped to the scanned band.
/// Generic over the input domain: [`new_audio`](Self::new_audio) skims
/// real audio via Goertzel channels, [`new_iq`](SkimState::new_iq) skims
/// complex IQ via [`IqTone`] channels in absolute RF Hz.
struct SkimState<C: Channelizer = FftChannelizer, F: ToneFilter<Input = C::Input> = Goertzel> {
    detector: Detector<C>,
    tracker: ChannelTracker,
    channels: Vec<SkimChannel<F>>,
    spectrum: SpectrumEmitter,
    /// Displayed bin range of a detection frame (the scanned band).
    range: (usize, usize),
    next_id: u32,
    announced_ready: bool,
    wpm: f32,
    env_rate: f32,
    sample_rate: f32,
    total_samples: u64,
    /// Decode filter tuned to a detected tone, in this mode's frequency
    /// units (audio Hz or absolute RF Hz).
    make_filter: Box<dyn Fn(f32) -> F + Send + Sync>,
}

impl SkimState {
    fn new_audio(cfg: &PipelineConfig, sample_rate: f32) -> Self {
        let fft_size = skim::detect_fft_size(sample_rate, cfg.wpm);
        let detector = Detector::new(
            &DetectorConfig {
                fft_size,
                hop: skim::auto_hop(sample_rate, cfg.wpm, fft_size),
                min_freq_hz: cfg.scan_min_freq,
                max_freq_hz: cfg.scan_max_freq,
                snr_db: cfg.scan_snr_db,
                nms_radius: cfg.scan_nms_radius,
                max_channels: cfg.scan_max_channels,
                interval_s: cfg.scan_duration,
            },
            sample_rate,
        );
        let block_len = skim::decode_block_len(sample_rate, cfg.wpm, cfg.scan_min_freq);
        let make_filter = move |tone: f32| Goertzel::new(tone, sample_rate, block_len);
        Self::from_parts(cfg, sample_rate, detector, block_len, Box::new(make_filter))
    }
}

impl SkimState<cwdit_dsp::IqChannelizer, IqTone> {
    /// IQ skim over an RF bin grid centred on `center_freq`. Detection
    /// sizes by bin spacing rather than dit width (see the CLI's `IqSkim`
    /// rationale); `cfg.scan_min_freq` / `scan_max_freq` are absolute RF Hz.
    fn new_iq(cfg: &PipelineConfig, sample_rate: f32, center_freq: f32) -> Self {
        let fft_size = skim::detect_iq_fft_size(sample_rate);
        let detector = IqDetector::new_iq(
            &DetectorConfig {
                fft_size,
                hop: skim::auto_hop(sample_rate, cfg.wpm, fft_size).clamp(1, fft_size / 2),
                min_freq_hz: cfg.scan_min_freq,
                max_freq_hz: cfg.scan_max_freq,
                snr_db: cfg.scan_snr_db,
                nms_radius: cfg.scan_nms_radius,
                max_channels: cfg.scan_max_channels,
                interval_s: cfg.scan_duration,
            },
            sample_rate,
            center_freq,
        );
        let block_len = skim::iq_decode_block_len(sample_rate, cfg.wpm);
        let make_filter =
            move |tone: f32| IqTone::new(tone - center_freq, sample_rate, block_len);
        Self::from_parts(cfg, sample_rate, detector, block_len, Box::new(make_filter))
    }
}

impl<C: Channelizer, F: ToneFilter<Input = C::Input>> SkimState<C, F> {
    fn from_parts(
        cfg: &PipelineConfig,
        sample_rate: f32,
        detector: Detector<C>,
        block_len: u32,
        make_filter: Box<dyn Fn(f32) -> F + Send + Sync>,
    ) -> Self {
        let range = detector.bin_range();
        let spectrum = SpectrumEmitter::new(
            detector.frame_rate(),
            detector.bin_frequency(range.0),
            detector.bin_frequency(range.1 - 1),
        );
        Self {
            tracker: ChannelTracker::new(TrackerConfig {
                match_radius_hz: SKIM_MATCH_RADIUS_HZ.max(detector.bin_spacing_hz()),
                timeout_s: cfg.channel_timeout,
                max_channels: cfg.scan_max_channels,
            }),
            channels: Vec::new(),
            spectrum,
            range,
            next_id: 0,
            announced_ready: false,
            wpm: cfg.wpm,
            env_rate: sample_rate / block_len as f32,
            sample_rate,
            total_samples: 0,
            make_filter,
            detector,
        }
    }

    /// Feed one input sample. Returns `false` when the receiver is gone.
    async fn push(&mut self, sample: C::Input, tx: &mpsc::Sender<Event>) -> bool {
        self.total_samples += 1;
        for ch in &mut self.channels {
            if !feed_skim_channel(ch, sample, self.env_rate, tx).await {
                return false;
            }
        }
        if self.detector.push(sample)
            && let Some(frame) = self.detector.latest_frame()
            && let Some(ev) = self.spectrum.maybe_emit(&frame[self.range.0..self.range.1])
            && tx.send(ev).await.is_err()
        {
            return false;
        }
        if self.detector.interval_complete() {
            return self.detection_round(tx).await;
        }
        true
    }

    /// End-of-interval bookkeeping: detect, reap, spawn (replaying the
    /// discovery interval into each new channel), reset.
    async fn detection_round(&mut self, tx: &mpsc::Sender<Event>) -> bool {
        let tones = self.detector.detect();
        let now_s = self.total_samples as f32 / self.sample_rate;
        let update = self.tracker.observe(now_s, &tones);

        for &idx in &update.reaped {
            let mut ch = self.channels.remove(idx);
            for ev in ch.chain.finish() {
                if !send_decoded(tx, ch.id, ev).await {
                    return false;
                }
            }
            if tx.send(Event::ChannelClose { id: ch.id }).await.is_err() {
                return false;
            }
        }

        for &tone in &update.spawned {
            let id = self.next_id;
            self.next_id += 1;
            let open = Event::ChannelOpen {
                id,
                freq_hz: tone,
                wpm: self.wpm,
            };
            if tx.send(open).await.is_err() {
                return false;
            }
            let mut ch = SkimChannel {
                id,
                filter: (self.make_filter)(tone),
                chain: ChannelChain::new(self.env_rate, self.wpm, 0.0),
            };
            // Replay the discovery interval so the transmission that
            // triggered detection is decoded from its start.
            for &s in self.detector.interval_audio() {
                if !feed_skim_channel(&mut ch, s, self.env_rate, tx).await {
                    return false;
                }
            }
            self.channels.push(ch);
        }

        if !self.announced_ready {
            self.announced_ready = true;
            let ready = Event::ScanStatus {
                state: ScanState::Ready,
                detected: Some(self.channels.len()),
            };
            if tx.send(ready).await.is_err() {
                return false;
            }
        }

        self.detector.reset_interval();
        true
    }

    /// Flush every live channel at end of input.
    async fn finish(&mut self, tx: &mpsc::Sender<Event>) -> bool {
        for ch in &mut self.channels {
            for ev in ch.chain.finish() {
                if !send_decoded(tx, ch.id, ev).await {
                    return false;
                }
            }
        }
        true
    }
}

/// Feed one sample to a skim channel, forwarding decode and WPM events.
/// Returns `false` when the receiver is gone.
async fn feed_skim_channel<F: ToneFilter>(
    ch: &mut SkimChannel<F>,
    sample: F::Input,
    env_rate: f32,
    tx: &mpsc::Sender<Event>,
) -> bool {
    let Some(env) = ch.filter.push(sample) else {
        return true;
    };
    for ev in ch.chain.feed_envelope(env) {
        if !send_decoded(tx, ch.id, ev).await {
            return false;
        }
    }
    if let Some(wpm) = ch.chain.take_wpm_update(env_rate)
        && tx
            .send(Event::Wpm {
                channel: ch.id,
                wpm,
            })
            .await
            .is_err()
    {
        return false;
    }
    true
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
    smoother: MovingAverage,
    threshold: Threshold,
    rle: RunLengthEncoder,
    debouncer: Debouncer,
    decoder: BootstrapDecoder,
    last_reported_wpm: f32,
}

impl ChannelChain {
    fn new(env_rate: f32, wpm: f32, on_floor: f32) -> Self {
        // Smooth over ~1/4 dit and absorb runs under ~1/5 dit, sized from
        // the seed WPM (see the cwdit-cli chain for the rationale).
        let dit_ticks = 1.2 * env_rate / wpm;
        let smooth_len = ((dit_ticks / 4.0).round() as usize).clamp(2, 16);
        let min_run = ((dit_ticks / 5.0) as u32).max(2);
        let mut threshold = Threshold::new(env_rate, PEAK_HALF_LIFE_S, MIN_PEAK);
        if on_floor > 0.0 {
            threshold = threshold.with_absolute_on_floor(on_floor);
        }
        let decoder = BootstrapDecoder::new(TimingEstimator::from_wpm(wpm, env_rate));
        Self {
            smoother: MovingAverage::new(smooth_len),
            threshold,
            rle: RunLengthEncoder::new(),
            debouncer: Debouncer::new(min_run),
            decoder,
            last_reported_wpm: wpm,
        }
    }

    fn feed_envelope(&mut self, env: f32) -> Vec<Decoded> {
        let mark = self.threshold.push(self.smoother.push(env));
        let mut out = Vec::new();
        if let Some(run) = self.rle.push(mark).and_then(|r| self.debouncer.push(r)) {
            for ev in self.decoder.push(run.mark, run.duration) {
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
    fn new(frame_rate: f32, f_min: f32, f_max: f32) -> Self {
        // frame_rate is the channelizer's frame rate. We want roughly
        // SPECTRUM_TARGET_FPS frames per second going to the wire; emit
        // once every Nth frame.
        let decimate = (frame_rate / SPECTRUM_TARGET_FPS).round().max(1.0) as u32;
        Self {
            decimate,
            // Start at decimate-1 so the very first frame produces an
            // immediate emit — gives the UI something to draw without a
            // ~40 ms warmup pause.
            counter: decimate.saturating_sub(1),
            bytes: Vec::new(),
            f_min,
            f_max,
        }
    }

    fn maybe_emit(&mut self, mag: &[f32]) -> Option<Event> {
        self.counter += 1;
        if self.counter < self.decimate {
            return None;
        }
        self.counter = 0;
        let pool = mag.len().div_ceil(MAX_SPECTRUM_BINS).max(1);
        let out_len = mag.len().div_ceil(pool);
        if self.bytes.len() != out_len {
            self.bytes.resize(out_len, 0);
        }
        for (b, group) in self.bytes.iter_mut().zip(mag.chunks(pool)) {
            let peak = group.iter().copied().fold(0.0_f32, f32::max);
            *b = mag_to_u8(peak);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Frames wider than the display cap are max-pooled, and a single
    /// narrow spike must survive the pooling (max, not mean).
    #[test]
    fn spectrum_emitter_pools_wide_frames_preserving_peaks() {
        let mut emitter = SpectrumEmitter::new(SPECTRUM_TARGET_FPS, 0.0, 1_000.0);
        let mut mag = vec![1e-6_f32; 10_000];
        mag[5_003] = 1.0; // one hot bin inside what will become one pool group
        let Some(Event::Spectrum { bins, f_min, f_max, .. }) = emitter.maybe_emit(&mag) else {
            panic!("first frame must emit immediately");
        };
        let bytes = BASE64.decode(bins).expect("base64");
        // 10_000 bins pool by ceil(10_000 / 2_048) = 5 → 2_000 wire bins.
        assert_eq!(bytes.len(), 2_000);
        assert!(bytes.len() <= MAX_SPECTRUM_BINS);
        assert_eq!(bytes[1_000], mag_to_u8(1.0), "spike lost in pooling");
        // The frequency span is unchanged by pooling.
        assert!((f_min - 0.0).abs() < f32::EPSILON && (f_max - 1_000.0).abs() < f32::EPSILON);
    }

    /// Narrow frames pass through unpooled.
    #[test]
    fn spectrum_emitter_leaves_narrow_frames_alone() {
        let mut emitter = SpectrumEmitter::new(SPECTRUM_TARGET_FPS, 0.0, 1_000.0);
        let mag = vec![0.5_f32; 512];
        let Some(Event::Spectrum { bins, .. }) = emitter.maybe_emit(&mag) else {
            panic!("first frame must emit immediately");
        };
        assert_eq!(BASE64.decode(bins).expect("base64").len(), 512);
    }
}
