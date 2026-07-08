//! Continuous detection front-end for skimming.
//!
//! A skimmer splits its DSP into two halves with opposite window trades:
//!
//! - **Detection** wants a *long* analysis window (narrow bins: deep-noise
//!   sensitivity, close-station separation) — smearing the keying doesn't
//!   matter because detection only accumulates per-bin statistics.
//! - **Decode** wants a *short* window (≤ ~dit/4) so keying edges survive
//!   integration.
//!
//! [`Detector`] is the detection half: generic over any
//! [`Channelizer`], it accumulates [`BinStats`] per calibration interval
//! and at each interval boundary produces ghost-filtered,
//! fractionally-interpolated tone frequencies for the caller's channel
//! tracker ([`ChannelTracker`](crate::ChannelTracker)). It also buffers
//! the interval's raw input so a newly spawned decode channel can replay
//! the transmission that triggered its detection, and exposes each FFT
//! frame's magnitudes for waterfall display. [`Detector::new`] builds the
//! real-audio detector ([`FftChannelizer`]); [`IqDetector::new_iq`] the
//! IQ one ([`IqChannelizer`](crate::IqChannelizer)), which works in
//! absolute RF Hz.
//!
//! The sizing helpers at the bottom encode the window policy both the CLI
//! and server front-ends share; the audio-rate ones were tuned against
//! the regression harness in `cwdit-cli/tests/decode_quality.rs`. IQ-rate
//! detection sizes by *bin spacing* instead of dit width: at Msps rates a
//! dit-length window would give bins hundreds of Hz wide — too coarse to
//! separate stations working each other tens of Hz apart.

use crate::channelizer::{Channelizer, FftChannelizer};
use crate::iq_channelizer::IqChannelizer;
use crate::scan::{BinStats, ScanConfig, suppress_correlated_ghosts};

/// Decode analysis window, as a fraction of a dit.
pub const DECODE_WINDOW_DITS: f32 = 0.25;

/// Detection analysis window, in dits.
pub const DETECT_WINDOW_DITS: f32 = 2.0;

/// Sliding-window radius for detection's local noise-floor estimate, in
/// Hz. Wide enough that most bins under the window are unoccupied, narrow
/// enough to track a receiver's audio-passband shaping.
pub const FLOOR_RADIUS_HZ: f32 = 300.0;

/// Reach of the correlated-ghost filter around a detected peak, in Hz.
/// Hard keying puts click sidebands at multiples of the keying rate —
/// hundreds of Hz from a strong carrier.
pub const GHOST_RADIUS_HZ: f32 = 375.0;

/// Correlation above which a much weaker neighbour is dropped as a ghost;
/// see [`suppress_correlated_ghosts`].
pub const GHOST_CORR: f32 = 0.5;

/// The parent must exceed a ghost candidate by this many dB; near-equal
/// peaks are always kept as distinct signals.
pub const GHOST_MIN_DB: f32 = 6.0;

/// Auto-hop target: envelope samples per dit.
pub const TARGET_SAMPLES_PER_DIT: f32 = 10.0;

/// Bounds for auto-selected audio-rate FFT sizes.
pub const MIN_AUTO_FFT_SIZE: usize = 128;
pub const MAX_AUTO_FFT_SIZE: usize = 4096;

/// Minimum Goertzel block length regardless of sample rate / tone.
pub const MIN_BLOCK_LEN: u32 = 16;

/// FFT size for the detection channelizer: a [`DETECT_WINDOW_DITS`]-long
/// window, rounded down to a power of two and clamped.
#[must_use]
pub fn detect_fft_size(sample_rate: f32, wpm: f32) -> usize {
    window_fft_size(sample_rate, wpm, DETECT_WINDOW_DITS)
}

/// FFT size for a decode channelizer: a [`DECODE_WINDOW_DITS`]-long
/// window, rounded down to a power of two and clamped.
#[must_use]
pub fn decode_fft_size(sample_rate: f32, wpm: f32) -> usize {
    window_fft_size(sample_rate, wpm, DECODE_WINDOW_DITS)
}

/// FFT size whose window spans `window_dits` dits at `wpm`.
#[must_use]
pub fn window_fft_size(sample_rate: f32, wpm: f32, window_dits: f32) -> usize {
    let dit_s = 1.2 / wpm;
    let raw = sample_rate * dit_s * window_dits;
    let cap = if raw >= 1.0 { raw as usize } else { 1 };
    prev_pow2(cap).clamp(MIN_AUTO_FFT_SIZE, MAX_AUTO_FFT_SIZE)
}

/// Hop size targeting [`TARGET_SAMPLES_PER_DIT`] envelope frames per dit,
/// clamped to half the FFT size.
#[must_use]
pub fn auto_hop(sample_rate: f32, wpm: f32, fft_size: usize) -> usize {
    let dit_s = 1.2 / wpm;
    let raw = (sample_rate * dit_s / TARGET_SAMPLES_PER_DIT).floor();
    let hop = if raw >= 1.0 { raw as usize } else { 1 };
    hop.clamp(1, fft_size / 2)
}

/// Goertzel block length for decode: a [`DECODE_WINDOW_DITS`]-long window,
/// never shorter than one full cycle of the lowest tone of interest.
#[must_use]
pub fn decode_block_len(sample_rate: f32, wpm: f32, lowest_tone_hz: f32) -> u32 {
    let dit_s = 1.2 / wpm;
    let raw = (DECODE_WINDOW_DITS * dit_s * sample_rate).round() as u32;
    let min_cycle = (sample_rate / lowest_tone_hz).ceil() as u32 + 1;
    raw.max(min_cycle).max(MIN_BLOCK_LEN)
}

/// Target bin spacing for IQ-rate *detection* FFTs, in Hz. Stations in a
/// pile-up work each other well under 100 Hz apart; ~25 Hz spacing (before
/// the power-of-two round-up narrows it further) keeps them in separate
/// bins. Detection window length falls out of the spacing — narrower bins
/// mean a longer window, which detection wants anyway.
pub const IQ_DETECT_BIN_SPACING_HZ: f32 = 25.0;

/// Bounds for auto-selected IQ-rate FFT sizes. The ceiling caps FFT cost
/// at high sample rates (a 65 536-point FFT at 1.024 Msps still gives
/// 15.6 Hz bins).
pub const MIN_IQ_FFT_SIZE: usize = 4_096;
pub const MAX_IQ_FFT_SIZE: usize = 65_536;

/// FFT size for IQ-rate detection: bin spacing at or below
/// [`IQ_DETECT_BIN_SPACING_HZ`], rounded up to a power of two and clamped.
/// Pair with [`auto_hop`] for the hop, as on the audio path.
#[must_use]
pub fn detect_iq_fft_size(sample_rate: f32) -> usize {
    let raw = (sample_rate / IQ_DETECT_BIN_SPACING_HZ).ceil();
    let n = if raw >= 1.0 { raw as usize } else { 1 };
    n.next_power_of_two().clamp(MIN_IQ_FFT_SIZE, MAX_IQ_FFT_SIZE)
}

/// Integration block length for an [`IqTone`](crate::IqTone) decode
/// filter: a [`DECODE_WINDOW_DITS`]-long window. Unlike
/// [`decode_block_len`] there is no lowest-tone cycle bound — the mixer
/// shifts the target to DC, so any block length integrates coherently.
#[must_use]
pub fn iq_decode_block_len(sample_rate: f32, wpm: f32) -> u32 {
    let dit_s = 1.2 / wpm;
    let raw = (DECODE_WINDOW_DITS * dit_s * sample_rate).round() as u32;
    raw.max(MIN_BLOCK_LEN)
}

fn prev_pow2(n: usize) -> usize {
    if n < 2 {
        1
    } else {
        1usize << (usize::BITS - 1 - n.leading_zeros())
    }
}

/// Configuration for [`Detector`]. Radii expressed in Hz are converted to
/// bins internally, so they survive window-size changes.
#[derive(Debug, Clone)]
pub struct DetectorConfig {
    pub fft_size: usize,
    pub hop: usize,
    /// Lower / upper bounds of the searched band, in Hz.
    pub min_freq_hz: f32,
    pub max_freq_hz: f32,
    /// Minimum peak SNR (dB) against the local noise floor.
    pub snr_db: f32,
    /// Hard non-max-suppression radius, in bins.
    pub nms_radius: usize,
    /// Cap on detections returned per interval.
    pub max_channels: usize,
    /// Calibration interval length, in seconds.
    pub interval_s: f32,
}

/// Continuous occupied-tone detector over any [`Channelizer`]; see the
/// module docs. The real-audio detector is the default type parameter;
/// [`IqDetector`] skims complex IQ on an RF bin grid.
pub struct Detector<C: Channelizer = FftChannelizer> {
    channelizer: C,
    cfg: ScanConfig,
    stats: BinStats,
    /// Raw samples of the current interval — replayed into new decode
    /// channels, and re-channelized at interval end to build envelope
    /// histories for the candidate bins (storing live histories for every
    /// scan-range bin would cost hundreds of MB at SDR rates).
    interval: Vec<C::Input>,
    interval_samples: usize,
    mag_frame: Vec<f32>,
    has_frame: bool,
    ghost_bins: usize,
    min_bin: usize,
    max_bin: usize,
    sample_rate: f32,
}

impl Detector<FftChannelizer> {
    /// Build a detector for real audio at `sample_rate` Hz.
    ///
    /// # Panics
    /// Panics on degenerate configuration (see [`FftChannelizer::new`]).
    #[must_use]
    pub fn new(cfg: &DetectorConfig, sample_rate: f32) -> Self {
        Self::from_channelizer(
            FftChannelizer::new(cfg.fft_size, cfg.hop, sample_rate),
            cfg,
            sample_rate,
        )
    }
}

/// [`Detector`] over complex IQ centred on an RF carrier.
pub type IqDetector = Detector<IqChannelizer>;

impl IqDetector {
    /// Build a detector for complex IQ at `sample_rate` Hz centred on
    /// `center_freq_hz`. `cfg.min_freq_hz` / `cfg.max_freq_hz` are
    /// absolute RF Hz, as are all reported tone frequencies.
    ///
    /// # Panics
    /// Panics on degenerate configuration (see [`IqChannelizer::new`]).
    #[must_use]
    pub fn new_iq(cfg: &DetectorConfig, sample_rate: f32, center_freq_hz: f32) -> Self {
        Self::from_channelizer(
            IqChannelizer::new(cfg.fft_size, cfg.hop, sample_rate, center_freq_hz),
            cfg,
            sample_rate,
        )
    }
}

impl<C: Channelizer> Detector<C> {
    /// Build a detector around an already-constructed channelizer whose
    /// FFT size and hop match `cfg`.
    #[must_use]
    pub fn from_channelizer(channelizer: C, cfg: &DetectorConfig, sample_rate: f32) -> Self {
        let spacing = channelizer.bin_spacing_hz();
        // `.max(1)` skips the DC bin on the real-audio grid; on the IQ
        // grid bin 0 is just the lower band edge and the clamp is inert.
        let min_bin = channelizer.bin_index_for(cfg.min_freq_hz).max(1);
        let max_bin =
            (channelizer.bin_index_for(cfg.max_freq_hz) + 1).min(channelizer.channel_count());
        let scan_cfg = ScanConfig {
            peak_snr_db: cfg.snr_db,
            max_channels: cfg.max_channels,
            nms_radius: cfg.nms_radius,
            // Peak-ratio dominance can't tell a strong signal's keying
            // sidebands from a genuinely weaker neighbour; the correlated
            // ghost filter in `detect` decides instead.
            dominance_db: f32::INFINITY,
            floor_radius: Some(((FLOOR_RADIUS_HZ / spacing).round() as usize).max(8)),
            min_bin,
            max_bin: Some(max_bin),
            ..ScanConfig::default()
        };
        let n_bins = channelizer.channel_count();
        Self {
            cfg: scan_cfg,
            stats: BinStats::new(n_bins),
            interval: Vec::new(),
            interval_samples: ((cfg.interval_s * sample_rate) as usize).max(cfg.fft_size),
            mag_frame: vec![0.0; n_bins],
            has_frame: false,
            ghost_bins: ((GHOST_RADIUS_HZ / spacing).round() as usize).max(1),
            min_bin,
            max_bin,
            sample_rate,
            channelizer,
        }
    }

    /// Feed one input sample. Returns `true` when the channelizer emitted
    /// a new frame (so [`latest_frame`](Self::latest_frame) has fresh
    /// magnitudes, e.g. for a waterfall).
    pub fn push(&mut self, sample: C::Input) -> bool {
        self.interval.push(sample);
        if let Some(bins) = self.channelizer.push(sample) {
            for (dst, c) in self.mag_frame.iter_mut().zip(bins) {
                *dst = c.norm();
            }
            self.stats.observe(&self.mag_frame);
            self.has_frame = true;
            true
        } else {
            false
        }
    }

    /// Magnitudes of the most recent FFT frame (all bins, DC..Nyquist), or
    /// `None` before the first frame.
    #[must_use]
    pub fn latest_frame(&self) -> Option<&[f32]> {
        self.has_frame.then_some(self.mag_frame.as_slice())
    }

    /// Frame rate of [`latest_frame`](Self::latest_frame) updates, in Hz.
    #[must_use]
    pub fn frame_rate(&self) -> f32 {
        self.channelizer.output_sample_rate()
    }

    /// Bin spacing of the detection channelizer, in Hz. Useful for sizing
    /// a downstream channel tracker's match radius off the grid resolution.
    #[must_use]
    pub fn bin_spacing_hz(&self) -> f32 {
        self.channelizer.bin_spacing_hz()
    }

    /// Searched bin range as `(first, last_exclusive)` — the slice of a
    /// frame worth displaying on a waterfall.
    #[must_use]
    pub const fn bin_range(&self) -> (usize, usize) {
        (self.min_bin, self.max_bin)
    }

    /// Centre frequency of bin `idx`, in Hz.
    ///
    /// # Panics
    /// Panics if `idx` is out of range.
    #[must_use]
    pub fn bin_frequency(&self, idx: usize) -> f32 {
        self.channelizer.bin_frequency(idx)
    }

    /// True once the current calibration interval has accumulated enough
    /// samples for [`detect`](Self::detect) to be meaningful.
    #[must_use]
    pub fn interval_complete(&self) -> bool {
        self.interval.len() >= self.interval_samples
    }

    /// Ghost-filtered, fractionally-interpolated tone frequencies for the
    /// current interval.
    #[must_use]
    pub fn detect(&self) -> Vec<f32> {
        let candidates = self.stats.detect(&self.cfg);
        let histories = self.candidate_histories(&candidates);
        let (bins, _ghosts) = suppress_correlated_ghosts(
            &candidates,
            &histories,
            &self.stats,
            self.ghost_bins,
            GHOST_MIN_DB,
            GHOST_CORR,
        );
        let spacing = self.channelizer.bin_spacing_hz();
        bins.iter()
            .map(|&b| self.channelizer.bin_frequency(b) + self.stats.peak_offset(b) * spacing)
            .collect()
    }

    /// Per-frame envelope history of each candidate bin over the current
    /// interval, built by re-channelizing the buffered interval input.
    /// One extra pass per interval, but only at the candidate bins —
    /// keeping live histories for the whole scan range would be hundreds
    /// of MB at SDR bin counts.
    fn candidate_histories(&self, candidates: &[usize]) -> Vec<Vec<f32>> {
        let mut hist: Vec<Vec<f32>> = vec![Vec::new(); candidates.len()];
        if candidates.is_empty() {
            return hist;
        }
        let mut chan = self.channelizer.fresh();
        for &sample in &self.interval {
            if let Some(bins) = chan.push(sample) {
                for (h, &b) in hist.iter_mut().zip(candidates) {
                    h.push(bins[b].norm());
                }
            }
        }
        hist
    }

    /// Raw input of the current interval — replay this into a channel
    /// spawned from this interval's detections, *before* calling
    /// [`reset_interval`](Self::reset_interval).
    #[must_use]
    pub fn interval_audio(&self) -> &[C::Input] {
        &self.interval
    }

    /// Start the next calibration interval. The channelizer keeps its
    /// ring state, so frames stay continuous across the boundary.
    pub fn reset_interval(&mut self) {
        self.stats = BinStats::new(self.channelizer.channel_count());
        self.interval.clear();
    }

    /// Elapsed source time represented by everything pushed so far is the
    /// caller's business; the detector only knows its sample rate.
    #[must_use]
    pub const fn sample_rate(&self) -> f32 {
        self.sample_rate
    }
}
