//! Envelope slicer — turn a continuous envelope into a key-up / key-down
//! boolean stream.
//!
//! Two trackers run side by side:
//!
//! * a fast-attack, exponentially-decaying **peak** tracker, and
//! * a **noise floor** tracker: an EMA of the envelope that only updates
//!   while the key is up, so it settles on the mean of the noise between
//!   marks and cannot ride up onto the keyed tone.
//!
//! The on/off hysteresis thresholds sit at fractions of the span between
//! floor and peak, so they ride above the noise instead of above zero. On
//! top of that an **SNR gate** requires the peak to clear the floor by a
//! minimum ratio before the slicer will report key-down — on a channel
//! carrying only noise the peak hugs the floor and the gate stays shut.
//!
//! The gate assumes the envelope has been smoothed (see
//! [`MovingAverage`](crate::smooth::MovingAverage)) to roughly a third of a
//! dit: raw single-block Goertzel noise is Rayleigh-distributed and its
//! excursions above the mean are wide enough to sneak past any usable gate
//! ratio; smoothing tightens the distribution so the default ~6 dB gate
//! separates noise from signal cleanly.
//!
//! For the first [`priming`](Threshold::new) fraction of a second the floor
//! is a low quantile of the envelope seen so far and the output is forced
//! to key-up. This snaps the floor onto the ambient noise level immediately
//! instead of waiting a time constant for the EMA to get there — the
//! difference between squelching a noisy channel from the first tick and
//! spewing garbage over it for the first second. A quantile rather than the
//! mean, because the stream may open mid-signal (a wide FFT window smears
//! keying into the very first frame): the quantile latches onto the
//! envelope dips and ignores the marks, where a mean would sit uselessly
//! between the two.

/// Default fraction of the floor→peak span above which the slicer switches
/// to key-down.
pub const DEFAULT_ON_FRACTION: f32 = 0.55;

/// Default fraction of the floor→peak span below which the slicer switches
/// to key-up.
pub const DEFAULT_OFF_FRACTION: f32 = 0.35;

/// Default minimum peak/floor ratio (~8 dB) required before the slicer
/// will report key-down. Empirically the tightest value that stays shut
/// across several seconds of dit-smoothed band noise; anything looser lets
/// occasional noise excursions through as spurious dits.
pub const DEFAULT_SNR_GATE: f32 = 2.5;

/// Time constant, in seconds, of the noise-floor EMA (key-up samples only).
const FLOOR_TC_S: f32 = 0.5;

/// Length of the priming phase in seconds. Matches the shortest lead-in
/// silence the synth fixtures use, so priming never eats keyed audio.
const PRIMING_S: f32 = 0.1;

/// Hysteretic envelope slicer with noise-floor tracking and an SNR gate.
#[derive(Debug, Clone)]
pub struct Threshold {
    peak: f32,
    floor: f32,
    decay_per_sample: f32,
    floor_alpha: f32,
    on_fraction: f32,
    off_fraction: f32,
    min_peak: f32,
    absolute_on_floor: f32,
    snr_gate: f32,
    /// Samples remaining in the priming phase. While non-zero the floor is
    /// a low quantile of the envelope so far and the output is forced off.
    priming_left: u32,
    /// Envelope samples collected during priming; drained when it ends.
    priming_buf: Vec<f32>,
    state: bool,
}

impl Threshold {
    /// Create a slicer whose peak-detector half-life is `peak_half_life_s`
    /// seconds at the given envelope sample rate.
    ///
    /// `min_peak` is a noise-floor guard: the peak never decays below this
    /// value, so pure silence never flips the slicer on from numerical noise.
    #[must_use]
    pub fn new(envelope_sample_rate_hz: f32, peak_half_life_s: f32, min_peak: f32) -> Self {
        assert!(envelope_sample_rate_hz > 0.0);
        assert!(peak_half_life_s > 0.0);
        assert!(min_peak >= 0.0);

        // decay^(half_life_samples) = 0.5
        let half_life_samples = peak_half_life_s * envelope_sample_rate_hz;
        let decay_per_sample = 0.5_f32.powf(1.0 / half_life_samples);
        let floor_alpha = 1.0 - (-1.0 / (FLOOR_TC_S * envelope_sample_rate_hz)).exp();

        Self {
            peak: min_peak,
            floor: 0.0,
            decay_per_sample,
            floor_alpha,
            on_fraction: DEFAULT_ON_FRACTION,
            off_fraction: DEFAULT_OFF_FRACTION,
            min_peak,
            absolute_on_floor: 0.0,
            snr_gate: DEFAULT_SNR_GATE,
            priming_left: (PRIMING_S * envelope_sample_rate_hz).ceil().max(1.0) as u32,
            priming_buf: Vec::new(),
            state: false,
        }
    }

    /// Require the envelope to exceed `floor` — in the same units as the
    /// input envelope — before the slicer will ever turn on, regardless of
    /// the peak-tracker's relative threshold. This suppresses false
    /// detections on quiet channels that see only sidelobe leakage from
    /// strong nearby signals: the peak-tracker happily rises to match that
    /// leakage, and a relative threshold alone is not enough to reject it.
    #[must_use]
    pub fn with_absolute_on_floor(mut self, floor: f32) -> Self {
        assert!(floor >= 0.0);
        self.absolute_on_floor = floor;
        self
    }

    /// Override the on/off hysteresis fractions. Both are expressed as a
    /// fraction of the floor→peak span; `on_fraction` must be >
    /// `off_fraction`.
    #[must_use]
    pub fn with_hysteresis(mut self, on_fraction: f32, off_fraction: f32) -> Self {
        assert!(on_fraction > off_fraction);
        assert!(off_fraction > 0.0);
        self.on_fraction = on_fraction;
        self.off_fraction = off_fraction;
        self
    }

    /// Override the SNR gate: the peak must exceed `ratio` × floor before
    /// the slicer will report key-down. `1.0` disables the gate.
    ///
    /// # Panics
    /// Panics if `ratio` < 1.0.
    #[must_use]
    pub fn with_snr_gate(mut self, ratio: f32) -> Self {
        assert!(ratio >= 1.0);
        self.snr_gate = ratio;
        self
    }

    /// Feed one envelope sample. Returns the current key state
    /// (`true` = key down / mark).
    pub fn push(&mut self, envelope: f32) -> bool {
        if envelope > self.peak {
            self.peak = envelope;
        } else {
            self.peak = (self.peak * self.decay_per_sample).max(self.min_peak);
        }

        if self.priming_left > 0 {
            // Lower-quartile of everything seen so far: the noise level
            // when the stream opens on noise, the inter-element dips when
            // it opens on a signal already in progress.
            self.priming_buf.push(envelope);
            let mut sorted = self.priming_buf.clone();
            sorted.sort_unstable_by(f32::total_cmp);
            self.floor = sorted[sorted.len() / 4];
            self.priming_left -= 1;
            if self.priming_left == 0 {
                self.priming_buf = Vec::new();
            }
            return false;
        }

        if !self.state {
            self.floor =
                (self.floor + self.floor_alpha * (envelope - self.floor)).min(self.peak);
        }

        let span = (self.peak - self.floor).max(0.0);
        let on_thresh = (self.floor + self.on_fraction * span).max(self.absolute_on_floor);
        let off_thresh = self.floor + self.off_fraction * span;
        let gate_open = self.peak > self.snr_gate * self.floor;

        if self.state {
            if !gate_open || envelope < off_thresh || envelope < self.absolute_on_floor * 0.5 {
                self.state = false;
            }
        } else if gate_open && envelope > on_thresh {
            self.state = true;
        }

        self.state
    }

    /// Current peak estimate. Exposed for tests and debugging.
    #[must_use]
    pub const fn peak(&self) -> f32 {
        self.peak
    }

    /// Current noise-floor estimate. Exposed for tests and debugging.
    #[must_use]
    pub const fn floor(&self) -> f32 {
        self.floor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(t: &mut Threshold, env: &[f32]) -> Vec<bool> {
        env.iter().map(|&e| t.push(e)).collect()
    }

    /// Deterministic pseudo-noise around `mean` with ±30% spread —
    /// resembles a Goertzel noise envelope after dit-scale smoothing.
    fn noise(mean: f32, n: usize) -> Vec<f32> {
        let mut state = 0x9E37_79B9_u32;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let u = (state >> 8) as f32 / 16_777_216.0;
                mean * (0.7 + 0.6 * u)
            })
            .collect()
    }

    #[test]
    fn follows_simple_on_off_pattern() {
        // At 100 Hz, priming is 10 samples; give a silent lead-in to cover
        // it (like the lead silence every real recording has).
        let mut t = Threshold::new(100.0, 1.0, 0.01);
        let mut env = vec![0.0_f32; 12];
        env.extend(std::iter::repeat_n(1.0_f32, 20));
        env.extend(std::iter::repeat_n(0.0_f32, 20));
        env.extend(std::iter::repeat_n(1.0_f32, 20));

        let out = run(&mut t, &env);

        assert!(out[22], "expected on during first high block");
        assert!(!out[42], "expected off during low block");
        assert!(out[62], "expected on during second high block");
    }

    #[test]
    fn hysteresis_prevents_mid_threshold_chatter() {
        let mut t = Threshold::new(100.0, 1.0, 0.01);
        // Silent lead-in past priming, then a strong keyed warm-up so the
        // peak establishes at ~1 with the floor still near zero.
        for _ in 0..12 {
            t.push(0.0);
        }
        for _ in 0..50 {
            t.push(1.0);
        }
        // Values at the nominal mid-threshold must not toggle the slicer
        // back on once it has gone off.
        t.push(0.0); // force off
        for _ in 0..20 {
            assert!(!t.push(0.45), "should remain off at 0.45");
        }
    }

    #[test]
    fn silence_does_not_flip_on() {
        let mut t = Threshold::new(100.0, 1.0, 0.01);
        for _ in 0..1_000 {
            assert!(!t.push(0.0), "silence must never produce key-down");
        }
    }

    #[test]
    fn pure_noise_stays_squelched() {
        // Noise only: the peak never clears the SNR gate over the floor,
        // so the slicer must stay off for the whole stream.
        let mut t = Threshold::new(200.0, 1.0, 0.005);
        let env = noise(0.1, 4_000);
        let out = run(&mut t, &env);
        assert!(
            out.iter().all(|&s| !s),
            "noise-only envelope produced key-down"
        );
    }

    #[test]
    fn keying_over_noise_is_sliced() {
        // Noise floor ~0.1, keyed tone at 1.0 — a strong signal must open
        // the gate and slice cleanly after a long noise-only lead-in.
        let mut t = Threshold::new(200.0, 1.0, 0.005);
        let mut env = noise(0.1, 400); // 2 s noise lead-in
        let n0 = env.len();
        for (i, e) in noise(0.1, 240).into_iter().enumerate() {
            // Alternate 40 samples keyed, 40 samples gap.
            let keyed = (i / 40) % 2 == 0;
            env.push(if keyed { 1.0 } else { e });
        }
        let out = run(&mut t, &env);
        assert!(
            out[..n0].iter().all(|&s| !s),
            "gate opened during the noise lead-in"
        );
        assert!(out[n0 + 20], "expected on mid-mark");
        assert!(!out[n0 + 60], "expected off mid-gap");
        assert!(out[n0 + 100], "expected on in second mark");
    }

    #[test]
    fn floor_tracks_noise_mean() {
        let mut t = Threshold::new(200.0, 1.0, 0.005);
        for e in noise(0.1, 2_000) {
            t.push(e);
        }
        let floor = t.floor();
        assert!(
            (0.07..=0.13).contains(&floor),
            "floor should sit near the noise mean (~0.1), got {floor}"
        );
    }

    #[test]
    fn floor_does_not_ride_up_during_marks() {
        let mut t = Threshold::new(200.0, 1.0, 0.005);
        for e in noise(0.05, 100) {
            t.push(e); // prime + settle on quiet noise
        }
        for _ in 0..80 {
            t.push(1.0); // a long dah of key-down
        }
        assert!(
            t.floor() < 0.2,
            "floor must freeze while the key is down, got {}",
            t.floor()
        );
    }
}
