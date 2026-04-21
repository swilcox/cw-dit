//! Envelope slicer — turn a continuous envelope into a key-up / key-down
//! boolean stream.
//!
//! The slicer tracks an exponentially-decaying peak of the envelope and
//! gates the output with hysteresis (`on_fraction` > `off_fraction`). This
//! handles clean CW recordings well; a more elaborate adaptive scheme with
//! separate noise-floor tracking can replace it later if weak-signal
//! performance matters.

/// Default fraction of `peak` above which the slicer switches to key-down.
pub const DEFAULT_ON_FRACTION: f32 = 0.55;

/// Default fraction of `peak` below which the slicer switches to key-up.
pub const DEFAULT_OFF_FRACTION: f32 = 0.35;

/// Hysteretic envelope slicer.
#[derive(Debug, Clone)]
pub struct Threshold {
    peak: f32,
    decay_per_sample: f32,
    on_fraction: f32,
    off_fraction: f32,
    min_peak: f32,
    absolute_on_floor: f32,
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

        Self {
            peak: min_peak,
            decay_per_sample,
            on_fraction: DEFAULT_ON_FRACTION,
            off_fraction: DEFAULT_OFF_FRACTION,
            min_peak,
            absolute_on_floor: 0.0,
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
    /// fraction of `peak`; `on_fraction` must be > `off_fraction`.
    #[must_use]
    pub fn with_hysteresis(mut self, on_fraction: f32, off_fraction: f32) -> Self {
        assert!(on_fraction > off_fraction);
        assert!(off_fraction > 0.0);
        self.on_fraction = on_fraction;
        self.off_fraction = off_fraction;
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

        let on_thresh = (self.peak * self.on_fraction).max(self.absolute_on_floor);
        let off_thresh = self.peak * self.off_fraction;

        if self.state {
            if envelope < off_thresh || envelope < self.absolute_on_floor * 0.5 {
                self.state = false;
            }
        } else if envelope > on_thresh {
            self.state = true;
        }

        self.state
    }

    /// Current peak estimate. Exposed for tests and debugging.
    #[must_use]
    pub const fn peak(&self) -> f32 {
        self.peak
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(t: &mut Threshold, env: &[f32]) -> Vec<bool> {
        env.iter().map(|&e| t.push(e)).collect()
    }

    #[test]
    fn follows_simple_on_off_pattern() {
        let mut t = Threshold::new(100.0, 1.0, 0.01);
        // Envelope: 20 samples high at 1.0, 20 low at 0.0, repeat.
        let mut env = Vec::new();
        env.extend(std::iter::repeat_n(1.0_f32, 20));
        env.extend(std::iter::repeat_n(0.0_f32, 20));
        env.extend(std::iter::repeat_n(1.0_f32, 20));

        let out = run(&mut t, &env);

        // After a short settling period the output should be high in the
        // high blocks and low in the low blocks.
        assert!(out[10], "expected on during first high block");
        assert!(!out[30], "expected off during low block");
        assert!(out[50], "expected on during second high block");
    }

    #[test]
    fn hysteresis_prevents_mid_threshold_chatter() {
        let mut t = Threshold::new(100.0, 1.0, 0.01);
        // Warm up with a strong tone so peak establishes.
        for _ in 0..50 {
            t.push(1.0);
        }
        // Now feed values that sit right at the nominal threshold.
        // With on=0.55, off=0.35 and peak≈1, a value of 0.45 must not
        // toggle the slicer once it's in the off state.
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
}
