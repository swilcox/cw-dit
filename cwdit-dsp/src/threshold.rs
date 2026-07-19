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

/// Default rail tracking rate for [`QuantileSlicer`], in dB of amplitude
/// per second. Fast enough to ride typical HF QSB (a few dB per second);
/// much faster and the rails start following the keying itself.
pub const DEFAULT_TRACK_DB_PER_S: f32 = 48.0;

/// Rail span (log2 units; 1.0 ≈ 6 dB) below which [`QuantileSlicer`]
/// squelches: mark and noise estimates this close together mean there is
/// no keyed signal to slice.
const SQUELCH_SPAN_LOG2: f32 = 1.0;

/// A fresh signal this far (log2) above the high rail engages fast
/// attack so acquisition takes a few marks, not seconds.
const FAST_ATTACK_MARGIN_LOG2: f32 = 1.0;
const FAST_ATTACK_ALPHA: f32 = 0.15;

/// Quantile probabilities for the low (noise) and high (mark) rails.
/// Marks occupy roughly a third to a half of active keying, so the 80th
/// percentile sits inside the mark mode and the 20th inside the noise.
const RAIL_P_LO: f32 = 0.2;
const RAIL_P_HI: f32 = 0.8;

/// Seconds of priming during which both rails EMA onto the observed
/// envelope and the output is forced off.
const RAIL_PRIMING_S: f32 = 0.3;

/// Quantile-tracking envelope slicer for fading channels.
///
/// [`Threshold`]'s peak tracker only *decays* on wall-clock time: once a
/// QSB dip drops the keyed level below the on-threshold, no marks are
/// detected, so nothing realigns the peak — the slicer deadlocks until
/// the decay catches up, and mid-fade copy is lost even at workable SNR.
///
/// This slicer instead tracks two quantiles of the log2 envelope with
/// Robbins–Monro steps — a low rail near the noise floor ([`RAIL_P_LO`])
/// and a high rail inside the mark level ([`RAIL_P_HI`]). The rails
/// observe every envelope sample regardless of slicing state, so they
/// follow a fade at the configured rate no matter what the slicer
/// currently thinks is a mark. Slicing then uses the same linear-domain
/// hysteresis fractions as [`Threshold`] over the rails' span, and a
/// pinched span (under ~6 dB) squelches the output entirely — the
/// noise-only-channel guard that [`Threshold`] gets from its SNR gate.
#[derive(Debug, Clone)]
pub struct QuantileSlicer {
    lo: f32,
    hi: f32,
    step: f32,
    on: bool,
    priming_left: u32,
}

impl QuantileSlicer {
    /// Create a slicer whose rails can follow a level change at
    /// `track_db_per_s` dB of amplitude per second.
    ///
    /// # Panics
    /// Panics if either argument is not positive.
    #[must_use]
    pub fn new(envelope_sample_rate_hz: f32, track_db_per_s: f32) -> Self {
        assert!(envelope_sample_rate_hz > 0.0);
        assert!(track_db_per_s > 0.0);
        // A rail's fast direction moves at step × max(p, 1-p) per sample;
        // scale so that direction meets the requested dB/s. (The slow
        // direction is 4× slower — that asymmetry is what pins each rail
        // to its quantile.)
        let db_per_sample = track_db_per_s / envelope_sample_rate_hz;
        let step = (db_per_sample / 6.02) / RAIL_P_HI;
        Self {
            lo: -20.0,
            hi: -10.0,
            step,
            on: false,
            priming_left: (RAIL_PRIMING_S * envelope_sample_rate_hz).ceil().max(1.0) as u32,
        }
    }

    /// Feed one envelope sample. Returns the current key state
    /// (`true` = key down / mark).
    pub fn push(&mut self, envelope: f32) -> bool {
        let x = envelope.max(1e-12).log2();
        if self.priming_left > 0 {
            self.priming_left -= 1;
            let a = 0.05;
            self.lo += a * (x - self.lo);
            self.hi += a * (x - self.hi);
            return false;
        }

        // Robbins–Monro quantile steps: q += step × (p − [x < q]).
        self.lo += self.step * (if x < self.lo { RAIL_P_LO - 1.0 } else { RAIL_P_LO });
        if x > self.hi + FAST_ATTACK_MARGIN_LOG2 {
            self.hi += FAST_ATTACK_ALPHA * (x - self.hi);
        } else {
            self.hi += self.step * (if x < self.hi { RAIL_P_HI - 1.0 } else { RAIL_P_HI });
        }
        if self.hi < self.lo {
            self.hi = self.lo;
        }

        if self.hi - self.lo < SQUELCH_SPAN_LOG2 {
            self.on = false;
            return false;
        }

        let lo_lin = self.lo.exp2();
        let span_lin = (self.hi.exp2() - lo_lin).max(0.0);
        if self.on {
            if envelope < lo_lin + DEFAULT_OFF_FRACTION * span_lin {
                self.on = false;
            }
        } else if envelope > lo_lin + DEFAULT_ON_FRACTION * span_lin {
            self.on = true;
        }
        self.on
    }

    /// Current mark-level (high-rail) estimate, linear. Exposed for tests.
    #[must_use]
    pub fn mark_level(&self) -> f32 {
        self.hi.exp2()
    }

    /// Current noise-level (low-rail) estimate, linear. Exposed for tests.
    #[must_use]
    pub fn noise_level(&self) -> f32 {
        self.lo.exp2()
    }
}

/// Either envelope slicer behind a single `push`, so decode chains can
/// pick per input domain ([`Threshold`] for near-full-scale audio,
/// [`QuantileSlicer`] for fading SDR IQ) without generics.
#[derive(Debug, Clone)]
pub enum Slicer {
    Classic(Threshold),
    Rails(QuantileSlicer),
}

impl Slicer {
    /// Feed one envelope sample. Returns the current key state.
    pub fn push(&mut self, envelope: f32) -> bool {
        match self {
            Self::Classic(t) => t.push(envelope),
            Self::Rails(q) => q.push(envelope),
        }
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

    // --- QuantileSlicer ---

    const Q_RATE: f32 = 200.0;

    /// Keyed envelope at IQ-like levels: `pattern` gives per-dit key
    /// states, `dit_ticks` envelope samples per dit, ±30% noise on the
    /// noise level and ±10% on the mark level.
    fn keyed_env(pattern: &[bool], dit_ticks: usize, mark: f32, noise_lvl: f32) -> Vec<f32> {
        let mut out = Vec::new();
        let mut seed = 7_u32;
        let mut rnd = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1 << 24) as f32
        };
        for &k in pattern {
            for _ in 0..dit_ticks {
                if k {
                    out.push(mark * (0.9 + 0.2 * rnd()));
                } else {
                    out.push(noise_lvl * (0.7 + 0.6 * rnd()));
                }
            }
        }
        out
    }

    /// Fraction of post-settle samples (priming plus rail acquisition,
    /// ~0.5 s) where the slicer output matches the key.
    fn key_agreement(s: &mut QuantileSlicer, pattern: &[bool], env: &[f32], dit_ticks: usize) -> f32 {
        let settle = (0.5 * Q_RATE) as usize;
        let mut hits = 0usize;
        for (i, &e) in env.iter().enumerate() {
            if s.push(e) == pattern[i / dit_ticks] && i >= settle {
                hits += 1;
            }
        }
        hits as f32 / (env.len() - settle) as f32
    }

    #[test]
    fn quantile_slicer_slices_clean_keying() {
        // "PARIS "-ish alternation at IQ envelope levels.
        let pattern: Vec<bool> = [true, false, true, true, true, false, false, false]
            .repeat(40);
        let env = keyed_env(&pattern, 12, 1e-4, 3e-5);
        let mut s = QuantileSlicer::new(Q_RATE, DEFAULT_TRACK_DB_PER_S);
        let agree = key_agreement(&mut s, &pattern, &env, 12);
        assert!(agree > 0.9, "agreement {agree}");
    }

    #[test]
    fn quantile_slicer_stays_off_on_noise() {
        let pattern = vec![false; 400];
        let env = keyed_env(&pattern, 12, 0.0, 3e-5);
        let mut s = QuantileSlicer::new(Q_RATE, DEFAULT_TRACK_DB_PER_S);
        let marks = env.iter().filter(|&&e| s.push(e)).count();
        assert!(
            marks < env.len() / 100,
            "noise-only channel produced {marks} mark ticks"
        );
    }

    #[test]
    fn quantile_slicer_follows_a_deep_fade() {
        // Keyed signal that fades 18 dB over ~2 s and stays there: rails
        // must re-acquire and keep slicing at the faded level.
        let pattern: Vec<bool> = [true, false, true, true, true, false, false, false]
            .repeat(80);
        let mut env = keyed_env(&pattern, 12, 1e-4, 3e-6);
        let n = env.len();
        for (i, e) in env.iter_mut().enumerate() {
            let t = i as f32 / n as f32;
            // Fade from 0 dB to −18 dB across the first half, hold after.
            let fade_db = 18.0 * (2.0 * t).min(1.0);
            let g = 10f32.powf(-fade_db / 20.0);
            if *e > 1e-5 {
                *e *= g;
            }
        }
        let mut s = QuantileSlicer::new(Q_RATE, DEFAULT_TRACK_DB_PER_S);
        // Agreement over the FADED second half is what matters.
        let half = n / 2;
        let mut hits = 0usize;
        for (i, &e) in env.iter().enumerate() {
            let mark = s.push(e);
            if i >= half && mark == pattern[i / 12] {
                hits += 1;
            }
        }
        let agree = hits as f32 / (n - half) as f32;
        assert!(agree > 0.85, "faded-half agreement {agree}");
    }

    #[test]
    fn quantile_slicer_rails_track_levels() {
        let pattern: Vec<bool> = [true, false].repeat(300);
        let env = keyed_env(&pattern, 12, 2e-4, 3e-5);
        let mut s = QuantileSlicer::new(Q_RATE, DEFAULT_TRACK_DB_PER_S);
        for &e in &env {
            s.push(e);
        }
        let mark = s.mark_level();
        let noise = s.noise_level();
        assert!(
            (1e-4..4e-4).contains(&mark),
            "mark rail {mark} not near 2e-4"
        );
        assert!(
            (1e-5..1e-4).contains(&noise),
            "noise rail {noise} not near 3e-5"
        );
    }
}
