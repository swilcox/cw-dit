//! Adaptive timing: estimate the dot-unit and classify durations into
//! [`Element`]s and [`Gap`]s.
//!
//! Nominal Morse timing (ITU-R M.1677-1):
//!
//! | interval             | duration in dot units |
//! |----------------------|-----------------------|
//! | dit (mark)           | 1                     |
//! | dah (mark)           | 3                     |
//! | intra-character gap  | 1                     |
//! | inter-character gap  | 3                     |
//! | word gap             | 7                     |
//!
//! Classification uses the natural midpoints 2 T and 5 T as thresholds.

use crate::element::{Element, Gap};

/// Minimum dot-unit this estimator will hold, in caller-supplied duration
/// units. Prevents divide-by-zero after pathological input.
const MIN_UNIT: f32 = 1.0;

/// Weight given to a newly observed dit when adapting the dot-unit.
const DIT_ADAPT_ALPHA: f32 = 0.2;

/// Adaptive estimator of the Morse dot-unit `T`.
///
/// The estimator holds a single value — the current `T` — and answers two
/// kinds of question:
///
/// * [`classify_mark`](Self::classify_mark) — is this key-down interval a dit
///   or a dah?
/// * [`classify_gap`](Self::classify_gap) — is this key-up interval an
///   intra-character, inter-character, or word gap?
///
/// It can be nudged toward the true rate via
/// [`observe_mark`](Self::observe_mark), which updates `T` from observed dits.
///
/// The unit is held internally as `f32` so repeated small adaptation steps
/// accumulate cleanly; integer rounding at the classification boundary lets
/// callers work in whole samples.
#[derive(Debug, Clone)]
pub struct TimingEstimator {
    unit: f32,
}

impl TimingEstimator {
    /// Create an estimator with a dot-unit measured in the same duration units
    /// the caller will pass to [`classify_mark`](Self::classify_mark) and
    /// [`classify_gap`](Self::classify_gap).
    #[must_use]
    pub fn from_unit(unit: u32) -> Self {
        Self {
            unit: (unit as f32).max(MIN_UNIT),
        }
    }

    /// Create an estimator from a target rate in words-per-minute and a
    /// sample rate in Hertz. The resulting dot-unit is in samples.
    ///
    /// Uses the PARIS convention: 1 word == 50 dot units.
    #[must_use]
    pub fn from_wpm(wpm: f32, sample_rate_hz: f32) -> Self {
        assert!(wpm > 0.0, "wpm must be positive");
        assert!(sample_rate_hz > 0.0, "sample_rate_hz must be positive");
        Self {
            unit: (1.2 * sample_rate_hz / wpm).max(MIN_UNIT),
        }
    }

    /// Current dot-unit estimate in the caller's duration units, rounded to
    /// the nearest whole unit.
    #[must_use]
    pub fn unit(&self) -> u32 {
        self.unit.round() as u32
    }

    /// Effective WPM given a sample rate, using the PARIS convention.
    #[must_use]
    pub fn wpm(&self, sample_rate_hz: f32) -> f32 {
        1.2 * sample_rate_hz / self.unit
    }

    /// Classify a key-down interval as a dit or a dah using the current
    /// dot-unit estimate.
    #[must_use]
    pub fn classify_mark(&self, duration: u32) -> Element {
        // Threshold: 2 T. Below → dit, at or above → dah.
        if (duration as f32) < 2.0 * self.unit {
            Element::Dit
        } else {
            Element::Dah
        }
    }

    /// Classify a key-up interval as an intra-character, inter-character, or
    /// word gap using the current dot-unit estimate.
    #[must_use]
    pub fn classify_gap(&self, duration: u32) -> Gap {
        // Thresholds: 2 T (char) and 5 T (word).
        let d = duration as f32;
        if d < 2.0 * self.unit {
            Gap::IntraChar
        } else if d < 5.0 * self.unit {
            Gap::Char
        } else {
            Gap::Word
        }
    }

    /// Nudge the dot-unit estimate toward an observed key-down interval.
    ///
    /// Dah observations are used more conservatively than dit observations:
    /// the signal is noisier because dahs span 3 units and small errors in
    /// an operator's fist affect them proportionally less.
    pub fn observe_mark(&mut self, duration: u32, element: Element) {
        let target = match element {
            Element::Dit => duration as f32,
            // A dah is nominally 3 T; infer what T would make this dah exact.
            Element::Dah => duration as f32 / 3.0,
        };
        let alpha = match element {
            Element::Dit => DIT_ADAPT_ALPHA,
            Element::Dah => DIT_ADAPT_ALPHA * 0.5,
        };
        self.unit = ((1.0 - alpha) * self.unit + alpha * target).max(MIN_UNIT);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_mark_uses_2t_threshold() {
        let t = TimingEstimator::from_unit(60);
        // Well below 2T
        assert_eq!(t.classify_mark(50), Element::Dit);
        // At 2T boundary → dah
        assert_eq!(t.classify_mark(120), Element::Dah);
        // Well above 2T
        assert_eq!(t.classify_mark(180), Element::Dah);
    }

    #[test]
    fn classify_gap_uses_2t_and_5t_thresholds() {
        let t = TimingEstimator::from_unit(60);
        assert_eq!(t.classify_gap(60), Gap::IntraChar); // 1T
        assert_eq!(t.classify_gap(180), Gap::Char); // 3T
        assert_eq!(t.classify_gap(420), Gap::Word); // 7T
        // Boundaries
        assert_eq!(t.classify_gap(119), Gap::IntraChar);
        assert_eq!(t.classify_gap(120), Gap::Char);
        assert_eq!(t.classify_gap(299), Gap::Char);
        assert_eq!(t.classify_gap(300), Gap::Word);
    }

    #[test]
    fn from_wpm_matches_paris_convention() {
        // 20 WPM, 48 kHz sample rate → T = 1.2 * 48000 / 20 = 2880 samples
        let t = TimingEstimator::from_wpm(20.0, 48_000.0);
        assert_eq!(t.unit(), 2880);
        // Round-trip back to WPM
        assert!((t.wpm(48_000.0) - 20.0).abs() < 0.01);
    }

    #[test]
    fn observe_mark_nudges_unit_toward_dit() {
        let mut t = TimingEstimator::from_unit(100);
        // Operator is actually keying at T=50: feed a stream of 50-unit dits.
        for _ in 0..50 {
            t.observe_mark(50, Element::Dit);
        }
        assert!(
            (t.unit() as i32 - 50).abs() <= 2,
            "expected unit≈50, got {}",
            t.unit()
        );
    }

    #[test]
    fn observe_mark_handles_dah_by_dividing_by_three() {
        let mut t = TimingEstimator::from_unit(100);
        // True T=50 means dahs are 150.
        for _ in 0..200 {
            t.observe_mark(150, Element::Dah);
        }
        assert!(
            (t.unit() as i32 - 50).abs() <= 3,
            "expected unit≈50, got {}",
            t.unit()
        );
    }

    #[test]
    fn unit_never_drops_below_min() {
        let mut t = TimingEstimator::from_unit(2);
        for _ in 0..100 {
            t.observe_mark(0, Element::Dit);
        }
        assert!(t.unit() >= MIN_UNIT as u32);
    }
}
