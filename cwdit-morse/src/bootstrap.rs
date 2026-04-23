//! Dot-unit bootstrap wrapper around [`Decoder`].
//!
//! When the caller's initial WPM seed is close to the truth, [`Decoder`]'s
//! adaptive timing closes the gap over a few characters. But a seed that's
//! ~2× off or more gets early characters wrong: dahs of a fast channel get
//! misclassified as dits of a slower seed, or intra-character gaps of a
//! slow channel get split into word gaps by a faster seed. In a band scan
//! where each channel's speed is unknown, that first-character damage is
//! unavoidable without looking at the input before classifying it.
//!
//! [`BootstrapDecoder`] buffers the first `target_marks` mark events
//! (along with any interleaved gaps), estimates `T` from the observed mark
//! durations, then replays the buffered events through a freshly-seeded
//! [`Decoder`]. After bootstrap the wrapper is a thin pass-through.
//!
//! The estimator takes the median of the shorter half of sorted mark
//! durations as `T` — a robust approximation of the dit peak in the
//! bimodal dit/dah distribution of English Morse text, not sensitive to a
//! few spurious short or long marks.

use crate::decoder::Decoder;
use crate::element::Decoded;
use crate::timing::TimingEstimator;

/// Default number of marks buffered before bootstrap fires. Two or three
/// typical characters — enough to see a mix of dits and dahs without
/// delaying output noticeably.
pub const DEFAULT_BOOTSTRAP_MARKS: u32 = 8;

/// Wraps [`Decoder`] with a calibration phase that derives the initial
/// dot-unit from the input stream itself.
#[derive(Debug, Clone)]
pub struct BootstrapDecoder {
    inner: Option<Decoder>,
    seed_timing: TimingEstimator,
    adapt: bool,
    target_marks: u32,
    marks_observed: u32,
    buffered: Vec<(bool, u32)>,
}

impl BootstrapDecoder {
    /// Create a bootstrap decoder. `seed_timing` is only used as a fallback
    /// if the input ends before `target_marks` marks are observed.
    #[must_use]
    pub fn new(seed_timing: TimingEstimator) -> Self {
        Self {
            inner: None,
            seed_timing,
            adapt: true,
            target_marks: DEFAULT_BOOTSTRAP_MARKS,
            marks_observed: 0,
            buffered: Vec::new(),
        }
    }

    /// Override the number of marks buffered before bootstrap fires.
    ///
    /// # Panics
    /// Panics if `n` is zero.
    #[must_use]
    pub fn with_target_marks(mut self, n: u32) -> Self {
        assert!(n > 0, "target_marks must be positive");
        self.target_marks = n;
        self
    }

    /// Enable or disable continuing EMA-based adaptation after bootstrap.
    #[must_use]
    pub const fn with_adapt(mut self, adapt: bool) -> Self {
        self.adapt = adapt;
        self
    }

    /// Has the bootstrap phase completed? Before completion, [`push`] only
    /// buffers events and returns an empty [`Vec`].
    #[must_use]
    pub const fn is_bootstrapped(&self) -> bool {
        self.inner.is_some()
    }

    /// Current dot-unit estimate. Returns the seed value during bootstrap
    /// and the inner decoder's live value afterwards.
    #[must_use]
    pub fn timing(&self) -> &TimingEstimator {
        match &self.inner {
            Some(dec) => dec.timing(),
            None => &self.seed_timing,
        }
    }

    /// Feed one key-up / key-down interval. Returns decoded events, which
    /// may be empty during bootstrap and possibly large on the frame that
    /// completes bootstrap (since the whole buffered history is flushed at
    /// once).
    pub fn push(&mut self, mark: bool, duration: u32) -> Vec<Decoded> {
        if let Some(inner) = &mut self.inner {
            return inner.push(mark, duration).into_iter().collect();
        }

        self.buffered.push((mark, duration));
        if mark {
            self.marks_observed += 1;
        }
        if self.marks_observed < self.target_marks {
            return Vec::new();
        }
        self.finalize_bootstrap();
        self.replay_buffered()
    }

    /// Flush any in-progress character. Also finalises bootstrap from
    /// whatever marks were observed if the input ended early.
    pub fn finish(&mut self) -> Vec<Decoded> {
        if self.inner.is_none() {
            self.finalize_bootstrap();
            let mut out = self.replay_buffered();
            if let Some(inner) = &mut self.inner {
                out.extend(inner.finish());
            }
            return out;
        }
        self.inner.as_mut().unwrap().finish().into_iter().collect()
    }

    fn finalize_bootstrap(&mut self) {
        let mark_durations: Vec<u32> = self
            .buffered
            .iter()
            .filter_map(|(m, d)| m.then_some(*d))
            .collect();
        let timing = if mark_durations.is_empty() {
            self.seed_timing.clone()
        } else {
            let unit = estimate_unit_from_marks(&mark_durations);
            TimingEstimator::from_unit(unit)
        };
        self.inner = Some(Decoder::new(timing).with_adapt(self.adapt));
    }

    fn replay_buffered(&mut self) -> Vec<Decoded> {
        let events = std::mem::take(&mut self.buffered);
        let inner = self
            .inner
            .as_mut()
            .expect("finalize_bootstrap must run before replay");
        let mut out = Vec::new();
        for (mark, duration) in events {
            out.extend(inner.push(mark, duration));
        }
        out
    }
}

/// Estimate the Morse dot-unit from a set of observed mark durations.
///
/// Takes the median of the shorter half — the peak of the dit mode in a
/// bimodal distribution, robust to a small number of outliers in either
/// direction. Assumes at least a handful of dits are present in the
/// sample; with too few marks, returns whichever value sits at the
/// quarter-point of the sorted durations.
fn estimate_unit_from_marks(durations: &[u32]) -> u32 {
    assert!(!durations.is_empty());
    let mut sorted: Vec<u32> = durations.to_vec();
    sorted.sort_unstable();
    let half = sorted.len() / 2;
    let lower = if half == 0 { &sorted[..1] } else { &sorted[..half] };
    lower[lower.len() / 2].max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet;

    /// Build a run-length stream for `text` at dot-unit `t`. Same structure
    /// as the helper in `decoder::tests::synth` but parameterised by `t`.
    fn synth(text: &str, t: u32) -> Vec<(bool, u32)> {
        let mut out = Vec::new();
        let mut first_word = true;
        for word in text.split(' ') {
            if !first_word {
                out.push((false, 7 * t));
            }
            first_word = false;
            let mut first_char = true;
            for ch in word.chars() {
                if !first_char {
                    out.push((false, 3 * t));
                }
                first_char = false;
                let pattern = alphabet::pattern_for_char(ch).expect("known char");
                let mut first_elem = true;
                for glyph in pattern.chars() {
                    if !first_elem {
                        out.push((false, t));
                    }
                    first_elem = false;
                    let dur = match glyph {
                        '.' => t,
                        '-' => 3 * t,
                        _ => unreachable!(),
                    };
                    out.push((true, dur));
                }
            }
        }
        out
    }

    fn run(dec: &mut BootstrapDecoder, events: &[(bool, u32)]) -> Vec<Decoded> {
        let mut out = Vec::new();
        for &(m, d) in events {
            out.extend(dec.push(m, d));
        }
        out.extend(dec.finish());
        out
    }

    fn decoded_string(events: &[Decoded]) -> String {
        events
            .iter()
            .map(|e| match e {
                Decoded::Char(c) => *c,
                Decoded::WordBreak => ' ',
                Decoded::Unknown => '?',
            })
            .collect()
    }

    #[test]
    fn bootstrap_rescues_badly_seeded_fast_channel() {
        // Seed says T=100 (slow); actual T=20 (fast). A naive decoder
        // would classify early dahs (60 units) as dits (< 2*100=200).
        let mut dec =
            BootstrapDecoder::new(TimingEstimator::from_unit(100)).with_target_marks(8);
        let text = "CQ DE W1AW";
        let events = synth(text, 20);
        let out = run(&mut dec, &events);
        assert_eq!(decoded_string(&out), text);
    }

    #[test]
    fn bootstrap_rescues_badly_seeded_slow_channel() {
        // Seed says T=10 (fast); actual T=100 (slow). A naive decoder
        // would split characters because intra-char gaps (100 units) look
        // like word gaps under a T=10 seed (5*10=50 < 100).
        let mut dec = BootstrapDecoder::new(TimingEstimator::from_unit(10)).with_target_marks(8);
        let text = "CQ DE W1AW";
        let events = synth(text, 100);
        let out = run(&mut dec, &events);
        assert_eq!(decoded_string(&out), text);
    }

    #[test]
    fn bootstrap_passes_through_after_calibration() {
        let mut dec = BootstrapDecoder::new(TimingEstimator::from_unit(1)).with_target_marks(4);
        let events = synth("SOS TEST", 1);
        let out = run(&mut dec, &events);
        assert_eq!(decoded_string(&out), "SOS TEST");
        assert!(dec.is_bootstrapped());
    }

    #[test]
    fn finish_on_short_input_falls_back_to_seed() {
        // Only two marks before the stream ends; bootstrap never triggers.
        // Fallback to the seed is used.
        let mut dec = BootstrapDecoder::new(TimingEstimator::from_unit(1)).with_target_marks(8);
        // "E" = single dit, then stream ends.
        let out = run(&mut dec, &[(true, 1)]);
        assert_eq!(decoded_string(&out), "E");
    }

    #[test]
    fn emits_accurate_wpm_after_bootstrap() {
        // At env_rate=200 Hz, T=20 samples → 1.2*200/20 = 12 WPM.
        let mut dec = BootstrapDecoder::new(TimingEstimator::from_unit(100)).with_target_marks(8);
        let events = synth("CQ DE W1AW", 20);
        let _ = run(&mut dec, &events);
        let wpm = dec.timing().wpm(200.0);
        // Adaptation may nudge the estimate slightly during playback; just
        // check it's in the right ballpark.
        assert!((wpm - 12.0).abs() < 2.0, "wpm estimate = {wpm}");
    }

    #[test]
    fn unit_estimator_picks_dit_from_mixed_marks() {
        // 3 dits (20) and 3 dahs (60); algorithm should pick ~20.
        let unit = estimate_unit_from_marks(&[60, 20, 60, 20, 60, 20]);
        assert_eq!(unit, 20);
    }

    #[test]
    fn unit_estimator_robust_to_one_outlier() {
        // One spuriously short mark among otherwise sensible dit/dah data.
        let unit = estimate_unit_from_marks(&[2, 20, 20, 20, 60, 60, 60]);
        // Median of lower half ([2, 20, 20]) = 20 — the spurious 2 is
        // ignored.
        assert_eq!(unit, 20);
    }
}
