//! Run-length debouncer — absorb glitch runs the slicer let through.
//!
//! Even with a smoothed envelope and an SNR-gated slicer, a marginal
//! channel produces occasional runs far shorter than any real Morse
//! element: a one-tick mark from a noise spike, or a one-tick space
//! punched into a dah by a fade. Downstream, each of those corrupts a
//! character *and* — worse — feeds the adaptive timing estimator a bogus
//! "dit".
//!
//! [`Debouncer`] sits between [`RunLengthEncoder`](crate::RunLengthEncoder)
//! and the decoder. Any run shorter than `min_run` ticks is merged into the
//! run before it (and, since runs alternate, the follow-on run of the
//! original state then merges too): short marks in a long space vanish,
//! short dropouts inside a mark are bridged. Callers size `min_run` well
//! below one dit — a quarter to a fifth — so legitimate elements are never
//! touched.

use crate::runlength::Run;

/// Streaming run-length debouncer.
#[derive(Debug, Clone)]
pub struct Debouncer {
    min_run: u32,
    pending: Option<Run>,
}

impl Debouncer {
    /// Create a debouncer that absorbs runs shorter than `min_run` ticks.
    /// `min_run == 1` is a pass-through (every run is at least one tick).
    ///
    /// # Panics
    /// Panics if `min_run` is zero.
    #[must_use]
    pub fn new(min_run: u32) -> Self {
        assert!(min_run >= 1, "min_run must be at least 1");
        Self {
            min_run,
            pending: None,
        }
    }

    /// Feed one run. Returns the previous run once it can no longer be
    /// extended by glitch absorption.
    pub fn push(&mut self, run: Run) -> Option<Run> {
        match &mut self.pending {
            None => {
                self.pending = Some(run);
                None
            }
            // Same state as pending: extend it. Happens after a glitch of
            // the opposite state was absorbed in between.
            Some(p) if p.mark == run.mark => {
                p.duration += run.duration;
                None
            }
            Some(p) => {
                if run.duration < self.min_run {
                    // Glitch: bridge it into the pending run.
                    p.duration += run.duration;
                    None
                } else {
                    let out = *p;
                    self.pending = Some(run);
                    Some(out)
                }
            }
        }
    }

    /// Flush the in-progress run, if any. Call at end-of-stream.
    pub fn finish(&mut self) -> Option<Run> {
        self.pending.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mark(duration: u32) -> Run {
        Run {
            mark: true,
            duration,
        }
    }

    fn space(duration: u32) -> Run {
        Run {
            mark: false,
            duration,
        }
    }

    fn drive(min_run: u32, runs: &[Run]) -> Vec<Run> {
        let mut d = Debouncer::new(min_run);
        let mut out: Vec<Run> = runs.iter().filter_map(|&r| d.push(r)).collect();
        out.extend(d.finish());
        out
    }

    #[test]
    fn passes_clean_runs_through() {
        let runs = [mark(10), space(10), mark(30), space(70)];
        assert_eq!(drive(3, &runs), runs);
    }

    #[test]
    fn min_run_one_is_passthrough() {
        let runs = [mark(1), space(1), mark(2)];
        assert_eq!(drive(1, &runs), runs);
    }

    #[test]
    fn drops_isolated_mark_glitch_in_space() {
        // A 1-tick noise blip inside a long silence disappears entirely and
        // the silence is reported as one run.
        let runs = [space(50), mark(1), space(50)];
        assert_eq!(drive(3, &runs), vec![space(101)]);
    }

    #[test]
    fn bridges_dropout_inside_mark() {
        // A 2-tick fade splitting a dah is healed back into one mark.
        let runs = [space(10), mark(15), space(2), mark(14), space(10)];
        assert_eq!(drive(3, &runs), vec![space(10), mark(31), space(10)]);
    }

    #[test]
    fn absorbs_consecutive_glitches() {
        // A burst of alternating sub-min runs all folds into the run that
        // preceded it.
        let runs = [space(40), mark(1), space(2), mark(1), space(40)];
        assert_eq!(drive(3, &runs), vec![space(84)]);
    }

    #[test]
    fn preserves_durations_across_bridging() {
        // Total ticks in must equal total ticks out.
        let runs = [space(5), mark(9), space(1), mark(2), space(20)];
        let out = drive(3, &runs);
        let total_in: u32 = runs.iter().map(|r| r.duration).sum();
        let total_out: u32 = out.iter().map(|r| r.duration).sum();
        assert_eq!(total_in, total_out);
    }
}
