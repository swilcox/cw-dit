//! Collapse a boolean key-state stream into `(mark, duration)` run-length
//! events suitable for feeding into `cwdit-morse`'s decoder.
//!
//! Durations are counted in "ticks" — whatever unit the caller is using for
//! the boolean stream. When driven from the envelope slicer the tick rate
//! equals the envelope sample rate (input sample rate / block length).

/// Streaming run-length encoder.
#[derive(Debug, Clone, Default)]
pub struct RunLengthEncoder {
    current: Option<bool>,
    run: u32,
}

/// One `(mark, duration)` event emitted by [`RunLengthEncoder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Run {
    /// `true` = key down / mark, `false` = key up / space.
    pub mark: bool,
    /// Run length in ticks (≥ 1).
    pub duration: u32,
}

impl RunLengthEncoder {
    /// Create a new encoder with no in-progress run.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            current: None,
            run: 0,
        }
    }

    /// Feed one boolean sample. Returns `Some(run)` whenever the state flips;
    /// otherwise extends the current run and returns `None`.
    pub fn push(&mut self, sample: bool) -> Option<Run> {
        match self.current {
            None => {
                self.current = Some(sample);
                self.run = 1;
                None
            }
            Some(cur) if cur == sample => {
                self.run += 1;
                None
            }
            Some(cur) => {
                let run = Run {
                    mark: cur,
                    duration: self.run,
                };
                self.current = Some(sample);
                self.run = 1;
                Some(run)
            }
        }
    }

    /// Flush the in-progress run, if any. Call at end-of-stream.
    pub fn finish(&mut self) -> Option<Run> {
        self.current.take().map(|m| {
            let run = Run {
                mark: m,
                duration: self.run,
            };
            self.run = 0;
            run
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drive(samples: &[bool]) -> Vec<Run> {
        let mut rle = RunLengthEncoder::new();
        let mut out: Vec<Run> = samples.iter().filter_map(|&s| rle.push(s)).collect();
        out.extend(rle.finish());
        out
    }

    #[test]
    fn encodes_alternating_runs() {
        let samples = [true, true, true, false, false, true];
        assert_eq!(
            drive(&samples),
            vec![
                Run { mark: true, duration: 3 },
                Run { mark: false, duration: 2 },
                Run { mark: true, duration: 1 },
            ],
        );
    }

    #[test]
    fn single_run_only_emitted_at_finish() {
        let samples = vec![false; 5];
        assert_eq!(
            drive(&samples),
            vec![Run { mark: false, duration: 5 }],
        );
    }

    #[test]
    fn empty_stream_emits_nothing() {
        let out: Vec<Run> = drive(&[]);
        assert!(out.is_empty());
    }
}
