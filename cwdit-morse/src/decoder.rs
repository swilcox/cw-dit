//! Streaming Morse decoder.
//!
//! The decoder consumes a run-length stream of key-up / key-down intervals —
//! one call to [`Decoder::push`] per interval — and emits [`Decoded`] events
//! as characters and word breaks are recognised.
//!
//! The decoder is agnostic to the unit of `duration`: samples, milliseconds,
//! and arbitrary "dot-units" all work, as long as the [`TimingEstimator`]
//! supplied to the constructor uses the same unit.

use crate::alphabet;
use crate::element::{Decoded, Gap};
use crate::timing::TimingEstimator;

/// Maximum number of glyphs in a single Morse character. The longest standard
/// patterns are 6 elements (`.-.-.-` for `.`); 10 leaves plenty of slack for
/// prosigns and noisy input without unbounded growth.
const MAX_PATTERN: usize = 10;

/// Streaming Morse decoder.
#[derive(Debug, Clone)]
pub struct Decoder {
    timing: TimingEstimator,
    pattern: heapless_pattern::Pattern,
    adapt: bool,
    /// Whether any character has been emitted yet. Suppresses
    /// leading `WordBreak` events produced by an initial run of silence.
    any_char_emitted: bool,
    /// Set when a word gap arrives; emitted lazily before the next character
    /// so trailing silence does not produce a dangling `WordBreak`.
    pending_word_break: bool,
}

impl Decoder {
    /// Create a decoder seeded with a timing estimate. The estimator will
    /// adapt as input arrives unless [`with_adapt(false)`](Self::with_adapt)
    /// is set.
    #[must_use]
    pub fn new(timing: TimingEstimator) -> Self {
        Self {
            timing,
            pattern: heapless_pattern::Pattern::new(),
            adapt: true,
            any_char_emitted: false,
            pending_word_break: false,
        }
    }

    /// Enable or disable dot-unit adaptation. Disabled estimators are useful
    /// for tests and for machine-generated input where the rate is fixed.
    #[must_use]
    pub const fn with_adapt(mut self, adapt: bool) -> Self {
        self.adapt = adapt;
        self
    }

    /// Borrow the current timing estimator.
    #[must_use]
    pub const fn timing(&self) -> &TimingEstimator {
        &self.timing
    }

    /// Feed one key-up / key-down interval.
    ///
    /// Returns the events produced: zero, one, or two of them. A character
    /// flush produces `[Char(ch)]` or `[Unknown]`; a word boundary produces
    /// `[…, WordBreak]`.
    pub fn push(&mut self, mark: bool, duration: u32) -> DecodedBatch {
        let mut out = DecodedBatch::new();
        if mark {
            let element = self.timing.classify_mark(duration);
            self.pattern.push(element.glyph());
            if self.adapt {
                self.timing.observe_mark(duration, element);
            }
        } else {
            match self.timing.classify_gap(duration) {
                Gap::IntraChar => {}
                Gap::Char => self.flush(&mut out),
                Gap::Word => {
                    self.flush(&mut out);
                    // Defer emission: a trailing word gap (silence after the
                    // final character) should not produce a dangling break.
                    // The next character flush will prepend the break.
                    if self.any_char_emitted {
                        self.pending_word_break = true;
                    }
                }
            }
        }
        out
    }

    /// Flush any in-progress character. Call after the input stream ends to
    /// avoid losing a trailing character that wasn't followed by a gap.
    ///
    /// Returns a possibly-empty batch because a flush can emit a deferred
    /// [`WordBreak`](Decoded::WordBreak) in front of the final character.
    pub fn finish(&mut self) -> DecodedBatch {
        let mut out = DecodedBatch::new();
        self.flush(&mut out);
        out
    }

    fn flush(&mut self, out: &mut DecodedBatch) {
        if self.pattern.is_empty() {
            return;
        }
        let decoded = match alphabet::char_for_pattern(self.pattern.as_str()) {
            Some(ch) => Decoded::Char(ch),
            None => Decoded::Unknown,
        };
        if self.pending_word_break {
            out.push(Decoded::WordBreak);
            self.pending_word_break = false;
        }
        out.push(decoded);
        self.any_char_emitted = true;
        self.pattern.clear();
    }
}

/// Inline-stored batch of [`Decoded`] events returned from
/// [`Decoder::push`]. Never holds more than two events.
#[derive(Debug, Clone, Copy, Default)]
pub struct DecodedBatch {
    items: [Option<Decoded>; 2],
    len: u8,
}

impl DecodedBatch {
    const fn new() -> Self {
        Self {
            items: [None, None],
            len: 0,
        }
    }

    fn push(&mut self, d: Decoded) {
        self.items[self.len as usize] = Some(d);
        self.len += 1;
    }

    /// Number of events in the batch (0, 1, or 2).
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len as usize
    }

    /// Whether the batch is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl IntoIterator for DecodedBatch {
    type Item = Decoded;
    type IntoIter = DecodedBatchIter;

    fn into_iter(self) -> Self::IntoIter {
        DecodedBatchIter { batch: self, idx: 0 }
    }
}

/// Iterator returned by [`DecodedBatch::into_iter`].
pub struct DecodedBatchIter {
    batch: DecodedBatch,
    idx: u8,
}

impl Iterator for DecodedBatchIter {
    type Item = Decoded;

    fn next(&mut self) -> Option<Decoded> {
        if self.idx >= self.batch.len {
            return None;
        }
        let item = self.batch.items[self.idx as usize].take();
        self.idx += 1;
        item
    }
}

mod heapless_pattern {
    use super::MAX_PATTERN;

    /// Fixed-capacity ASCII buffer for an in-progress pattern. Avoids a
    /// heap allocation on the hot path without pulling in a `heapless` dep.
    #[derive(Debug, Clone)]
    pub struct Pattern {
        buf: [u8; MAX_PATTERN],
        len: u8,
    }

    impl Pattern {
        pub const fn new() -> Self {
            Self {
                buf: [0; MAX_PATTERN],
                len: 0,
            }
        }

        pub fn push(&mut self, glyph: char) {
            debug_assert!(glyph == '.' || glyph == '-');
            if (self.len as usize) < MAX_PATTERN {
                self.buf[self.len as usize] = glyph as u8;
                self.len += 1;
            }
            // If saturated, further glyphs are dropped — flush() will emit
            // `Unknown` for the overlong pattern.
        }

        pub fn clear(&mut self) {
            self.len = 0;
        }

        pub const fn is_empty(&self) -> bool {
            self.len == 0
        }

        pub fn as_str(&self) -> &str {
            // Safe: only ever populated with `.` or `-` (ASCII) via `push`.
            std::str::from_utf8(&self.buf[..self.len as usize])
                .expect("pattern buffer holds only ASCII '.' and '-'")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive the decoder with a sequence of (mark, duration) pairs and
    /// collect every `Decoded` event produced.
    fn run(dec: &mut Decoder, events: &[(bool, u32)]) -> Vec<Decoded> {
        let mut out = Vec::new();
        for &(m, d) in events {
            out.extend(dec.push(m, d));
        }
        out.extend(dec.finish());
        out
    }

    /// Expand `"SOS"` into perfect-timing (mark, duration) pairs at T=1.
    /// Between elements: 1-unit gap. Between characters: 3-unit gap.
    /// Between words: 7-unit gap. Input is uppercase A–Z, 0–9, and space.
    fn synth(text: &str) -> Vec<(bool, u32)> {
        let mut out = Vec::new();
        let mut first_word = true;
        for word in text.split(' ') {
            if !first_word {
                out.push((false, 7));
            }
            first_word = false;
            let mut first_char = true;
            for ch in word.chars() {
                if !first_char {
                    out.push((false, 3));
                }
                first_char = false;
                let pattern = crate::alphabet::pattern_for_char(ch)
                    .expect("test text uses only known chars");
                let mut first_elem = true;
                for glyph in pattern.chars() {
                    if !first_elem {
                        out.push((false, 1));
                    }
                    first_elem = false;
                    let dur = match glyph {
                        '.' => 1,
                        '-' => 3,
                        _ => unreachable!(),
                    };
                    out.push((true, dur));
                }
            }
        }
        out
    }

    #[test]
    fn decodes_single_character() {
        let mut dec = Decoder::new(TimingEstimator::from_unit(1)).with_adapt(false);
        let events = synth("S");
        let out = run(&mut dec, &events);
        assert_eq!(out, vec![Decoded::Char('S')]);
    }

    #[test]
    fn decodes_single_word() {
        let mut dec = Decoder::new(TimingEstimator::from_unit(1)).with_adapt(false);
        let events = synth("SOS");
        let out = run(&mut dec, &events);
        assert_eq!(
            out,
            vec![Decoded::Char('S'), Decoded::Char('O'), Decoded::Char('S')],
        );
    }

    #[test]
    fn decodes_word_break() {
        let mut dec = Decoder::new(TimingEstimator::from_unit(1)).with_adapt(false);
        let events = synth("CQ DE");
        let out = run(&mut dec, &events);
        assert_eq!(
            out,
            vec![
                Decoded::Char('C'),
                Decoded::Char('Q'),
                Decoded::WordBreak,
                Decoded::Char('D'),
                Decoded::Char('E'),
            ],
        );
    }

    #[test]
    fn unknown_pattern_emits_unknown() {
        let mut dec = Decoder::new(TimingEstimator::from_unit(1)).with_adapt(false);
        // 8 dits — no such character.
        let mut events = Vec::new();
        for i in 0..8 {
            if i > 0 {
                events.push((false, 1));
            }
            events.push((true, 1));
        }
        let out = run(&mut dec, &events);
        assert_eq!(out, vec![Decoded::Unknown]);
    }

    #[test]
    fn finish_flushes_trailing_pattern() {
        let mut dec = Decoder::new(TimingEstimator::from_unit(1)).with_adapt(false);
        // Send the pattern for 'R' (.-.) with no trailing gap.
        dec.push(true, 1);
        dec.push(false, 1);
        dec.push(true, 3);
        dec.push(false, 1);
        dec.push(true, 1);
        let tail: Vec<Decoded> = dec.finish().into_iter().collect();
        assert_eq!(tail, vec![Decoded::Char('R')]);
    }
}
