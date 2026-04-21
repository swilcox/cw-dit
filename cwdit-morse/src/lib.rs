//! Morse-code timing, alphabet, and streaming decoder.
//!
//! This crate operates on a run-length stream of key-up / key-down intervals,
//! classifies each into a Morse [`Element`] or [`Gap`], and emits decoded
//! characters. It has no IO and no dependency on any specific signal source —
//! callers produce the run-length stream from audio, IQ envelopes, or test
//! fixtures as appropriate.
//!
//! # Quick start
//!
//! ```
//! use cwdit_morse::{Decoded, Decoder, TimingEstimator};
//!
//! // Seed the decoder with a dot-unit of 1 "tick".
//! let mut dec = Decoder::new(TimingEstimator::from_unit(1)).with_adapt(false);
//!
//! // Send "E" — a single dit — followed by a character gap.
//! let mut out = Vec::new();
//! out.extend(dec.push(true, 1));
//! out.extend(dec.push(false, 3));
//!
//! assert_eq!(out, vec![Decoded::Char('E')]);
//! ```

pub mod alphabet;
pub mod decoder;
pub mod element;
pub mod timing;

pub use decoder::{Decoder, DecodedBatch};
pub use element::{Decoded, Element, Gap};
pub use timing::TimingEstimator;
