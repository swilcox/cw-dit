//! Pure-DSP building blocks for the cw-dit skimmer.
//!
//! The pipeline at this stage is:
//!
//! ```text
//!   audio samples ──► Goertzel ──► Threshold ──► RunLengthEncoder ──► Run events
//! ```
//!
//! Each stage is an independent, IO-free struct that consumes one sample at
//! a time and optionally produces one output. Callers drive them from any
//! [`cwdit_source::Source`](../cwdit_source/trait.Source.html) — files for
//! test fixtures, a sound card or SDR later.

pub mod bank;
pub mod envelope;
pub mod runlength;
pub mod threshold;

pub use bank::GoertzelBank;
pub use envelope::Goertzel;
pub use runlength::{Run, RunLengthEncoder};
pub use threshold::Threshold;
