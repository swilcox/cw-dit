//! Pure-DSP building blocks for the cw-dit skimmer.
//!
//! The pipeline at this stage is:
//!
//! ```text
//!   audio samples ──► Goertzel ──► MovingAverage ──► Threshold
//!                        ──► RunLengthEncoder ──► Debouncer ──► Run events
//! ```
//!
//! `MovingAverage` narrows the post-detection noise bandwidth toward the
//! keying bandwidth, `Threshold` slices with noise-floor tracking plus an
//! SNR gate, and `Debouncer` absorbs any glitch runs that still get
//! through — together they keep noise from reaching the Morse decoder.
//!
//! Each stage is an independent, IO-free struct that consumes one sample at
//! a time and optionally produces one output. Callers drive them from any
//! [`cwdit_source::Source`](../cwdit_source/trait.Source.html) — files for
//! test fixtures, a sound card or SDR later.

pub mod bank;
pub mod channelizer;
pub mod debounce;
pub mod envelope;
pub mod iq_channelizer;
pub mod runlength;
pub mod scan;
pub mod skim;
pub mod smooth;
pub mod threshold;
pub mod track;

pub use bank::GoertzelBank;
pub use channelizer::{Channelizer, FftChannelizer};
pub use debounce::Debouncer;
pub use envelope::{Goertzel, IqTone, ToneFilter};
pub use iq_channelizer::IqChannelizer;
pub use runlength::{Run, RunLengthEncoder};
pub use scan::{BinStats, ScanConfig, envelope_correlation, suppress_correlated_ghosts};
pub use skim::{Detector, DetectorConfig, IqDetector};
pub use smooth::MovingAverage;
pub use threshold::Threshold;
pub use track::{ChannelTracker, TrackerConfig, TrackerUpdate};
