//! Sample sources for the cw-dit skimmer.
//!
//! The [`Source`] trait abstracts over files, sound cards, and SDR devices.
//! Concrete implementations live in submodules; pull them in under a feature
//! gate where appropriate.
//!
//! Currently provides [`wav::WavSource`] for mono PCM WAV files and
//! [`audio::AudioSource`] for live input from a system audio device.

use std::fmt;

pub mod audio;
pub mod wav;

pub use audio::AudioSource;
pub use wav::WavSource;

/// A stream of samples at a fixed sample rate.
///
/// Implementations pull samples into a caller-supplied buffer, returning the
/// number of samples actually written. A return value of `Ok(0)` signals
/// end-of-stream for finite sources (files).
pub trait Source {
    /// The element type of the stream — `f32` for real audio, `Complex<f32>`
    /// for IQ, etc.
    type Sample: Copy;

    /// Sample rate of the stream, in Hertz.
    fn sample_rate(&self) -> f32;

    /// Read up to `buf.len()` samples into `buf`. Returns the number of
    /// samples written; `Ok(0)` indicates end-of-stream.
    fn read(&mut self, buf: &mut [Self::Sample]) -> Result<usize, SourceError>;
}

/// Errors returned by [`Source`] implementations.
#[derive(Debug)]
pub enum SourceError {
    /// Underlying IO error.
    Io(std::io::Error),
    /// Source is not in a format this implementation supports (e.g. stereo
    /// WAV when only mono is supported).
    UnsupportedFormat(String),
    /// Source-specific decoding failure (e.g. corrupt WAV header).
    Decode(String),
}

impl fmt::Display for SourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::UnsupportedFormat(msg) => write!(f, "unsupported format: {msg}"),
            Self::Decode(msg) => write!(f, "decode error: {msg}"),
        }
    }
}

impl std::error::Error for SourceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for SourceError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
