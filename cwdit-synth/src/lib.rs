//! CW audio synthesiser.
//!
//! Renders one or more Morse-keyed tones to a mono 16-bit PCM WAV. Used
//! by `cwdit-synth` (binary) to generate fixtures for the decoder and
//! by integration tests across the workspace that want a WAV without
//! duplicating synthesis code.

use std::fmt;
use std::io::Cursor;
use std::path::Path;

use cwdit_morse::alphabet;
use hound::{SampleFormat, WavSpec, WavWriter};

/// A single keyed tone: a message, its keying speed, and the tone frequency
/// it occupies. Multi-track synthesis mixes tracks into one channel.
#[derive(Clone, Debug)]
pub struct Track {
    pub text: String,
    pub wpm: f32,
    pub tone_hz: f32,
}

impl Track {
    pub fn new(text: impl Into<String>, wpm: f32, tone_hz: f32) -> Self {
        Self {
            text: text.into(),
            wpm,
            tone_hz,
        }
    }
}

/// Per-rendering options independent of the per-track parameters.
#[derive(Clone, Debug)]
pub struct SynthOptions {
    pub sample_rate: u32,
    pub lead_silence_s: f32,
    pub tail_silence_s: f32,
    pub ramp_ms: f32,
    /// Peak output amplitude in [0.0, 1.0] relative to i16 full-scale.
    /// Tracks are normalised so that the post-mix peak lands here.
    pub amplitude: f32,
}

impl Default for SynthOptions {
    fn default() -> Self {
        Self {
            sample_rate: 8_000,
            lead_silence_s: 0.2,
            tail_silence_s: 0.2,
            ramp_ms: 10.0,
            amplitude: 0.8,
        }
    }
}

/// Errors produced while synthesising a WAV.
#[derive(Debug)]
pub enum SynthError {
    /// The message contains a character with no entry in the Morse
    /// alphabet.
    UnknownChar(char),
    /// No tracks were supplied.
    EmptyTracks,
    /// Writing the WAV envelope to the underlying buffer failed.
    Hound(hound::Error),
    /// IO error writing a WAV file to disk.
    Io(std::io::Error),
}

impl fmt::Display for SynthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SynthError::UnknownChar(c) => {
                write!(f, "no Morse pattern for character {c:?}")
            }
            SynthError::EmptyTracks => f.write_str("at least one track is required"),
            SynthError::Hound(e) => write!(f, "wav error: {e}"),
            SynthError::Io(e) => write!(f, "io error: {e}"),
        }
    }
}

impl std::error::Error for SynthError {}

impl From<hound::Error> for SynthError {
    fn from(e: hound::Error) -> Self {
        SynthError::Hound(e)
    }
}

impl From<std::io::Error> for SynthError {
    fn from(e: std::io::Error) -> Self {
        SynthError::Io(e)
    }
}

/// Build a keying stream (true = tone on, false = silence) for `track` at
/// the given sample rate. Returns [`SynthError::UnknownChar`] if the text
/// contains a character not in the Morse alphabet.
pub fn keying_samples(track: &Track, sample_rate: u32) -> Result<Vec<bool>, SynthError> {
    let dot = (1.2 * sample_rate as f32 / track.wpm).round() as usize;
    let mut keying = Vec::new();
    let mut first_word = true;
    for word in track.text.split(' ') {
        if !first_word {
            keying.extend(std::iter::repeat_n(false, 7 * dot));
        }
        first_word = false;
        let mut first_char = true;
        for ch in word.chars() {
            if !first_char {
                keying.extend(std::iter::repeat_n(false, 3 * dot));
            }
            first_char = false;
            let pattern =
                alphabet::pattern_for_char(ch).ok_or(SynthError::UnknownChar(ch))?;
            let mut first_elem = true;
            for g in pattern.chars() {
                if !first_elem {
                    keying.extend(std::iter::repeat_n(false, dot));
                }
                first_elem = false;
                let n = if g == '.' { dot } else { 3 * dot };
                keying.extend(std::iter::repeat_n(true, n));
            }
        }
    }
    Ok(keying)
}

/// Render `tracks` to a mono 16-bit PCM WAV, returning the full file as a
/// byte buffer. All tracks start at the same instant (after `lead_silence_s`)
/// and are summed into a single channel, so they may overlap freely in time.
pub fn synth_bytes(tracks: &[Track], options: &SynthOptions) -> Result<Vec<u8>, SynthError> {
    if tracks.is_empty() {
        return Err(SynthError::EmptyTracks);
    }
    let sample_rate = options.sample_rate;
    let keyings: Vec<Vec<bool>> = tracks
        .iter()
        .map(|t| keying_samples(t, sample_rate))
        .collect::<Result<_, _>>()?;
    let max_len = keyings.iter().map(Vec::len).max().unwrap_or(0);

    let lead = (options.lead_silence_s * sample_rate as f32) as usize;
    let tail = (options.tail_silence_s * sample_rate as f32) as usize;
    let total = lead + max_len + tail;
    let mut mix = vec![0.0_f32; total];

    let ramp = ((options.ramp_ms * sample_rate as f32 / 1000.0) as usize).max(1);
    let step = 1.0_f32 / ramp as f32;

    for (track, keying) in tracks.iter().zip(&keyings) {
        let ang = 2.0 * std::f32::consts::PI * track.tone_hz / sample_rate as f32;
        let mut env = 0.0_f32;
        for (i, &on) in keying.iter().enumerate() {
            let target = if on { 1.0 } else { 0.0 };
            if (target - env).abs() < step {
                env = target;
            } else if env < target {
                env += step;
            } else {
                env -= step;
            }
            let pos = lead + i;
            let t = pos as f32;
            mix[pos] += env * (ang * t).sin();
        }
    }

    // Normalise so the post-mix peak never exceeds `amplitude`. When the
    // mix sits below unit amplitude (e.g. a single track), we leave the
    // peak at exactly `amplitude` — matching the existing test fixtures.
    let peak = mix.iter().fold(0.0_f32, |m, &x| m.max(x.abs())).max(1.0);
    let scale = options.amplitude.clamp(0.0, 1.0) / peak;

    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut buf = Cursor::new(Vec::new());
    {
        let mut w = WavWriter::new(&mut buf, spec)?;
        for &s in &mix {
            let clipped = (s * scale).clamp(-1.0, 1.0);
            let i = (clipped * f32::from(i16::MAX)) as i16;
            w.write_sample(i)?;
        }
        w.finalize()?;
    }
    Ok(buf.into_inner())
}

/// Render `tracks` and write the resulting WAV to `path`.
pub fn synth_to_path(
    path: &Path,
    tracks: &[Track],
    options: &SynthOptions,
) -> Result<(), SynthError> {
    let bytes = synth_bytes(tracks, options)?;
    std::fs::write(path, bytes)?;
    Ok(())
}
