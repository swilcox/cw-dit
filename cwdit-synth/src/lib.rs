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
    /// Add white Gaussian noise at this SNR in dB, measured full-band
    /// against one keyed track's tone power. `None` adds no noise. The
    /// noise spans the whole file (lead and tail silence included) and is
    /// mixed before normalisation, so the SNR survives output scaling and
    /// the result never clips.
    pub noise_snr_db: Option<f32>,
    /// Seed for the noise generator; renders are fully deterministic for a
    /// given seed, which is what test fixtures want.
    pub noise_seed: u64,
}

impl Default for SynthOptions {
    fn default() -> Self {
        Self {
            sample_rate: 8_000,
            lead_silence_s: 0.2,
            tail_silence_s: 0.2,
            ramp_ms: 10.0,
            amplitude: 0.8,
            noise_snr_db: None,
            noise_seed: 0x5EED_CAFE_F00D_D00D,
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

    if let Some(snr_db) = options.noise_snr_db {
        // Each track is keyed at unit amplitude before normalisation, so a
        // tone's power here is 1/2; solve for the noise variance that puts
        // it `snr_db` below that.
        let noise_std = (0.5 / 10.0_f32.powf(snr_db / 10.0)).sqrt();
        let mut rng = NoiseGen::new(options.noise_seed);
        for s in &mut mix {
            *s += noise_std * rng.next_gaussian();
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

/// Deterministic Gaussian noise source: xorshift64* uniforms fed through
/// Box–Muller. Self-contained so fixtures don't pull in a rand dependency.
struct NoiseGen {
    state: u64,
    spare: Option<f32>,
}

impl NoiseGen {
    fn new(seed: u64) -> Self {
        Self {
            // xorshift never leaves the all-zero state.
            state: if seed == 0 { 0xBAD_5EED } else { seed },
            spare: None,
        }
    }

    /// Uniform in (0, 1] — the half-open end matters so `ln` below never
    /// sees zero.
    fn next_uniform(&mut self) -> f32 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        let r = self.state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((r >> 40) as f32 + 1.0) / 16_777_216.0
    }

    /// Standard normal sample via Box–Muller (pairs cached in `spare`).
    fn next_gaussian(&mut self) -> f32 {
        if let Some(s) = self.spare.take() {
            return s;
        }
        let radius = (-2.0 * self.next_uniform().ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * self.next_uniform();
        self.spare = Some(radius * theta.sin());
        radius * theta.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(noise_snr_db: Option<f32>, noise_seed: u64) -> SynthOptions {
        SynthOptions {
            noise_snr_db,
            noise_seed,
            ..SynthOptions::default()
        }
    }

    fn samples(bytes: &[u8]) -> Vec<i16> {
        let mut reader = hound::WavReader::new(Cursor::new(bytes.to_vec())).expect("wav");
        reader.samples::<i16>().map(|s| s.expect("sample")).collect()
    }

    #[test]
    fn noise_render_is_deterministic() {
        let tracks = [Track::new("CQ", 20.0, 700.0)];
        let a = synth_bytes(&tracks, &opts(Some(10.0), 7)).expect("synth");
        let b = synth_bytes(&tracks, &opts(Some(10.0), 7)).expect("synth");
        assert_eq!(a, b, "same seed must render identical bytes");
    }

    #[test]
    fn different_seeds_render_different_noise() {
        let tracks = [Track::new("CQ", 20.0, 700.0)];
        let a = synth_bytes(&tracks, &opts(Some(10.0), 1)).expect("synth");
        let b = synth_bytes(&tracks, &opts(Some(10.0), 2)).expect("synth");
        assert_ne!(a, b);
    }

    #[test]
    fn noise_fills_the_lead_silence() {
        let tracks = [Track::new("E", 20.0, 700.0)];
        let clean = synth_bytes(&tracks, &opts(None, 1)).expect("synth");
        let noisy = synth_bytes(&tracks, &opts(Some(10.0), 1)).expect("synth");
        // First 0.1 s sits inside the default 0.2 s lead: silent when
        // clean, non-silent when noise is requested.
        let head = 800;
        assert!(samples(&clean)[..head].iter().all(|&s| s == 0));
        let sum_sq: f64 = samples(&noisy)[..head]
            .iter()
            .map(|&s| f64::from(s) * f64::from(s))
            .sum();
        let rms = (sum_sq / head as f64).sqrt();
        assert!(rms > 100.0, "lead should carry audible noise, rms={rms}");
    }

    #[test]
    fn noise_level_matches_requested_snr() {
        // 10 dB SNR: tone power P, noise power P/10. Measure noise power in
        // the lead silence and tone+noise power mid-file, both relative to
        // the same normalisation.
        let tracks = [Track::new("OOO", 20.0, 700.0)];
        let bytes = synth_bytes(&tracks, &opts(Some(10.0), 42)).expect("synth");
        let all = samples(&bytes);
        let power = |s: &[i16]| {
            s.iter()
                .map(|&x| f64::from(x) * f64::from(x))
                .sum::<f64>()
                / s.len() as f64
        };
        let noise_p = power(&all[..1_200]); // lead: noise only
        // "O" = dah dah dah; the first dah spans 0.2 s..0.38 s.
        let mark_p = power(&all[1_800..2_800]) - noise_p; // first dah, noise removed
        let snr_db = 10.0 * (mark_p / noise_p).log10();
        assert!(
            (snr_db - 10.0).abs() < 1.5,
            "expected ~10 dB, measured {snr_db:.1} dB"
        );
    }
}
