//! WAV-file source.
//!
//! [`WavSource`] reads a mono PCM WAV file and yields real-valued `f32`
//! samples normalised to `[-1.0, 1.0]`. Integer formats (`i16`, `i32`) are
//! scaled to the corresponding floating-point range; `f32` WAV is passed
//! through.
//!
//! The entire file is read into memory at construction. This keeps the
//! implementation simple and is adequate for the recordings the skimmer is
//! expected to process; stream-in-chunks will come later if needed.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use hound::{SampleFormat, WavReader};

use crate::{Source, SourceError};

/// Reads a mono PCM WAV file in full and plays it back as an `f32` sample
/// stream.
#[derive(Debug)]
pub struct WavSource {
    samples: Vec<f32>,
    cursor: usize,
    sample_rate: f32,
}

impl WavSource {
    /// Open a WAV file on disk.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, SourceError> {
        let file = File::open(path.as_ref())?;
        Self::from_reader(BufReader::new(file))
    }

    /// Read a WAV stream from any [`Read`] source. Useful for tests that
    /// supply an in-memory `Cursor`.
    pub fn from_reader<R: Read>(reader: R) -> Result<Self, SourceError> {
        let reader = WavReader::new(reader)
            .map_err(|e| SourceError::Decode(format!("wav header: {e}")))?;
        let spec = reader.spec();
        if spec.channels != 1 {
            return Err(SourceError::UnsupportedFormat(format!(
                "expected mono WAV, got {} channels",
                spec.channels
            )));
        }

        let samples = decode_samples(reader, spec)?;

        Ok(Self {
            samples,
            cursor: 0,
            sample_rate: spec.sample_rate as f32,
        })
    }

    /// Total number of samples in the file.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the file had zero samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

impl Source for WavSource {
    type Sample = f32;

    fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    fn read(&mut self, buf: &mut [f32]) -> Result<usize, SourceError> {
        let remaining = self.samples.len() - self.cursor;
        let n = remaining.min(buf.len());
        buf[..n].copy_from_slice(&self.samples[self.cursor..self.cursor + n]);
        self.cursor += n;
        Ok(n)
    }
}

/// Decode the WAV payload to normalised `f32`. Splits on sample format /
/// bit-depth because hound's `samples::<T>()` is typed.
fn decode_samples<R: Read>(
    reader: WavReader<R>,
    spec: hound::WavSpec,
) -> Result<Vec<f32>, SourceError> {
    match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 16) => collect_scaled::<R, i16>(reader, f32::from(i16::MAX)),
        (SampleFormat::Int, 24 | 32) => collect_scaled::<R, i32>(reader, bits_scale(spec.bits_per_sample)),
        (SampleFormat::Float, 32) => {
            let mut out = Vec::with_capacity(reader.len() as usize);
            for s in reader.into_samples::<f32>() {
                out.push(s.map_err(|e| SourceError::Decode(format!("wav sample: {e}")))?);
            }
            Ok(out)
        }
        (fmt, bits) => Err(SourceError::UnsupportedFormat(format!(
            "sample format {fmt:?} at {bits} bits"
        ))),
    }
}

fn bits_scale(bits: u16) -> f32 {
    // For N-bit signed PCM stored in an i32, full-scale is 2^(N-1) - 1.
    let full_scale = (1u64 << (bits - 1)) - 1;
    full_scale as f32
}

fn collect_scaled<R, S>(reader: WavReader<R>, scale: f32) -> Result<Vec<f32>, SourceError>
where
    R: Read,
    S: hound::Sample + Into<i32>,
{
    let mut out = Vec::with_capacity(reader.len() as usize);
    for s in reader.into_samples::<S>() {
        let v = s.map_err(|e| SourceError::Decode(format!("wav sample: {e}")))?;
        let i: i32 = v.into();
        out.push(i as f32 / scale);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use hound::{WavSpec, WavWriter};

    use super::*;

    fn write_sine(freq_hz: f32, sample_rate: u32, duration_s: f32) -> Vec<u8> {
        let spec = WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut buf = Cursor::new(Vec::new());
        {
            let mut w = WavWriter::new(&mut buf, spec).unwrap();
            let total = (sample_rate as f32 * duration_s) as u32;
            for n in 0..total {
                let t = n as f32 / sample_rate as f32;
                let sample = (2.0 * std::f32::consts::PI * freq_hz * t).sin();
                let i = (sample * f32::from(i16::MAX)) as i16;
                w.write_sample(i).unwrap();
            }
            w.finalize().unwrap();
        }
        buf.into_inner()
    }

    #[test]
    fn reads_mono_16bit_wav() {
        let bytes = write_sine(700.0, 8_000, 0.25);
        let mut src = WavSource::from_reader(Cursor::new(bytes)).unwrap();

        assert!((src.sample_rate() - 8_000.0).abs() < 0.5);
        assert_eq!(src.len(), 2_000);

        let mut buf = vec![0.0_f32; 2_000];
        let n = src.read(&mut buf).unwrap();
        assert_eq!(n, 2_000);
        // Sine at full scale should span most of [-1, 1].
        let (lo, hi) = buf
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &x| {
                (lo.min(x), hi.max(x))
            });
        assert!(hi > 0.9, "peak {hi}");
        assert!(lo < -0.9, "trough {lo}");
    }

    #[test]
    fn read_signals_eof_with_zero() {
        let bytes = write_sine(700.0, 8_000, 0.01);
        let mut src = WavSource::from_reader(Cursor::new(bytes)).unwrap();
        let mut buf = [0.0_f32; 1_000];
        // Drain
        while src.read(&mut buf).unwrap() > 0 {}
        // Now at EOF
        assert_eq!(src.read(&mut buf).unwrap(), 0);
    }

    #[test]
    fn rejects_stereo_wav() {
        let spec = WavSpec {
            channels: 2,
            sample_rate: 8_000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut buf = Cursor::new(Vec::new());
        {
            let mut w = WavWriter::new(&mut buf, spec).unwrap();
            for _ in 0..100 {
                w.write_sample(0_i16).unwrap();
                w.write_sample(0_i16).unwrap();
            }
            w.finalize().unwrap();
        }
        let err = WavSource::from_reader(Cursor::new(buf.into_inner())).unwrap_err();
        assert!(matches!(err, SourceError::UnsupportedFormat(_)));
    }
}
