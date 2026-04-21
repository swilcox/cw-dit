//! Narrow-band envelope detector.
//!
//! A single-bin Goertzel filter estimates the magnitude of the target tone
//! across non-overlapping blocks of `block_len` input samples. Each completed
//! block produces one envelope sample — so the envelope stream's sample rate
//! is `input_sample_rate / block_len`.
//!
//! Goertzel is a perfect fit here: it's the cheapest way to compute one DFT
//! bin, and a narrow CW tone is exactly one bin.

/// Single-bin Goertzel filter.
#[derive(Debug, Clone)]
pub struct Goertzel {
    coeff: f32,
    s1: f32,
    s2: f32,
    n: u32,
    block_len: u32,
}

impl Goertzel {
    /// Create a detector tuned to `target_freq_hz` at `sample_rate_hz`,
    /// integrating over `block_len` input samples per envelope output.
    ///
    /// # Panics
    /// Panics if any argument is non-positive or if `block_len` is so short
    /// that less than one full cycle of the target frequency fits inside it.
    #[must_use]
    pub fn new(target_freq_hz: f32, sample_rate_hz: f32, block_len: u32) -> Self {
        assert!(target_freq_hz > 0.0);
        assert!(sample_rate_hz > 0.0);
        assert!(block_len >= 1);

        // k = (block_len * target_freq / sample_rate); fractional is fine.
        let k = block_len as f32 * target_freq_hz / sample_rate_hz;
        let cycles = block_len as f32 * target_freq_hz / sample_rate_hz;
        assert!(
            cycles >= 1.0,
            "block_len ({block_len}) must span at least one cycle of \
             target ({target_freq_hz} Hz) at sample rate {sample_rate_hz} Hz"
        );
        let w = 2.0 * core::f32::consts::PI * k / block_len as f32;
        let coeff = 2.0 * w.cos();

        Self {
            coeff,
            s1: 0.0,
            s2: 0.0,
            n: 0,
            block_len,
        }
    }

    /// Feed one input sample. Returns `Some(magnitude)` at the end of each
    /// block, otherwise `None`.
    ///
    /// The returned value is the magnitude (not power) of the target bin,
    /// normalised by `block_len` so envelopes from different block sizes are
    /// directly comparable.
    pub fn push(&mut self, sample: f32) -> Option<f32> {
        let s = sample + self.coeff * self.s1 - self.s2;
        self.s2 = self.s1;
        self.s1 = s;
        self.n += 1;
        if self.n == self.block_len {
            let power = self.s1 * self.s1 + self.s2 * self.s2 - self.coeff * self.s1 * self.s2;
            let magnitude = power.max(0.0).sqrt() / self.block_len as f32;
            self.reset();
            Some(magnitude)
        } else {
            None
        }
    }

    /// Block size, in input samples, between envelope outputs.
    #[must_use]
    pub const fn block_len(&self) -> u32 {
        self.block_len
    }

    fn reset(&mut self) {
        self.s1 = 0.0;
        self.s2 = 0.0;
        self.n = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 8_000.0;

    fn sine(freq: f32, sample_rate: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (2.0 * core::f32::consts::PI * freq * t).sin()
            })
            .collect()
    }

    #[test]
    fn on_tone_produces_high_magnitude() {
        let mut g = Goertzel::new(700.0, SR, 128);
        let samples = sine(700.0, SR, 128);
        let out: Vec<f32> = samples.iter().filter_map(|&s| g.push(s)).collect();
        assert_eq!(out.len(), 1);
        // A unit-amplitude sine should give ~0.5 magnitude in the single bin
        // (half the energy is at +f, half at -f).
        assert!(
            out[0] > 0.4 && out[0] < 0.6,
            "expected ~0.5, got {}",
            out[0]
        );
    }

    #[test]
    fn off_tone_produces_low_magnitude() {
        let mut g = Goertzel::new(700.0, SR, 128);
        // A tone at 1600 Hz is far outside the 700 Hz bin.
        let samples = sine(1_600.0, SR, 128);
        let out: Vec<f32> = samples.iter().filter_map(|&s| g.push(s)).collect();
        assert_eq!(out.len(), 1);
        assert!(out[0] < 0.05, "expected near-zero, got {}", out[0]);
    }

    #[test]
    fn silence_produces_zero() {
        let mut g = Goertzel::new(700.0, SR, 128);
        let out: Vec<f32> = (0..128).filter_map(|_| g.push(0.0)).collect();
        assert_eq!(out, vec![0.0]);
    }

    #[test]
    fn emits_one_value_per_block() {
        let mut g = Goertzel::new(700.0, SR, 64);
        let samples = sine(700.0, SR, 64 * 4);
        let out: Vec<f32> = samples.iter().filter_map(|&s| g.push(s)).collect();
        assert_eq!(out.len(), 4);
    }

    #[test]
    #[should_panic(expected = "must span at least one cycle")]
    fn rejects_block_too_short_for_target() {
        // 700 Hz at 8 kHz → one cycle is ~11.4 samples; 8 samples is too few.
        let _ = Goertzel::new(700.0, SR, 8);
    }
}
