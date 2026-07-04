//! Post-detection envelope smoothing.
//!
//! A short moving average over the envelope stream, sized by the caller to
//! roughly a quarter of a dit. Averaging the detector output narrows the
//! effective noise bandwidth toward the keying bandwidth — worth several dB
//! of usable SNR — and tightens the noise distribution enough for the
//! slicer's SNR gate (see [`Threshold`](crate::threshold::Threshold)) to
//! separate noise from signal reliably.
//!
//! The window delays the envelope by `(len - 1) / 2` samples, but the delay
//! is the same on rising and falling edges, so mark/space durations — the
//! only thing the decoder consumes — are unaffected.

/// Streaming boxcar (moving-average) filter.
#[derive(Debug, Clone)]
pub struct MovingAverage {
    buf: Vec<f32>,
    idx: usize,
    filled: usize,
}

impl MovingAverage {
    /// Create a filter averaging over the last `len` samples. `len == 1`
    /// is a pass-through.
    ///
    /// # Panics
    /// Panics if `len` is zero.
    #[must_use]
    pub fn new(len: usize) -> Self {
        assert!(len >= 1, "window length must be at least 1");
        Self {
            buf: vec![0.0; len],
            idx: 0,
            filled: 0,
        }
    }

    /// Feed one sample; returns the mean of the last `len` samples (or of
    /// everything seen so far while the window is still filling).
    pub fn push(&mut self, sample: f32) -> f32 {
        self.buf[self.idx] = sample;
        self.idx = (self.idx + 1) % self.buf.len();
        self.filled = (self.filled + 1).min(self.buf.len());
        // The window is a handful of samples; summing it every push is
        // cheaper than defending a running sum against f32 drift.
        let sum: f32 = self.buf[..self.filled].iter().sum();
        sum / self.filled as f32
    }

    /// Window length in samples.
    #[must_use]
    pub fn window_len(&self) -> usize {
        self.buf.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn len_one_is_passthrough() {
        let mut m = MovingAverage::new(1);
        assert!((m.push(0.3) - 0.3).abs() < 1e-6);
        assert!((m.push(0.9) - 0.9).abs() < 1e-6);
    }

    #[test]
    fn averages_over_window() {
        let mut m = MovingAverage::new(4);
        m.push(1.0);
        m.push(2.0);
        m.push(3.0);
        assert!((m.push(4.0) - 2.5).abs() < 1e-6);
        // Window slides: [2, 3, 4, 5] → 3.5.
        assert!((m.push(5.0) - 3.5).abs() < 1e-6);
    }

    #[test]
    fn partial_window_averages_what_it_has() {
        let mut m = MovingAverage::new(8);
        assert!((m.push(1.0) - 1.0).abs() < 1e-6);
        assert!((m.push(3.0) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn suppresses_single_sample_glitch() {
        let mut m = MovingAverage::new(4);
        for _ in 0..8 {
            m.push(0.0);
        }
        // A lone spike of 1.0 must not exceed 1/len after averaging.
        let out = m.push(1.0);
        assert!(out <= 0.25 + 1e-6, "glitch leaked through: {out}");
    }

    #[test]
    #[should_panic(expected = "window length must be at least 1")]
    fn rejects_zero_length() {
        let _ = MovingAverage::new(0);
    }
}
