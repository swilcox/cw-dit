//! Uniform-grid FFT channelizer (overlap-save / weighted-overlap-add).
//!
//! Where [`GoertzelBank`](crate::bank::GoertzelBank) tunes a handful of
//! independent filters to specific tones, an [`FftChannelizer`] splits a
//! stream of real audio samples into `N/2 + 1` uniformly-spaced complex
//! sub-bands in one shot. It is the building block that turns cw-dit from
//! "decode a list of known tones" into "decode every CW signal inside a
//! passband at once".
//!
//! The implementation is a straightforward windowed-DFT filterbank:
//!
//! 1. push input samples into an `N`-long ring buffer;
//! 2. every `hop` samples, copy the most recent `N` samples out in
//!    chronological order, multiply by an analysis window, and FFT;
//! 3. emit the positive-frequency half of the spectrum (bins `0..=N/2`) as
//!    complex samples at a decimated rate of `sample_rate / hop`.
//!
//! Downstream code takes `|z|` on whichever bin(s) it cares about to get an
//! envelope stream, then feeds the existing
//! [`Threshold`](crate::threshold::Threshold) →
//! [`RunLengthEncoder`](crate::runlength::RunLengthEncoder) → morse decoder
//! chain.
//!
//! Normalisation is chosen so that a unit-amplitude sine tone at a bin
//! centre produces `|z| ≈ 0.5` on that bin, matching the convention used by
//! [`Goertzel`](crate::envelope::Goertzel).

use std::sync::Arc;

use rustfft::{Fft, FftPlanner, num_complex::Complex32};

/// Uniform-grid FFT channelizer over real-valued audio.
pub struct FftChannelizer {
    fft_size: usize,
    hop: usize,
    sample_rate: f32,
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    norm: f32,
    ring: Vec<f32>,
    write_idx: usize,
    filled: usize,
    samples_since_emit: usize,
    emitted_once: bool,
    fft_buf: Vec<Complex32>,
    scratch: Vec<Complex32>,
    bins: Vec<Complex32>,
}

impl FftChannelizer {
    /// Build a channelizer with FFT size `fft_size`, hop size `hop`, for
    /// real input at `sample_rate_hz`. Uses a Hann analysis window.
    ///
    /// # Panics
    /// Panics if `fft_size < 2`, if `hop` is zero or greater than
    /// `fft_size`, or if `sample_rate_hz` is non-positive.
    #[must_use]
    pub fn new(fft_size: usize, hop: usize, sample_rate_hz: f32) -> Self {
        assert!(fft_size >= 2, "fft_size must be at least 2");
        assert!(hop > 0, "hop must be positive");
        assert!(
            hop <= fft_size,
            "hop ({hop}) must be <= fft_size ({fft_size})"
        );
        assert!(sample_rate_hz > 0.0, "sample_rate_hz must be positive");

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(fft_size);
        let window = hann(fft_size);
        let sum_w: f32 = window.iter().sum();
        let norm = sum_w.recip();
        let scratch_len = fft.get_inplace_scratch_len();

        Self {
            fft_size,
            hop,
            sample_rate: sample_rate_hz,
            fft,
            window,
            norm,
            ring: vec![0.0; fft_size],
            write_idx: 0,
            filled: 0,
            samples_since_emit: 0,
            emitted_once: false,
            fft_buf: vec![Complex32::new(0.0, 0.0); fft_size],
            scratch: vec![Complex32::new(0.0, 0.0); scratch_len],
            bins: vec![Complex32::new(0.0, 0.0); fft_size / 2 + 1],
        }
    }

    /// Number of output bins (positive half-spectrum, inclusive of DC and
    /// Nyquist): `fft_size / 2 + 1`.
    #[must_use]
    pub const fn channel_count(&self) -> usize {
        self.fft_size / 2 + 1
    }

    /// FFT size `N` (input samples per frame).
    #[must_use]
    pub const fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Hop size (input samples between successive frames).
    #[must_use]
    pub const fn hop(&self) -> usize {
        self.hop
    }

    /// Bin spacing in Hz (`sample_rate / fft_size`).
    #[must_use]
    pub fn bin_spacing_hz(&self) -> f32 {
        self.sample_rate / self.fft_size as f32
    }

    /// Output sample rate, in frames per second (`sample_rate / hop`).
    #[must_use]
    pub fn output_sample_rate(&self) -> f32 {
        self.sample_rate / self.hop as f32
    }

    /// Centre frequency of bin `idx`, in Hz.
    ///
    /// # Panics
    /// Panics if `idx >= channel_count()`.
    #[must_use]
    pub fn bin_frequency(&self, idx: usize) -> f32 {
        assert!(idx < self.channel_count(), "bin index out of range");
        idx as f32 * self.bin_spacing_hz()
    }

    /// Nearest bin index for `freq_hz`. Clamped to the valid range.
    #[must_use]
    pub fn bin_index_for(&self, freq_hz: f32) -> usize {
        let raw = (freq_hz / self.bin_spacing_hz()).round();
        if raw < 0.0 {
            0
        } else {
            (raw as usize).min(self.channel_count() - 1)
        }
    }

    /// Feed one real input sample.
    ///
    /// Returns `Some(bins)` every `hop` samples once the ring buffer has
    /// filled (after the first `fft_size` samples); otherwise `None`. The
    /// returned slice has `channel_count()` complex values, bin `k`
    /// corresponding to frequency `k * sample_rate / fft_size`.
    pub fn push(&mut self, sample: f32) -> Option<&[Complex32]> {
        self.ring[self.write_idx] = sample;
        self.write_idx = (self.write_idx + 1) % self.fft_size;
        if self.filled < self.fft_size {
            self.filled += 1;
        }
        self.samples_since_emit += 1;

        let ready = if self.emitted_once {
            self.samples_since_emit >= self.hop
        } else {
            self.filled >= self.fft_size
        };
        if !ready {
            return None;
        }

        for i in 0..self.fft_size {
            let idx = (self.write_idx + i) % self.fft_size;
            self.fft_buf[i] = Complex32::new(self.ring[idx] * self.window[i], 0.0);
        }
        self.fft
            .process_with_scratch(&mut self.fft_buf, &mut self.scratch);

        let half = self.fft_size / 2 + 1;
        for (dst, src) in self.bins.iter_mut().zip(&self.fft_buf[..half]) {
            *dst = Complex32::new(src.re * self.norm, src.im * self.norm);
        }

        self.samples_since_emit = 0;
        self.emitted_once = true;
        Some(&self.bins)
    }
}

fn hann(n: usize) -> Vec<f32> {
    assert!(n >= 2);
    let denom = (n - 1) as f32;
    (0..n)
        .map(|i| {
            let phase = core::f32::consts::TAU * i as f32 / denom;
            0.5 * (1.0 - phase.cos())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 8_000.0;
    const N: usize = 1024;
    const HOP: usize = 256;

    fn sine(freq: f32, sample_rate: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (core::f32::consts::TAU * freq * t).sin()
            })
            .collect()
    }

    fn drain(chan: &mut FftChannelizer, samples: &[f32]) -> Vec<Vec<Complex32>> {
        let mut out = Vec::new();
        for &s in samples {
            if let Some(bins) = chan.push(s) {
                out.push(bins.to_vec());
            }
        }
        out
    }

    #[test]
    fn bin_geometry_matches_sample_rate_and_size() {
        let chan = FftChannelizer::new(N, HOP, SR);
        assert_eq!(chan.channel_count(), N / 2 + 1);
        assert!((chan.bin_spacing_hz() - SR / N as f32).abs() < 1e-6);
        assert!((chan.output_sample_rate() - SR / HOP as f32).abs() < 1e-6);
        assert!(chan.bin_frequency(0).abs() < 1e-6);
        assert!((chan.bin_frequency(1) - chan.bin_spacing_hz()).abs() < 1e-6);
    }

    #[test]
    fn bin_index_for_rounds_and_clamps() {
        let chan = FftChannelizer::new(N, HOP, SR);
        let spacing = chan.bin_spacing_hz();
        // Exact centre.
        assert_eq!(chan.bin_index_for(10.0 * spacing), 10);
        // Rounds to nearest.
        assert_eq!(chan.bin_index_for(10.4 * spacing), 10);
        assert_eq!(chan.bin_index_for(10.6 * spacing), 11);
        // Clamps negatives to 0 and above-Nyquist to N/2.
        assert_eq!(chan.bin_index_for(-100.0), 0);
        assert_eq!(chan.bin_index_for(SR), N / 2);
    }

    #[test]
    fn first_emission_after_fft_size_samples_then_every_hop() {
        let mut chan = FftChannelizer::new(N, HOP, SR);
        let samples = sine(1_000.0, SR, N + 4 * HOP);
        let mut emit_indices = Vec::new();
        for (i, &s) in samples.iter().enumerate() {
            if chan.push(s).is_some() {
                emit_indices.push(i);
            }
        }
        assert_eq!(emit_indices.first().copied(), Some(N - 1));
        // Four more emissions spaced exactly HOP apart.
        let gaps: Vec<usize> = emit_indices.windows(2).map(|w| w[1] - w[0]).collect();
        assert!(gaps.iter().all(|&g| g == HOP), "gaps = {gaps:?}");
        assert_eq!(emit_indices.len(), 5);
    }

    #[test]
    fn on_bin_tone_peaks_in_correct_bin() {
        // Pick a frequency that lands exactly on a bin centre (k=128 →
        // 1000 Hz at SR/N = 7.8125 Hz spacing).
        let spacing = SR / N as f32;
        let target_bin = 128;
        let freq = target_bin as f32 * spacing;
        assert!((freq - 1_000.0).abs() < 1e-3);

        let mut chan = FftChannelizer::new(N, HOP, SR);
        let samples = sine(freq, SR, N * 2);
        let frames = drain(&mut chan, &samples);
        assert!(!frames.is_empty());

        // Use the last frame, by which time the window is fully populated
        // with steady-state tone.
        let last = frames.last().unwrap();
        let mags: Vec<f32> = last.iter().map(|c| c.norm()).collect();
        let (peak_idx, peak_val) = mags
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, v)| (i, *v))
            .unwrap();

        assert_eq!(peak_idx, target_bin, "peak at bin {peak_idx}, want {target_bin}");
        assert!(peak_val > 0.45 && peak_val < 0.55, "peak mag {peak_val}");

        // Neighbouring bins should be dB-quiet: Hann first sidelobe is
        // ~ -31 dB, so an order of magnitude below peak is a safe margin.
        assert!(mags[target_bin - 4] < 0.02, "near-bin leakage {}", mags[target_bin - 4]);
        assert!(mags[target_bin + 4] < 0.02, "near-bin leakage {}", mags[target_bin + 4]);
    }

    #[test]
    fn two_tones_peak_in_their_own_bins() {
        let spacing = SR / N as f32;
        let bin_a = 64_usize; // ~500 Hz
        let bin_b = 192_usize; // ~1500 Hz
        let fa = bin_a as f32 * spacing;
        let fb = bin_b as f32 * spacing;
        let a = sine(fa, SR, N * 2);
        let b = sine(fb, SR, N * 2);
        let mixed: Vec<f32> = a.iter().zip(&b).map(|(x, y)| 0.5 * (x + y)).collect();

        let mut chan = FftChannelizer::new(N, HOP, SR);
        let frames = drain(&mut chan, &mixed);
        let last = frames.last().unwrap();
        let mags: Vec<f32> = last.iter().map(|c| c.norm()).collect();

        assert!(mags[bin_a] > 0.2, "bin_a mag {}", mags[bin_a]);
        assert!(mags[bin_b] > 0.2, "bin_b mag {}", mags[bin_b]);

        // Cross-bin leakage deep below peaks.
        let mid = usize::midpoint(bin_a, bin_b);
        assert!(mags[mid] < 0.02, "mid-band leakage {}", mags[mid]);
    }

    #[test]
    fn silence_produces_near_zero_bins() {
        let mut chan = FftChannelizer::new(N, HOP, SR);
        let samples = vec![0.0_f32; N * 2];
        let frames = drain(&mut chan, &samples);
        let last = frames.last().unwrap();
        let peak = last
            .iter()
            .map(|c| c.norm())
            .fold(0.0_f32, f32::max);
        assert!(peak < 1e-6, "non-zero peak from silence: {peak}");
    }

    #[test]
    #[should_panic(expected = "hop")]
    fn rejects_hop_larger_than_fft_size() {
        let _ = FftChannelizer::new(64, 65, SR);
    }

    #[test]
    #[should_panic(expected = "fft_size")]
    fn rejects_tiny_fft_size() {
        let _ = FftChannelizer::new(1, 1, SR);
    }
}
