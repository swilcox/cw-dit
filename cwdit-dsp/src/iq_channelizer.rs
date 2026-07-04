//! Uniform-grid FFT channelizer over complex IQ baseband.
//!
//! Companion to [`FftChannelizer`](crate::channelizer::FftChannelizer): same
//! windowed-DFT filterbank, but the input is complex IQ centred on an RF
//! carrier rather than real audio. Output covers the *full* `N`-bin spectrum
//! (positive and negative offsets from the carrier), in fftshifted order so
//! frequency increases monotonically with bin index — bin `0` sits at
//! `centre - sample_rate / 2`, bin `N/2` at the carrier, bin `N - 1` at
//! `centre + (N/2 - 1) * sample_rate / N`.
//!
//! Returning bins in fftshifted order keeps every consumer of the bin grid —
//! [`BinStats`](crate::scan::BinStats) included — bin-grid agnostic: a
//! contiguous `[min_bin, max_bin]` range over RF frequency stays contiguous
//! in bin space.
//!
//! Normalisation matches [`FftChannelizer`]: a unit-amplitude tone at a bin
//! centre produces `|z| ≈ 0.5` on that bin. For a complex exponential the
//! energy lives entirely on one bin (no `+f`/`-f` split), so we halve the
//! window-sum normaliser to keep envelope levels — and therefore the
//! [`Threshold`](crate::threshold::Threshold) defaults — interchangeable
//! between the real and IQ paths.

use std::sync::Arc;

use rustfft::{Fft, FftPlanner, num_complex::Complex32};

/// Uniform-grid FFT channelizer over complex IQ baseband.
pub struct IqChannelizer {
    fft_size: usize,
    hop: usize,
    sample_rate: f32,
    center_freq: f32,
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    norm: f32,
    ring: Vec<Complex32>,
    write_idx: usize,
    filled: usize,
    samples_since_emit: usize,
    emitted_once: bool,
    fft_buf: Vec<Complex32>,
    scratch: Vec<Complex32>,
    bins: Vec<Complex32>,
}

impl IqChannelizer {
    /// Build a channelizer with FFT size `fft_size`, hop size `hop`, for
    /// IQ input at `sample_rate_hz` centred on `center_freq_hz`. Uses a Hann
    /// analysis window.
    ///
    /// # Panics
    /// Panics if `fft_size < 2`, if `fft_size` is odd, if `hop` is zero or
    /// greater than `fft_size`, or if `sample_rate_hz` is non-positive.
    #[must_use]
    pub fn new(fft_size: usize, hop: usize, sample_rate_hz: f32, center_freq_hz: f32) -> Self {
        assert!(fft_size >= 2, "fft_size must be at least 2");
        assert!(fft_size.is_multiple_of(2), "fft_size must be even");
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
        // Halve so |z| ≈ 0.5 for a unit-amp complex exponential at a bin
        // centre, matching the real-input convention used elsewhere.
        let norm = 0.5 / sum_w;
        let scratch_len = fft.get_inplace_scratch_len();

        Self {
            fft_size,
            hop,
            sample_rate: sample_rate_hz,
            center_freq: center_freq_hz,
            fft,
            window,
            norm,
            ring: vec![Complex32::new(0.0, 0.0); fft_size],
            write_idx: 0,
            filled: 0,
            samples_since_emit: 0,
            emitted_once: false,
            fft_buf: vec![Complex32::new(0.0, 0.0); fft_size],
            scratch: vec![Complex32::new(0.0, 0.0); scratch_len],
            bins: vec![Complex32::new(0.0, 0.0); fft_size],
        }
    }

    /// Number of output bins (full complex spectrum): `fft_size`.
    #[must_use]
    pub const fn channel_count(&self) -> usize {
        self.fft_size
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

    /// RF centre frequency the IQ stream is tuned to, in Hz.
    #[must_use]
    pub const fn center_freq_hz(&self) -> f32 {
        self.center_freq
    }

    /// Centre frequency of bin `idx` in RF Hz. Bins are fftshifted: bin `0`
    /// is the most negative offset, bin `N/2` is the carrier, bin `N-1` is
    /// the most positive offset below Nyquist.
    ///
    /// # Panics
    /// Panics if `idx >= channel_count()`.
    #[must_use]
    pub fn bin_frequency(&self, idx: usize) -> f32 {
        assert!(idx < self.channel_count(), "bin index out of range");
        let half = self.fft_size as i64 / 2;
        let offset = idx as i64 - half;
        self.center_freq + offset as f32 * self.bin_spacing_hz()
    }

    /// Nearest bin index for an RF frequency. Clamped to the valid range.
    #[must_use]
    pub fn bin_index_for(&self, freq_hz: f32) -> usize {
        let half = self.fft_size as i64 / 2;
        let raw = ((freq_hz - self.center_freq) / self.bin_spacing_hz()).round() as i64 + half;
        if raw < 0 {
            0
        } else {
            (raw as usize).min(self.fft_size - 1)
        }
    }

    /// Feed one complex IQ sample.
    ///
    /// Returns `Some(bins)` every `hop` samples once the ring buffer has
    /// filled (after the first `fft_size` samples); otherwise `None`. The
    /// returned slice has `channel_count()` complex values in fftshifted
    /// order — see [`bin_frequency`](Self::bin_frequency).
    pub fn push(&mut self, sample: Complex32) -> Option<&[Complex32]> {
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
            let s = self.ring[idx];
            let w = self.window[i];
            self.fft_buf[i] = Complex32::new(s.re * w, s.im * w);
        }
        self.fft
            .process_with_scratch(&mut self.fft_buf, &mut self.scratch);

        // fftshift while normalising: unshifted bin k for k < N/2 is positive
        // freq +k·Δf, which sits at shifted index k + N/2. For k >= N/2 it's
        // negative freq (k - N)·Δf, which sits at shifted index k - N/2.
        let half = self.fft_size / 2;
        for k in 0..self.fft_size {
            let shifted = if k < half { k + half } else { k - half };
            let src = self.fft_buf[k];
            self.bins[shifted] = Complex32::new(src.re * self.norm, src.im * self.norm);
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

    const SR: f32 = 1_024_000.0;
    const N: usize = 1024;
    const HOP: usize = 256;
    const CENTER: f32 = 7_040_000.0;

    fn complex_tone(freq_hz: f32, sample_rate: f32, center: f32, n: usize) -> Vec<Complex32> {
        let baseband = freq_hz - center;
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate;
                let phase = core::f32::consts::TAU * baseband * t;
                Complex32::new(phase.cos(), phase.sin())
            })
            .collect()
    }

    fn drain(chan: &mut IqChannelizer, samples: &[Complex32]) -> Vec<Vec<Complex32>> {
        let mut out = Vec::new();
        for &s in samples {
            if let Some(bins) = chan.push(s) {
                out.push(bins.to_vec());
            }
        }
        out
    }

    #[test]
    fn bin_geometry_centred_on_carrier() {
        let chan = IqChannelizer::new(N, HOP, SR, CENTER);
        assert_eq!(chan.channel_count(), N);
        assert!((chan.bin_spacing_hz() - SR / N as f32).abs() < 1e-6);
        assert!((chan.output_sample_rate() - SR / HOP as f32).abs() < 1e-6);
        // DC bin (the carrier) lives at index N/2.
        assert!((chan.bin_frequency(N / 2) - CENTER).abs() < 1e-3);
        // Bin 0 is -Fs/2 from carrier.
        assert!((chan.bin_frequency(0) - (CENTER - SR / 2.0)).abs() < 1e-3);
        // Bin N-1 is +(N/2 - 1)·Δf from carrier.
        let expected = CENTER + (N as f32 / 2.0 - 1.0) * chan.bin_spacing_hz();
        assert!((chan.bin_frequency(N - 1) - expected).abs() < 1e-3);
    }

    #[test]
    fn bin_index_for_round_trips_with_bin_frequency() {
        let chan = IqChannelizer::new(N, HOP, SR, CENTER);
        for &bin in &[0_usize, 1, 100, N / 2 - 1, N / 2, N / 2 + 1, N - 1] {
            let f = chan.bin_frequency(bin);
            assert_eq!(chan.bin_index_for(f), bin, "round-trip failed at bin {bin}");
        }
    }

    #[test]
    fn bin_index_for_clamps_out_of_range() {
        let chan = IqChannelizer::new(N, HOP, SR, CENTER);
        // Way below passband.
        assert_eq!(chan.bin_index_for(CENTER - SR), 0);
        // Way above passband.
        assert_eq!(chan.bin_index_for(CENTER + SR), N - 1);
    }

    #[test]
    fn first_emission_after_fft_size_samples_then_every_hop() {
        let mut chan = IqChannelizer::new(N, HOP, SR, CENTER);
        let samples = complex_tone(CENTER + 50_000.0, SR, CENTER, N + 4 * HOP);
        let mut emit_indices = Vec::new();
        for (i, &s) in samples.iter().enumerate() {
            if chan.push(s).is_some() {
                emit_indices.push(i);
            }
        }
        assert_eq!(emit_indices.first().copied(), Some(N - 1));
        let gaps: Vec<usize> = emit_indices.windows(2).map(|w| w[1] - w[0]).collect();
        assert!(gaps.iter().all(|&g| g == HOP), "gaps = {gaps:?}");
        assert_eq!(emit_indices.len(), 5);
    }

    #[test]
    fn positive_offset_lands_above_carrier() {
        // +100 bins from carrier — well above DC, well below Nyquist.
        let target_bin = N / 2 + 100;
        let freq = CENTER + 100.0 * (SR / N as f32);
        let mut chan = IqChannelizer::new(N, HOP, SR, CENTER);
        let samples = complex_tone(freq, SR, CENTER, N * 2);
        let frames = drain(&mut chan, &samples);
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
        assert!((chan.bin_frequency(peak_idx) - freq).abs() < 1.0);
    }

    #[test]
    fn negative_offset_lands_below_carrier() {
        let target_bin = N / 2 - 100;
        let freq = CENTER - 100.0 * (SR / N as f32);
        let mut chan = IqChannelizer::new(N, HOP, SR, CENTER);
        let samples = complex_tone(freq, SR, CENTER, N * 2);
        let frames = drain(&mut chan, &samples);
        let last = frames.last().unwrap();
        let mags: Vec<f32> = last.iter().map(|c| c.norm()).collect();
        let (peak_idx, _) = mags
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, v)| (i, *v))
            .unwrap();
        assert_eq!(peak_idx, target_bin, "peak at bin {peak_idx}, want {target_bin}");
        assert!((chan.bin_frequency(peak_idx) - freq).abs() < 1.0);
    }

    #[test]
    fn dc_tone_lands_at_bin_n_over_two() {
        let mut chan = IqChannelizer::new(N, HOP, SR, CENTER);
        let samples = complex_tone(CENTER, SR, CENTER, N * 2);
        let frames = drain(&mut chan, &samples);
        let last = frames.last().unwrap();
        let mags: Vec<f32> = last.iter().map(|c| c.norm()).collect();
        let (peak_idx, _) = mags
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, v)| (i, *v))
            .unwrap();
        assert_eq!(peak_idx, N / 2);
    }

    #[test]
    fn two_tones_either_side_of_carrier_peak_in_their_own_bins() {
        let bin_a = N / 2 - 200;
        let bin_b = N / 2 + 150;
        let fa = CENTER + (bin_a as f32 - N as f32 / 2.0) * (SR / N as f32);
        let fb = CENTER + (bin_b as f32 - N as f32 / 2.0) * (SR / N as f32);

        let a = complex_tone(fa, SR, CENTER, N * 2);
        let b = complex_tone(fb, SR, CENTER, N * 2);
        let mixed: Vec<Complex32> = a
            .iter()
            .zip(&b)
            .map(|(x, y)| Complex32::new(0.5 * (x.re + y.re), 0.5 * (x.im + y.im)))
            .collect();

        let mut chan = IqChannelizer::new(N, HOP, SR, CENTER);
        let frames = drain(&mut chan, &mixed);
        let last = frames.last().unwrap();
        let mags: Vec<f32> = last.iter().map(|c| c.norm()).collect();
        assert!(mags[bin_a] > 0.2, "bin_a mag {}", mags[bin_a]);
        assert!(mags[bin_b] > 0.2, "bin_b mag {}", mags[bin_b]);
        let mid = usize::midpoint(bin_a, bin_b);
        assert!(mags[mid] < 0.02, "mid-band leakage {}", mags[mid]);
    }

    #[test]
    fn silence_produces_near_zero_bins() {
        let mut chan = IqChannelizer::new(N, HOP, SR, CENTER);
        let samples = vec![Complex32::new(0.0, 0.0); N * 2];
        let frames = drain(&mut chan, &samples);
        let last = frames.last().unwrap();
        let peak = last.iter().map(|c| c.norm()).fold(0.0_f32, f32::max);
        assert!(peak < 1e-6, "non-zero peak from silence: {peak}");
    }

    #[test]
    #[should_panic(expected = "hop")]
    fn rejects_hop_larger_than_fft_size() {
        let _ = IqChannelizer::new(64, 65, SR, CENTER);
    }

    #[test]
    #[should_panic(expected = "fft_size must be even")]
    fn rejects_odd_fft_size() {
        let _ = IqChannelizer::new(63, 16, SR, CENTER);
    }
}
