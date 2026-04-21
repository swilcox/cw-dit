//! Parallel bank of tuned [`Goertzel`] detectors.
//!
//! A [`GoertzelBank`] wraps N independent [`Goertzel`] filters driven from a
//! single input stream. All filters share a block length, so one envelope
//! per tone is produced every `block_len` input samples — i.e. the bank
//! acts as a sparse, non-uniform N-channel envelope "channelizer" for
//! applications (like CW skimming) where the tones of interest are known
//! or discovered ahead of time.
//!
//! For uniform-grid spectrum coverage, a future FFT-based implementation
//! will live alongside this one.

use crate::envelope::Goertzel;

/// A parallel array of tuned Goertzel filters.
///
/// Each call to [`push`](Self::push) feeds the input sample to every
/// per-tone filter. When a block completes (every `block_len` samples) the
/// bank returns a slice of N envelope magnitudes in the same order as the
/// tone list passed at construction.
#[derive(Debug, Clone)]
pub struct GoertzelBank {
    filters: Vec<Goertzel>,
    envelopes: Vec<f32>,
    tones: Vec<f32>,
    sample_rate: f32,
    block_len: u32,
}

impl GoertzelBank {
    /// Build a bank of filters tuned to each frequency in `tones`.
    ///
    /// # Panics
    /// Panics if `tones` is empty, if any frequency would make
    /// [`Goertzel::new`] panic (e.g. `block_len` too short for a full
    /// cycle of that tone), or if `sample_rate_hz` or `block_len` is
    /// non-positive.
    #[must_use]
    pub fn new(tones: &[f32], sample_rate_hz: f32, block_len: u32) -> Self {
        assert!(!tones.is_empty(), "bank needs at least one tone");
        let filters: Vec<Goertzel> = tones
            .iter()
            .map(|&t| Goertzel::new(t, sample_rate_hz, block_len))
            .collect();
        let envelopes = vec![0.0; tones.len()];
        Self {
            filters,
            envelopes,
            tones: tones.to_vec(),
            sample_rate: sample_rate_hz,
            block_len,
        }
    }

    /// Number of channels in the bank.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.filters.len()
    }

    /// Block length, in input samples, between envelope outputs.
    #[must_use]
    pub const fn block_len(&self) -> u32 {
        self.block_len
    }

    /// Envelope sample rate — one envelope sample per `block_len` input
    /// samples.
    #[must_use]
    pub fn envelope_sample_rate(&self) -> f32 {
        self.sample_rate / self.block_len as f32
    }

    /// Tone frequency assigned to channel `idx`, in Hz.
    #[must_use]
    pub fn tone(&self, idx: usize) -> f32 {
        self.tones[idx]
    }

    /// Feed one input sample to every filter.
    ///
    /// Returns `Some(envs)` at the end of each block, where `envs[i]` is
    /// the magnitude of the target tone for channel `i`; otherwise `None`.
    pub fn push(&mut self, sample: f32) -> Option<&[f32]> {
        let mut ready = false;
        for (i, g) in self.filters.iter_mut().enumerate() {
            if let Some(mag) = g.push(sample) {
                self.envelopes[i] = mag;
                ready = true;
            }
        }
        if ready {
            Some(&self.envelopes)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 8_000.0;
    const BLOCK: u32 = 128;

    fn sine(freq: f32, sample_rate: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (2.0 * core::f32::consts::PI * freq * t).sin()
            })
            .collect()
    }

    fn sum_sines(freqs: &[f32], sample_rate: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate;
                freqs
                    .iter()
                    .map(|&f| (2.0 * core::f32::consts::PI * f * t).sin())
                    .sum::<f32>()
                    / freqs.len() as f32
            })
            .collect()
    }

    #[test]
    fn single_channel_matches_standalone_goertzel() {
        let mut bank = GoertzelBank::new(&[700.0], SR, BLOCK);
        let mut solo = Goertzel::new(700.0, SR, BLOCK);
        let samples = sine(700.0, SR, BLOCK as usize);
        let mut bank_out = 0.0;
        let mut solo_out = 0.0;
        for &s in &samples {
            if let Some(envs) = bank.push(s) {
                bank_out = envs[0];
            }
            if let Some(m) = solo.push(s) {
                solo_out = m;
            }
        }
        assert!(
            (bank_out - solo_out).abs() < 1e-6,
            "bank={bank_out} solo={solo_out}",
        );
    }

    #[test]
    fn channels_isolate_their_target_tones() {
        // Bank with three widely separated tones.
        let tones = [600.0, 1_000.0, 1_600.0];
        let mut bank = GoertzelBank::new(&tones, SR, BLOCK);

        // Feed a pure 1000 Hz tone — only channel 1 should respond.
        let samples = sine(1_000.0, SR, BLOCK as usize);
        let envs = feed_one_block(&mut bank, &samples);

        assert!(envs[0] < 0.05, "ch0 (600 Hz) responded to 1000 Hz: {}", envs[0]);
        assert!(envs[1] > 0.4, "ch1 (1000 Hz) failed to respond: {}", envs[1]);
        assert!(envs[2] < 0.05, "ch2 (1600 Hz) responded to 1000 Hz: {}", envs[2]);
    }

    #[test]
    fn superimposed_tones_each_detected() {
        let tones = [600.0, 1_400.0];
        let mut bank = GoertzelBank::new(&tones, SR, BLOCK);

        // Mix of the two target tones.
        let samples = sum_sines(&[600.0, 1_400.0], SR, BLOCK as usize);
        let envs = feed_one_block(&mut bank, &samples);

        // Both channels see their tone (at half amplitude because of the
        // averaging in sum_sines), with cross-talk well below signal.
        assert!(envs[0] > 0.15, "ch0 envelope {}", envs[0]);
        assert!(envs[1] > 0.15, "ch1 envelope {}", envs[1]);
    }

    #[test]
    fn emits_exactly_one_envelope_set_per_block() {
        let mut bank = GoertzelBank::new(&[700.0, 1_000.0], SR, BLOCK);
        let samples = sine(700.0, SR, BLOCK as usize * 3);
        let mut outputs = 0;
        for &s in &samples {
            if bank.push(s).is_some() {
                outputs += 1;
            }
        }
        assert_eq!(outputs, 3);
    }

    #[test]
    fn envelope_sample_rate_is_input_rate_over_block_len() {
        let bank = GoertzelBank::new(&[700.0], SR, 64);
        assert!((bank.envelope_sample_rate() - (SR / 64.0)).abs() < 1e-3);
    }

    #[test]
    #[should_panic(expected = "bank needs at least one tone")]
    fn empty_tone_list_panics() {
        let _ = GoertzelBank::new(&[], SR, BLOCK);
    }

    fn feed_one_block(bank: &mut GoertzelBank, samples: &[f32]) -> Vec<f32> {
        for &s in samples {
            if let Some(envs) = bank.push(s) {
                return envs.to_vec();
            }
        }
        panic!("bank never emitted a block");
    }
}
