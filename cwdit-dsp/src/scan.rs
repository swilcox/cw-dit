//! Occupied-bin detection from [`FftChannelizer`](crate::FftChannelizer)
//! output.
//!
//! After running a calibration window through the channelizer, the caller
//! feeds per-frame bin magnitudes into a [`BinStats`] accumulator (one
//! frame = one slice of `|z|` over the full half-spectrum). Calling
//! [`BinStats::detect`] with a [`ScanConfig`] then picks the set of bins
//! that look like keyed CW signals.
//!
//! The classifier is deliberately blunt:
//!
//! - An estimated noise floor is taken as the median of per-bin peak
//!   envelope and per-bin envelope standard deviation. Median is robust so
//!   long as most bins in the search range are unoccupied — typical for
//!   amateur CW activity.
//! - A bin is a candidate iff its peak sits above the noise-floor peak by
//!   at least the configured SNR **and** its stddev sits above the
//!   noise-floor stddev by at least `variance_ratio`. The second check is
//!   what distinguishes a keyed CW bin (large envelope swings) from a
//!   steady carrier or hum, and also rejects statistical outliers in the
//!   peak statistic alone.
//! - Candidates are then non-max-suppressed within ±`nms_radius` bins so a
//!   single strong signal whose main lobe straddles a few bins registers
//!   once.

/// Running peak / mean / stddev of bin envelope magnitudes.
#[derive(Debug, Clone)]
pub struct BinStats {
    n_bins: usize,
    frames: u64,
    peak: Vec<f32>,
    sum: Vec<f64>,
    sum_sq: Vec<f64>,
}

impl BinStats {
    /// New accumulator over `n_bins` bins.
    ///
    /// # Panics
    /// Panics if `n_bins` is zero.
    #[must_use]
    pub fn new(n_bins: usize) -> Self {
        assert!(n_bins > 0, "n_bins must be positive");
        Self {
            n_bins,
            frames: 0,
            peak: vec![0.0; n_bins],
            sum: vec![0.0; n_bins],
            sum_sq: vec![0.0; n_bins],
        }
    }

    /// Number of bins this accumulator covers.
    #[must_use]
    pub const fn bin_count(&self) -> usize {
        self.n_bins
    }

    /// Number of frames observed so far.
    #[must_use]
    pub const fn frames(&self) -> u64 {
        self.frames
    }

    /// Observe one frame of envelope magnitudes (one `|z|` per bin).
    ///
    /// # Panics
    /// Panics if `mags.len() != bin_count()`.
    pub fn observe(&mut self, mags: &[f32]) {
        assert_eq!(mags.len(), self.n_bins, "frame width mismatch");
        self.frames += 1;
        for (i, &m) in mags.iter().enumerate() {
            if m > self.peak[i] {
                self.peak[i] = m;
            }
            let m64 = f64::from(m);
            self.sum[i] += m64;
            self.sum_sq[i] += m64 * m64;
        }
    }

    /// Peak observed magnitude for `bin`.
    ///
    /// # Panics
    /// Panics if `bin >= bin_count()`.
    #[must_use]
    pub fn peak(&self, bin: usize) -> f32 {
        self.peak[bin]
    }

    /// Mean magnitude for `bin`, or 0.0 if no frames have been observed.
    ///
    /// # Panics
    /// Panics if `bin >= bin_count()`.
    #[must_use]
    pub fn mean(&self, bin: usize) -> f32 {
        if self.frames == 0 {
            0.0
        } else {
            (self.sum[bin] / self.frames as f64) as f32
        }
    }

    /// Population standard deviation of magnitude for `bin`, or 0.0 if no
    /// frames have been observed. Uses the raw moment form (may be slightly
    /// noisier than Welford's online method, but calibration windows here
    /// are short and bounded so numerical loss is negligible).
    ///
    /// # Panics
    /// Panics if `bin >= bin_count()`.
    #[must_use]
    pub fn stddev(&self, bin: usize) -> f32 {
        if self.frames == 0 {
            return 0.0;
        }
        let n = self.frames as f64;
        let mean = self.sum[bin] / n;
        let var = (self.sum_sq[bin] / n) - (mean * mean);
        var.max(0.0).sqrt() as f32
    }

    /// Detect occupied bins using `cfg`. Returns bin indices sorted in
    /// ascending order.
    #[must_use]
    pub fn detect(&self, cfg: &ScanConfig) -> Vec<usize> {
        let min_bin = cfg.min_bin.min(self.n_bins);
        let max_bin = cfg.max_bin.unwrap_or(self.n_bins).min(self.n_bins);
        if self.frames == 0 || min_bin >= max_bin {
            return Vec::new();
        }

        // Noise-floor estimators: median across the search range.
        let peaks: Vec<f32> = (min_bin..max_bin).map(|b| self.peak(b)).collect();
        let stds: Vec<f32> = (min_bin..max_bin).map(|b| self.stddev(b)).collect();
        let noise_peak = median(&peaks);
        let noise_std = median(&stds);

        let peak_thresh = noise_peak * 10_f32.powf(cfg.peak_snr_db / 20.0);
        let std_thresh = noise_std * cfg.variance_ratio;

        // Candidates: (peak, bin) for every bin in range that clears both
        // thresholds.
        let mut candidates: Vec<(f32, usize)> = (min_bin..max_bin)
            .filter_map(|b| {
                let p = self.peak(b);
                let s = self.stddev(b);
                (p > peak_thresh && s > std_thresh).then_some((p, b))
            })
            .collect();
        // Strongest first so NMS always keeps the dominant peak.
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let dominance_ratio = 10_f32.powf(cfg.dominance_db / 20.0);
        let mut picked: Vec<(usize, f32)> = Vec::new();
        for (peak, bin) in candidates {
            if picked.len() >= cfg.max_channels {
                break;
            }
            let suppressed = picked.iter().any(|&(pbin, ppeak)| {
                let dist = pbin.abs_diff(bin);
                if dist <= cfg.nms_radius {
                    // Near a stronger pick — same signal's mainlobe /
                    // first-sidelobe footprint.
                    true
                } else if dist <= cfg.dominance_radius && ppeak > peak * dominance_ratio {
                    // Far enough to look like a distinct bin, but too weak
                    // relative to a nearby dominant signal — most likely
                    // its keying-spectrum sidebands.
                    true
                } else {
                    false
                }
            });
            if !suppressed {
                picked.push((bin, peak));
            }
        }
        let mut bins: Vec<usize> = picked.into_iter().map(|(b, _)| b).collect();
        bins.sort_unstable();
        bins
    }
}

/// Configuration for [`BinStats::detect`].
#[derive(Debug, Clone)]
pub struct ScanConfig {
    /// Minimum peak-to-noise-floor ratio in dB. `12.0` is a sensible start:
    /// loud enough that casual QRM doesn't register, quiet enough that
    /// average ragchew signals do.
    pub peak_snr_db: f32,
    /// Multiplier on the noise-floor stddev. `3.0` reliably separates
    /// keyed bins from steady-state noise or carriers.
    pub variance_ratio: f32,
    /// Hard-suppression radius. Any candidate within ±`nms_radius` bins of
    /// an already-picked (stronger) peak is dropped outright — these are
    /// mainlobe + first-sidelobe bins of the same signal.
    pub nms_radius: usize,
    /// Soft-suppression radius (bins). Inside this radius but outside
    /// `nms_radius`, a candidate is dropped only if it's at least
    /// `dominance_db` weaker than a neighboring picked peak. This catches
    /// the sinc-shaped keying-spectrum sidebands of a strong CW signal,
    /// which can be 10+ bins from the carrier yet still well above the
    /// noise floor.
    pub dominance_radius: usize,
    /// See [`Self::dominance_radius`]. `20.0` dB is a robust default — a
    /// candidate more than 20 dB below a nearby dominant peak is almost
    /// certainly a sideband, not a distinct signal.
    pub dominance_db: f32,
    /// Cap on the number of bins returned.
    pub max_channels: usize,
    /// First bin to consider (inclusive). Use this to skip DC / very-low
    /// frequencies where hum and audio DC offset live.
    pub min_bin: usize,
    /// Last bin to consider (exclusive). `None` means up to `bin_count`.
    pub max_bin: Option<usize>,
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            peak_snr_db: 12.0,
            variance_ratio: 3.0,
            nms_radius: 3,
            dominance_radius: 16,
            dominance_db: 20.0,
            max_channels: 32,
            min_bin: 1, // always skip DC
            max_bin: None,
        }
    }
}

fn median(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v: Vec<f32> = values.to_vec();
    let mid = v.len() / 2;
    v.select_nth_unstable_by(mid, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    v[mid]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build `BinStats` for `n_bins` by driving `n_frames` of synthetic
    /// envelope through it. `per_bin` yields the magnitude for a given
    /// (`frame_idx`, bin) pair.
    fn stats_from<F>(n_bins: usize, n_frames: usize, per_bin: F) -> BinStats
    where
        F: Fn(usize, usize) -> f32,
    {
        let mut stats = BinStats::new(n_bins);
        let mut frame = vec![0.0; n_bins];
        for i in 0..n_frames {
            for (b, slot) in frame.iter_mut().enumerate() {
                *slot = per_bin(i, b);
            }
            stats.observe(&frame);
        }
        stats
    }

    /// Generate a deterministic pseudo-random magnitude in [0, 1), suitable
    /// for mocking noise without pulling in a `rand` dev-dep.
    fn rng(seed: u64) -> f32 {
        let mut x = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
        x ^= x >> 33;
        ((x & 0x00ff_ffff) as f32) / (0x00ff_ffff as f32)
    }

    #[test]
    fn detects_keyed_bin_against_noise() {
        // 128 bins, pure noise everywhere except bin 42 which is keyed
        // on/off between "high" and "noise".
        let stats = stats_from(128, 400, |frame, bin| {
            let noise = 0.05 * rng((frame as u64) * 131 + bin as u64);
            if bin == 42 && (frame / 20) % 2 == 0 {
                1.0 + noise
            } else {
                noise
            }
        });
        let cfg = ScanConfig::default();
        let picked = stats.detect(&cfg);
        assert_eq!(picked, vec![42]);
    }

    #[test]
    fn detects_multiple_keyed_bins() {
        let stats = stats_from(128, 400, |frame, bin| {
            let noise = 0.05 * rng((frame as u64) * 131 + bin as u64);
            let on_a = bin == 20 && (frame / 15) % 2 == 0;
            let on_b = bin == 90 && (frame / 25) % 2 == 0;
            if on_a || on_b {
                1.0 + noise
            } else {
                noise
            }
        });
        let picked = stats.detect(&ScanConfig::default());
        assert_eq!(picked, vec![20, 90]);
    }

    #[test]
    fn nms_collapses_adjacent_peaks() {
        // A single signal that spectrally spills into adjacent bins. NMS
        // should keep only the strongest.
        let stats = stats_from(128, 400, |frame, bin| {
            let noise = 0.05 * rng((frame as u64) * 131 + bin as u64);
            let on = (frame / 20) % 2 == 0;
            if !on {
                return noise;
            }
            match bin {
                50 => 0.4 + noise,
                51 => 1.0 + noise, // dominant
                52 => 0.3 + noise,
                _ => noise,
            }
        });
        let picked = stats.detect(&ScanConfig::default());
        assert_eq!(picked, vec![51]);
    }

    #[test]
    fn steady_carrier_rejected_by_variance_check() {
        // Bin 30 has a strong but *constant* tone (e.g., a beacon / AM
        // carrier): its peak clears the SNR gate, but its stddev doesn't.
        // Bin 60 is a keyed CW signal.
        let stats = stats_from(128, 400, |frame, bin| {
            let noise = 0.05 * rng((frame as u64) * 131 + bin as u64);
            if bin == 30 {
                return 1.0 + noise;
            }
            if bin == 60 && (frame / 15) % 2 == 0 {
                return 1.0 + noise;
            }
            noise
        });
        let picked = stats.detect(&ScanConfig::default());
        assert_eq!(picked, vec![60], "carrier at bin 30 should be filtered out");
    }

    #[test]
    fn min_bin_max_bin_filter_search_range() {
        let stats = stats_from(128, 400, |frame, bin| {
            let noise = 0.05 * rng((frame as u64) * 131 + bin as u64);
            let on = (frame / 20) % 2 == 0;
            if bin == 5 || bin == 64 || bin == 120 {
                if on { 1.0 + noise } else { noise }
            } else {
                noise
            }
        });
        let cfg = ScanConfig {
            min_bin: 10,
            max_bin: Some(100),
            ..ScanConfig::default()
        };
        let picked = stats.detect(&cfg);
        assert_eq!(picked, vec![64]);
    }

    #[test]
    fn max_channels_caps_the_result() {
        // Ten well-separated keyed bins; cap to 3. Keep the strongest.
        let strongest = [10_usize, 30, 50, 70, 90];
        let weaker = [15_usize, 35, 55, 75, 95];
        let stats = stats_from(128, 400, |frame, bin| {
            let noise = 0.05 * rng((frame as u64) * 131 + bin as u64);
            let on = (frame / 20) % 2 == 0;
            if !on {
                return noise;
            }
            if strongest.contains(&bin) {
                1.0 + noise
            } else if weaker.contains(&bin) {
                0.5 + noise
            } else {
                noise
            }
        });
        let cfg = ScanConfig {
            max_channels: 3,
            ..ScanConfig::default()
        };
        let picked = stats.detect(&cfg);
        assert_eq!(picked.len(), 3);
        // All picks should be among the strongest set.
        for b in &picked {
            assert!(strongest.contains(b), "bin {b} not in strongest set");
        }
    }

    #[test]
    fn no_frames_yields_no_detections() {
        let stats = BinStats::new(128);
        assert!(stats.detect(&ScanConfig::default()).is_empty());
    }

    #[test]
    fn dominance_rejects_distant_sideband_of_stronger_signal() {
        // Bin 40 is the dominant keyed signal. Bin 55 (15 bins away —
        // within dominance_radius=16, outside nms_radius=3) is also keyed
        // but 30 dB weaker. Bin 80 (40 bins away, well outside
        // dominance_radius) is also keyed and weaker, and *should* survive
        // because it's far enough to be a distinct signal.
        let stats = stats_from(256, 400, |frame, bin| {
            let noise = 0.02 * rng((frame as u64) * 131 + bin as u64);
            let on = (frame / 20) % 2 == 0;
            if !on {
                return noise;
            }
            match bin {
                40 => 1.0 + noise,
                55 => 0.03 + noise, // 30 dB below bin 40
                80 => 0.1 + noise,  // 20 dB below but far away
                _ => noise,
            }
        });
        let picked = stats.detect(&ScanConfig::default());
        assert_eq!(picked, vec![40, 80]);
    }

    #[test]
    fn pure_noise_yields_no_detections() {
        let stats = stats_from(128, 400, |frame, bin| {
            0.1 * rng((frame as u64) * 131 + bin as u64)
        });
        let picked = stats.detect(&ScanConfig::default());
        assert!(picked.is_empty(), "picked {picked:?} from pure noise");
    }
}
