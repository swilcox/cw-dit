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

    /// Signed fractional-bin offset of the true spectral peak around
    /// `bin`, from parabolic interpolation over the log-magnitude peak
    /// statistic of the bin and its immediate neighbours. Clamped to
    /// ±0.5 bin; returns 0.0 at the spectrum edges or when the
    /// neighbourhood isn't actually peaked at `bin`.
    ///
    /// Detection quantises each signal to its strongest bin; a decoder
    /// that can tune to arbitrary frequencies (e.g. a Goertzel bank)
    /// should add this offset so its filter sits on the signal rather
    /// than the bin grid.
    ///
    /// # Panics
    /// Panics if `bin >= bin_count()`.
    #[must_use]
    pub fn peak_offset(&self, bin: usize) -> f32 {
        assert!(bin < self.n_bins, "bin index out of range");
        if bin == 0 || bin + 1 >= self.n_bins {
            return 0.0;
        }
        let l = self.peak[bin - 1].max(f32::MIN_POSITIVE).ln();
        let c = self.peak[bin].max(f32::MIN_POSITIVE).ln();
        let r = self.peak[bin + 1].max(f32::MIN_POSITIVE).ln();
        let denom = l - 2.0 * c + r;
        if denom >= 0.0 {
            // Flat or valley-shaped neighbourhood — no peak to refine.
            return 0.0;
        }
        (0.5 * (l - r) / denom).clamp(-0.5, 0.5)
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

        // Noise-floor estimators: median across the search range, or a
        // sliding local median when `floor_radius` is set. The local form
        // matters for shaped passbands (e.g. a receiver's audio filter):
        // against a single global floor, every bin inside an elevated
        // noise band looks "loud", and the quiet region beyond the filter
        // edge drags the global median down further still.
        let peaks: Vec<f32> = (min_bin..max_bin).map(|b| self.peak(b)).collect();
        let stds: Vec<f32> = (min_bin..max_bin).map(|b| self.stddev(b)).collect();
        let floor_at = |values: &[f32], i: usize| -> f32 {
            match cfg.floor_radius {
                None => median(values),
                Some(r) => {
                    let lo = i.saturating_sub(r);
                    let hi = (i + r + 1).min(values.len());
                    median(&values[lo..hi])
                }
            }
        };

        let snr_ratio = 10_f32.powf(cfg.peak_snr_db / 20.0);

        // Candidates: (peak, bin) for every bin in range that clears both
        // thresholds against its own noise floor.
        let mut candidates: Vec<(f32, usize)> = (min_bin..max_bin)
            .filter_map(|b| {
                let i = b - min_bin;
                let p = self.peak(b);
                let s = self.stddev(b);
                let clears = p > floor_at(&peaks, i) * snr_ratio
                    && s > floor_at(&stds, i) * cfg.variance_ratio;
                clears.then_some((p, b))
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
    /// Radius (bins) of the sliding window used to estimate each bin's
    /// noise floor. `None` uses one global median over the whole search
    /// range — only appropriate when the noise spectrum is flat. Choose a
    /// radius wide enough that most window bins are unoccupied, but
    /// narrower than the passband shaping you need to track.
    pub floor_radius: Option<usize>,
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
            floor_radius: None,
            min_bin: 1, // always skip DC
            max_bin: None,
        }
    }
}

/// Drop detection candidates whose envelope keys in lockstep with a much
/// stronger neighbour — spurs and key-click sidebands of that signal, not
/// stations. `histories[i]` is the recorded envelope of `candidates[i]`
/// over the calibration interval (all equal length). A candidate within
/// `ghost_radius_bins` of a kept candidate at least `min_parent_db`
/// stronger is dropped when its envelope correlates above
/// `corr_threshold` with the parent's envelope (spurs follow the keying)
/// *or* with the parent envelope's rectified derivative (key clicks light
/// up at the keying transitions). Returns the surviving bins in ascending
/// order plus the number of ghosts dropped.
///
/// # Panics
/// Panics if `histories.len() != candidates.len()`.
#[must_use]
pub fn suppress_correlated_ghosts(
    candidates: &[usize],
    histories: &[Vec<f32>],
    stats: &BinStats,
    ghost_radius_bins: usize,
    min_parent_db: f32,
    corr_threshold: f32,
) -> (Vec<usize>, usize) {
    assert_eq!(
        histories.len(),
        candidates.len(),
        "one history per candidate"
    );
    let parent_ratio = 10_f32.powf(min_parent_db / 20.0);
    let mut order: Vec<usize> = (0..candidates.len()).collect();
    order.sort_by(|&a, &b| {
        stats
            .peak(candidates[b])
            .partial_cmp(&stats.peak(candidates[a]))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut kept: Vec<usize> = Vec::new();
    for i in order {
        let bin = candidates[i];
        let is_ghost = kept.iter().any(|&k| {
            let parent = candidates[k];
            if parent.abs_diff(bin) > ghost_radius_bins
                || stats.peak(parent) <= stats.peak(bin) * parent_ratio
                || histories[i].len() < 2
            {
                return false;
            }
            let edges: Vec<f32> = histories[k]
                .windows(2)
                .map(|w| (w[1] - w[0]).abs())
                .collect();
            envelope_correlation(&histories[k], &histories[i]) > corr_threshold
                || envelope_correlation(&edges, &histories[i][1..]) > corr_threshold
        });
        if !is_ghost {
            kept.push(i);
        }
    }
    let ghosts = candidates.len() - kept.len();
    let mut bins: Vec<usize> = kept.into_iter().map(|i| candidates[i]).collect();
    bins.sort_unstable();
    (bins, ghosts)
}

/// Pearson correlation of two equal-length envelope streams, in [-1, 1].
/// Returns 0.0 when either stream is constant (zero variance) or empty.
///
/// Used to tell a keying-click sideband from a genuine neighbouring
/// station: a sideband's envelope keys in lockstep with its parent
/// carrier (correlation near +1), while an independent station — even the
/// other side of the same QSO — keys on its own schedule (correlation
/// near zero, or negative for stations that alternate).
///
/// # Panics
/// Panics if the slices differ in length.
#[must_use]
pub fn envelope_correlation(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "envelope streams must be equal length");
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;
    let mean_a = a.iter().map(|&x| f64::from(x)).sum::<f64>() / nf;
    let mean_b = b.iter().map(|&x| f64::from(x)).sum::<f64>() / nf;
    let mut cov = 0.0_f64;
    let mut var_a = 0.0_f64;
    let mut var_b = 0.0_f64;
    for (&x, &y) in a.iter().zip(b) {
        let dx = f64::from(x) - mean_a;
        let dy = f64::from(y) - mean_b;
        cov += dx * dy;
        var_a += dx * dx;
        var_b += dy * dy;
    }
    if var_a <= 0.0 || var_b <= 0.0 {
        return 0.0;
    }
    (cov / (var_a.sqrt() * var_b.sqrt())) as f32
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
    fn ghost_suppression_drops_spurs_and_clicks_keeps_stations() {
        // Candidate 0: strong parent at bin 40. Candidate 1: spur at bin
        // 45 following the parent's envelope, 20 dB down. Candidate 2:
        // click sideband at bin 48 lighting up at the parent's keying
        // transitions. Candidate 3: independent station at bin 60, also
        // 20 dB down, keying its own alternate pattern.
        let n = 400;
        let key = |i: usize| (i / 20).is_multiple_of(2);
        let mut stats = BinStats::new(128);
        let mut hist: Vec<Vec<f32>> = vec![Vec::new(); 4];
        let mut frame = vec![0.0_f32; 128];
        let mut prev_parent = 0.0_f32;
        for i in 0..n {
            let parent = if key(i) { 1.0 } else { 0.01 };
            let edge = (parent - prev_parent).abs();
            prev_parent = parent;
            let station = if key(i) { 0.01 } else { 0.1 };
            frame[40] = parent;
            frame[45] = 0.1 * parent;
            frame[48] = 0.2 * edge + 0.005;
            frame[60] = station;
            stats.observe(&frame);
            for (h, &b) in hist.iter_mut().zip(&[40usize, 45, 48, 60]) {
                h.push(frame[b]);
            }
        }
        let (kept, ghosts) =
            suppress_correlated_ghosts(&[40, 45, 48, 60], &hist, &stats, 32, 6.0, 0.5);
        assert_eq!(kept, vec![40, 60], "spur and click must drop");
        assert_eq!(ghosts, 2);
    }

    #[test]
    fn envelope_correlation_separates_lockstep_from_independent() {
        // Parent carrier keying pattern and its sideband (same pattern,
        // scaled + noisy) vs an independent station keying alternately.
        let n = 400;
        let parent: Vec<f32> = (0..n)
            .map(|i| if (i / 20) % 2 == 0 { 1.0 } else { 0.05 })
            .collect();
        let sideband: Vec<f32> = parent
            .iter()
            .enumerate()
            .map(|(i, &p)| 0.1 * p + 0.01 * rng(i as u64))
            .collect();
        let alternate: Vec<f32> = (0..n)
            .map(|i| if (i / 20) % 2 == 1 { 0.3 } else { 0.02 })
            .collect();
        assert!(envelope_correlation(&parent, &sideband) > 0.9);
        assert!(envelope_correlation(&parent, &alternate) < -0.5);
        assert!(envelope_correlation(&parent, &vec![0.5; n]).abs() < f32::EPSILON);
        assert!(envelope_correlation(&[], &[]).abs() < f32::EPSILON);
    }

    #[test]
    fn local_floor_rejects_shaped_passband_noise() {
        // A receiver-shaped spectrum: elevated keyed-ish noise across bins
        // 20..=60 (audio passband), quiet beyond, one real keyed signal at
        // bin 40 well above the plateau. A global median floor sits on the
        // quiet region and flags the entire plateau; a local floor tracks
        // the plateau and keeps only the real signal.
        let stats = stats_from(256, 400, |frame, bin| {
            let noise = rng((frame as u64) * 131 + bin as u64);
            let on = (frame / 20) % 2 == 0;
            match bin {
                40 if on => 3.0 + 0.3 * noise,
                20..=60 => 0.3 * noise, // elevated, fluctuating band
                _ => 0.02 * noise,
            }
        });
        let global = stats.detect(&ScanConfig::default());
        assert!(
            global.len() > 1,
            "expected the global floor to over-detect the plateau, got {global:?}"
        );
        let local = stats.detect(&ScanConfig {
            floor_radius: Some(12),
            ..ScanConfig::default()
        });
        assert_eq!(local, vec![40], "local floor should keep only the real signal");
    }

    #[test]
    fn peak_offset_recovers_off_grid_tone() {
        // A tone between bins 50 and 51, closer to 50 (offset +0.3),
        // shaped like a sampled window mainlobe: neighbour magnitudes
        // fall off with distance from the true peak.
        let mainlobe = |dist: f32| (-dist * dist).exp();
        let stats = stats_from(128, 50, |_, bin| match bin {
            49..=52 => mainlobe(bin as f32 - 50.3),
            _ => 0.01,
        });
        let off = stats.peak_offset(50);
        assert!(
            (off - 0.3).abs() < 0.05,
            "expected offset ~0.3, got {off}"
        );
    }

    #[test]
    fn peak_offset_is_zero_on_centred_tone_and_edges() {
        let mainlobe = |dist: f32| (-dist * dist).exp();
        let stats = stats_from(128, 50, |_, bin| match bin {
            59..=61 => mainlobe(bin as f32 - 60.0),
            _ => 0.01,
        });
        assert!(stats.peak_offset(60).abs() < 1e-3);
        assert!(stats.peak_offset(0).abs() < f32::EPSILON);
        assert!(stats.peak_offset(127).abs() < f32::EPSILON);
        // Flat noise floor: no peak shape, no refinement.
        assert!(stats.peak_offset(100).abs() < f32::EPSILON);
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
