//! Channel lifecycle for continuous skimming.
//!
//! A skimmer's channel list is dynamic: stations appear when they key up
//! and vanish when they QSY or the QSO ends. [`ChannelTracker`] is the
//! pure bookkeeping half of that loop. The caller re-runs detection at
//! some cadence and feeds each round's detected frequencies to
//! [`ChannelTracker::observe`]; the tracker matches them against the
//! channels it knows about and reports which frequencies need a new
//! decode channel and which existing channels have gone silent for long
//! enough to close.
//!
//! The tracker holds no DSP state — it only tracks frequencies and
//! last-seen times — so the caller must keep its own per-channel decoder
//! list in the same order and apply [`TrackerUpdate::reaped`] indices to
//! both (they are returned in descending order so `Vec::remove` is safe).

/// Configuration for [`ChannelTracker`].
#[derive(Debug, Clone)]
pub struct TrackerConfig {
    /// A detection within this distance of a live channel refreshes that
    /// channel instead of spawning a new one. Should cover detection
    /// jitter (bin quantisation, fading) but stay below the closest
    /// station spacing worth separating.
    pub match_radius_hz: f32,
    /// A channel not re-detected for this long is reaped. Generous by
    /// design: the other side of a QSO stays silent for the length of its
    /// partner's over, and reaping it would discard the decoder's adapted
    /// timing.
    pub timeout_s: f32,
    /// Cap on concurrent channels. When full, new detections are ignored
    /// until something is reaped.
    pub max_channels: usize,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            match_radius_hz: 25.0,
            timeout_s: 30.0,
            max_channels: 32,
        }
    }
}

/// One tracked channel.
#[derive(Debug, Clone)]
struct Tracked {
    freq_hz: f32,
    last_seen_s: f32,
}

/// Result of one [`ChannelTracker::observe`] round.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrackerUpdate {
    /// Frequencies that need a new decode channel, in the order they
    /// should be appended to the caller's channel list.
    pub spawned: Vec<f32>,
    /// Indices of channels that timed out, in descending order so they
    /// can be `Vec::remove`d directly from a parallel list.
    pub reaped: Vec<usize>,
}

/// Matches per-round detections against live channels; see module docs.
#[derive(Debug, Clone)]
pub struct ChannelTracker {
    cfg: TrackerConfig,
    active: Vec<Tracked>,
}

impl ChannelTracker {
    #[must_use]
    pub fn new(cfg: TrackerConfig) -> Self {
        Self {
            cfg,
            active: Vec::new(),
        }
    }

    /// Number of live channels.
    #[must_use]
    pub fn len(&self) -> usize {
        self.active.len()
    }

    /// True when no channels are live.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.active.is_empty()
    }

    /// Frequency of channel `idx`, in Hz.
    ///
    /// # Panics
    /// Panics if `idx` is out of range.
    #[must_use]
    pub fn freq(&self, idx: usize) -> f32 {
        self.active[idx].freq_hz
    }

    /// Feed one detection round observed at time `now_s`. `detections`
    /// are the ghost-filtered frequencies from the current calibration
    /// interval. Reaping also happens here, so call this every interval
    /// even when `detections` is empty.
    ///
    /// Runs in three phases — refresh matches, reap timeouts, spawn the
    /// rest — so a reaped slot frees capacity for a detection from the
    /// same round. Apply `reaped` to a parallel list before appending
    /// `spawned`.
    pub fn observe(&mut self, now_s: f32, detections: &[f32]) -> TrackerUpdate {
        let mut update = TrackerUpdate::default();

        // Refresh the nearest live channel within the match radius.
        let mut unmatched: Vec<f32> = Vec::new();
        for &f in detections {
            let nearest = self
                .active
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| (a.freq_hz - f).abs().total_cmp(&(b.freq_hz - f).abs()))
                .map(|(i, ch)| (i, (ch.freq_hz - f).abs()));
            match nearest {
                Some((i, dist)) if dist <= self.cfg.match_radius_hz => {
                    self.active[i].last_seen_s = now_s;
                }
                _ => unmatched.push(f),
            }
        }

        for idx in (0..self.active.len()).rev() {
            if now_s - self.active[idx].last_seen_s > self.cfg.timeout_s {
                self.active.remove(idx);
                update.reaped.push(idx);
            }
        }

        for f in unmatched {
            if self.active.len() >= self.cfg.max_channels {
                break;
            }
            self.active.push(Tracked {
                freq_hz: f,
                last_seen_s: now_s,
            });
            update.spawned.push(f);
        }
        update
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> TrackerConfig {
        TrackerConfig {
            match_radius_hz: 25.0,
            timeout_s: 10.0,
            max_channels: 3,
        }
    }

    #[test]
    fn spawns_new_frequencies_and_matches_repeats() {
        let mut t = ChannelTracker::new(cfg());
        let up = t.observe(0.0, &[630.0, 682.0]);
        assert_eq!(up.spawned, vec![630.0, 682.0]);
        assert!(up.reaped.is_empty());

        // Jittered re-detections refresh, not spawn.
        let up = t.observe(3.0, &[631.5, 680.0]);
        assert!(up.spawned.is_empty());
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn reaps_after_timeout_and_respawns() {
        let mut t = ChannelTracker::new(cfg());
        t.observe(0.0, &[630.0]);
        // Kept alive while re-detected, reaped after 10 s of silence.
        assert!(t.observe(9.0, &[630.0]).reaped.is_empty());
        assert!(t.observe(15.0, &[]).reaped.is_empty());
        let up = t.observe(19.5, &[]);
        assert_eq!(up.reaped, vec![0]);
        assert!(t.is_empty());

        let up = t.observe(20.0, &[630.0]);
        assert_eq!(up.spawned, vec![630.0]);
    }

    #[test]
    fn reap_indices_descend_for_parallel_removal() {
        let mut t = ChannelTracker::new(cfg());
        t.observe(0.0, &[400.0, 600.0, 800.0]);
        // Keep only the middle one alive.
        t.observe(8.0, &[600.0]);
        let up = t.observe(11.0, &[600.0]);
        assert_eq!(up.reaped, vec![2, 0]);
        assert_eq!(t.len(), 1);
        assert!((t.freq(0) - 600.0).abs() < f32::EPSILON);
    }

    #[test]
    fn max_channels_caps_spawning_until_reap() {
        let mut t = ChannelTracker::new(cfg());
        t.observe(0.0, &[100.0, 200.0, 300.0]);
        let up = t.observe(1.0, &[400.0]);
        assert!(up.spawned.is_empty(), "should ignore while full");
        assert_eq!(t.len(), 3);

        // 100/200/300 all time out at t=12; 400 can then spawn.
        let up = t.observe(12.0, &[400.0]);
        assert_eq!(up.reaped.len(), 3);
        assert_eq!(up.spawned, vec![400.0]);
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn two_detections_near_one_channel_both_refresh() {
        let mut t = ChannelTracker::new(cfg());
        t.observe(0.0, &[630.0]);
        // A pair straddling one channel must not double-spawn.
        let up = t.observe(1.0, &[620.0, 640.0]);
        assert!(up.spawned.is_empty());
        assert_eq!(t.len(), 1);
    }
}
