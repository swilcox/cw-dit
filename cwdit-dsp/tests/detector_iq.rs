//! Integration test for the IQ [`IqDetector`]: synthesised keyed complex
//! tones on an RF grid → per-interval detection with interval reset, a
//! late-appearing station, and interval-audio replay bookkeeping.

use cwdit_dsp::skim;
use cwdit_dsp::{DetectorConfig, IqDetector, IqTone};
use rustfft::num_complex::Complex32;

const SAMPLE_RATE: f32 = 64_000.0;
const CENTER: f32 = 7_040_000.0;
const WPM: f32 = 20.0;
const INTERVAL_S: f32 = 2.0;

/// One keyed station: RF frequency, amplitude, and an on/off pattern
/// advanced once per dit.
struct Station {
    freq_hz: f32,
    amp: f32,
    pattern: &'static [bool],
    /// Sample index at which the station starts transmitting.
    start: usize,
}

/// Render stations into a complex baseband buffer with uniform noise.
fn render(stations: &[Station], n: usize, noise_amp: f32, seed: u64) -> Vec<Complex32> {
    let dit_samples = (1.2 / WPM * SAMPLE_RATE) as usize;
    let mut rng = seed;
    let mut next_noise = move || {
        // xorshift64* — deterministic, no dependency, flat in [-1, 1).
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let top24 = (rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32;
        top24 / 8_388_608.0 - 1.0
    };
    (0..n)
        .map(|i| {
            let mut s = Complex32::new(noise_amp * next_noise(), noise_amp * next_noise());
            for st in stations {
                if i < st.start {
                    continue;
                }
                let dit_idx = (i - st.start) / dit_samples;
                if !st.pattern[dit_idx % st.pattern.len()] {
                    continue;
                }
                let offset = st.freq_hz - CENTER;
                let phase = core::f32::consts::TAU * offset * i as f32 / SAMPLE_RATE;
                s += Complex32::new(st.amp * phase.cos(), st.amp * phase.sin());
            }
            s
        })
        .collect()
}

fn detector() -> IqDetector {
    let fft_size = skim::detect_iq_fft_size(SAMPLE_RATE);
    let cfg = DetectorConfig {
        fft_size,
        hop: skim::auto_hop(SAMPLE_RATE, WPM, fft_size),
        min_freq_hz: CENTER - 0.45 * SAMPLE_RATE,
        max_freq_hz: CENTER + 0.45 * SAMPLE_RATE,
        snr_db: 8.0,
        nms_radius: 3,
        max_channels: 8,
        interval_s: INTERVAL_S,
    };
    IqDetector::new_iq(&cfg, SAMPLE_RATE, CENTER)
}

/// Run `samples` through `det`, collecting each completed interval's
/// detections.
fn run(det: &mut IqDetector, samples: &[Complex32]) -> Vec<Vec<f32>> {
    let mut rounds = Vec::new();
    let mut frames = 0_usize;
    for &s in samples {
        if det.push(s) {
            frames += 1;
        }
        if det.interval_complete() {
            rounds.push(det.detect());
            det.reset_interval();
        }
    }
    assert!(frames > 0, "channelizer never produced a frame");
    rounds
}

// Alternating patterns with different periods so the two stations'
// envelopes stay uncorrelated.
const PATTERN_A: &[bool] = &[true, false];
const PATTERN_B: &[bool] = &[true, true, false, true, false];

#[test]
fn detects_stations_either_side_of_carrier_in_rf_hz() {
    let freq_lo = CENTER - 9_400.0;
    let freq_hi = CENTER + 12_700.0;
    let n = (2.0 * INTERVAL_S * SAMPLE_RATE) as usize;
    let samples = render(
        &[
            Station {
                freq_hz: freq_lo,
                amp: 0.6,
                pattern: PATTERN_A,
                start: 0,
            },
            Station {
                freq_hz: freq_hi,
                amp: 0.4,
                pattern: PATTERN_B,
                start: 0,
            },
        ],
        n,
        0.02,
        7,
    );
    let mut det = detector();
    let spacing = SAMPLE_RATE / skim::detect_iq_fft_size(SAMPLE_RATE) as f32;
    let rounds = run(&mut det, &samples);
    assert!(!rounds.is_empty());

    let near = |tones: &[f32], f: f32| tones.iter().any(|t| (t - f).abs() <= spacing);
    for (i, round) in rounds.iter().enumerate() {
        assert!(
            near(round, freq_lo) && near(round, freq_hi),
            "interval {i} should see both stations: {round:?}"
        );
    }
}

#[test]
fn late_station_appears_in_a_later_interval() {
    let freq_a = CENTER - 5_000.0;
    let freq_b = CENTER + 3_300.0;
    let interval_len = (INTERVAL_S * SAMPLE_RATE) as usize;
    let samples = render(
        &[
            Station {
                freq_hz: freq_a,
                amp: 0.5,
                pattern: PATTERN_A,
                start: 0,
            },
            // B keys up partway through the second interval.
            Station {
                freq_hz: freq_b,
                amp: 0.5,
                pattern: PATTERN_B,
                start: interval_len + interval_len / 4,
            },
        ],
        3 * interval_len,
        0.02,
        11,
    );
    let mut det = detector();
    let spacing = SAMPLE_RATE / skim::detect_iq_fft_size(SAMPLE_RATE) as f32;
    let rounds = run(&mut det, &samples);
    assert!(rounds.len() >= 3, "got {} intervals", rounds.len());

    let near = |tones: &[f32], f: f32| tones.iter().any(|t| (t - f).abs() <= spacing);
    assert!(near(&rounds[0], freq_a), "interval 0: {:?}", rounds[0]);
    assert!(
        !near(&rounds[0], freq_b),
        "B should be silent in interval 0: {:?}",
        rounds[0]
    );
    assert!(
        near(&rounds[2], freq_a) && near(&rounds[2], freq_b),
        "interval 2 should see both: {:?}",
        rounds[2]
    );
}

#[test]
fn interval_audio_replays_into_an_iq_tone_decoder() {
    let freq = CENTER + 4_800.0;
    let interval_len = (INTERVAL_S * SAMPLE_RATE) as usize;
    let samples = render(
        &[Station {
            freq_hz: freq,
            amp: 0.5,
            pattern: PATTERN_A,
            start: 0,
        }],
        interval_len,
        0.02,
        3,
    );
    let mut det = detector();
    for &s in &samples {
        det.push(s);
    }
    assert!(det.interval_complete());
    assert_eq!(det.interval_audio().len(), interval_len);
    assert_eq!(det.interval_audio()[..16], samples[..16]);

    // The buffered interval must be decodable: replay it through an
    // IqTone at the detected frequency and check the keying swings.
    let tones = det.detect();
    let spacing = SAMPLE_RATE / skim::detect_iq_fft_size(SAMPLE_RATE) as f32;
    let detected = tones
        .iter()
        .copied()
        .find(|t| (t - freq).abs() <= spacing)
        .expect("station detected");
    let block = skim::iq_decode_block_len(SAMPLE_RATE, WPM);
    let mut filter = IqTone::new(detected - CENTER, SAMPLE_RATE, block);
    let envs: Vec<f32> = det
        .interval_audio()
        .iter()
        .filter_map(|&s| filter.push(s))
        .collect();
    let peak = envs.iter().copied().fold(0.0_f32, f32::max);
    let trough = envs.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(peak > 0.2, "replayed envelope never keyed up: peak {peak}");
    assert!(
        trough < peak * 0.2,
        "replayed envelope never keyed down: trough {trough}, peak {peak}"
    );

    det.reset_interval();
    assert!(det.interval_audio().is_empty());
    assert!(!det.interval_complete());
}

#[test]
fn waterfall_frames_cover_the_rf_scan_range() {
    let freq = CENTER - 7_100.0;
    let n = (INTERVAL_S * SAMPLE_RATE) as usize;
    let samples = render(
        &[Station {
            freq_hz: freq,
            amp: 0.6,
            pattern: &[true],
            start: 0,
        }],
        n,
        0.02,
        5,
    );
    let mut det = detector();
    let mut saw_peak = false;
    for &s in &samples {
        if det.push(s) {
            let frame = det.latest_frame().expect("frame after push");
            let (lo, hi) = det.bin_range();
            assert!(lo >= 1 && hi <= frame.len());
            let (peak_bin, _) = frame[lo..hi]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            if (det.bin_frequency(lo + peak_bin) - freq).abs() < 50.0 {
                saw_peak = true;
            }
        }
    }
    assert!(saw_peak, "waterfall frames never peaked at the keyed tone");
    assert!(det.frame_rate() > 0.0);
    // The displayed range is in RF Hz around the carrier.
    let (lo, hi) = det.bin_range();
    assert!(det.bin_frequency(lo) < CENTER);
    assert!(det.bin_frequency(hi - 1) > CENTER);
}
