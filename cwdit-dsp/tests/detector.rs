//! Integration test for the continuous [`Detector`]: synthesised CW →
//! per-interval detection with interval reset, late-appearing stations,
//! and interval-audio replay bookkeeping.

use std::io::Cursor;

use cwdit_dsp::skim;
use cwdit_dsp::{Detector, DetectorConfig};
use cwdit_source::{Source, WavSource};
use cwdit_synth::{SynthOptions, Track, synth_bytes};

const SAMPLE_RATE: u32 = 8_000;
const WPM: f32 = 20.0;

fn samples(tracks: &[Track], tail_silence_s: f32) -> Vec<f32> {
    let bytes = synth_bytes(
        tracks,
        &SynthOptions {
            sample_rate: SAMPLE_RATE,
            noise_snr_db: Some(0.0),
            noise_seed: 7,
            tail_silence_s,
            ..SynthOptions::default()
        },
    )
    .expect("synth");
    let mut source = WavSource::from_reader(Cursor::new(bytes)).expect("wav");
    let mut out = Vec::new();
    let mut buf = vec![0.0_f32; 4_096];
    loop {
        let n = source.read(&mut buf).expect("read");
        if n == 0 {
            break;
        }
        out.extend_from_slice(&buf[..n]);
    }
    out
}

fn detector() -> Detector {
    let sr = SAMPLE_RATE as f32;
    let fft_size = skim::detect_fft_size(sr, WPM);
    let cfg = DetectorConfig {
        fft_size,
        hop: skim::auto_hop(sr, WPM, fft_size),
        min_freq_hz: 300.0,
        max_freq_hz: 3_000.0,
        snr_db: 8.0,
        nms_radius: 3,
        max_channels: 32,
        interval_s: 2.0,
    };
    Detector::new(&cfg, sr)
}

/// Run `audio` through `det`, collecting each completed interval's
/// detections.
fn run(det: &mut Detector, audio: &[f32]) -> Vec<Vec<f32>> {
    let mut rounds = Vec::new();
    let mut frames = 0_usize;
    for &s in audio {
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

#[test]
fn detects_two_stations_and_forgets_after_they_stop() {
    // Both stations key from the start and stop well before the long
    // noise-only tail; early intervals must see both tones, tail
    // intervals must see nothing.
    let audio = samples(
        &[
            Track::new("CQ CQ CQ DE K2D", 20.0, 620.0),
            Track::new("WD8DSV 5NN CT", 20.0, 900.0),
        ],
        8.0,
    );
    let mut det = detector();
    let rounds = run(&mut det, &audio);
    assert!(rounds.len() >= 4, "expected several intervals, got {}", rounds.len());

    let near = |tones: &[f32], f: f32| tones.iter().any(|t| (t - f).abs() < 20.0);
    assert!(
        near(&rounds[0], 620.0) && near(&rounds[0], 900.0),
        "first interval should see both stations: {:?}",
        rounds[0]
    );
    let last = rounds.last().unwrap();
    assert!(
        last.is_empty(),
        "noise-only tail should detect nothing: {last:?}"
    );
}

#[test]
fn interval_audio_is_replayable_and_resets() {
    let audio = samples(&[Track::new("CQ", 20.0, 700.0)], 0.5);
    let mut det = detector();
    let interval_len = (2.0 * SAMPLE_RATE as f32) as usize;
    for &s in &audio[..interval_len] {
        det.push(s);
    }
    assert!(det.interval_complete());
    // The buffered interval is exactly what was pushed, in order.
    assert_eq!(det.interval_audio().len(), interval_len);
    assert_eq!(det.interval_audio()[..64], audio[..64]);
    det.reset_interval();
    assert!(det.interval_audio().is_empty());
    assert!(!det.interval_complete());
}

#[test]
fn frames_expose_waterfall_magnitudes_in_scan_range() {
    let audio = samples(&[Track::new("OOO", 20.0, 700.0)], 0.0);
    let mut det = detector();
    let mut saw_peak = false;
    for &s in &audio {
        if det.push(s) {
            let frame = det.latest_frame().expect("frame after push");
            let (lo, hi) = det.bin_range();
            assert!(lo >= 1 && hi <= frame.len());
            // During key-down the 700 Hz bin should dominate the range.
            let (peak_bin, _) = frame[lo..hi]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            if (det.bin_frequency(lo + peak_bin) - 700.0).abs() < 20.0 {
                saw_peak = true;
            }
        }
    }
    assert!(saw_peak, "waterfall frames never peaked at the keyed tone");
    assert!(det.frame_rate() > 0.0);
}
