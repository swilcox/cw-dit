//! Noise-robustness test for the full decode chain.
//!
//! Synthesises CW over calibrated white noise via `cwdit-synth` — with a
//! long noise-only lead-in, the way a real receiver sounds — and runs it
//! through the production chain: `Goertzel` → `MovingAverage` →
//! `Threshold` → `RunLengthEncoder` → `Debouncer` → `BootstrapDecoder`.
//!
//! Three failure modes are covered, each of which used to garble output:
//! noise decoded as text during the lead-in (no squelch), a poisoned
//! bootstrap unit estimate from glitch marks, and the adaptive timing
//! collapsing on 1-tick "dits".

use std::io::Cursor;

use cwdit_dsp::{Debouncer, Goertzel, MovingAverage, RunLengthEncoder, Threshold};
use cwdit_morse::{BootstrapDecoder, Decoded, TimingEstimator};
use cwdit_source::{Source, WavSource};
use cwdit_synth::{SynthOptions, Track, synth_bytes};

const TONE_HZ: f32 = 700.0;
const SAMPLE_RATE: u32 = 8_000;
const BLOCK_LEN: u32 = 32; // envelope rate 250 Hz

fn noisy_wav(text: &str, wpm: f32, snr_db: f32, lead_s: f32) -> Vec<u8> {
    synth_bytes(
        &[Track::new(text, wpm, TONE_HZ)],
        &SynthOptions {
            sample_rate: SAMPLE_RATE,
            lead_silence_s: lead_s,
            tail_silence_s: 0.5,
            noise_snr_db: Some(snr_db),
            ..SynthOptions::default()
        },
    )
    .expect("synth")
}

/// Decode through the production chain and return the decoded string.
fn decode(wav_bytes: &[u8], seed_wpm: f32) -> String {
    let mut source = WavSource::from_reader(Cursor::new(wav_bytes.to_vec())).unwrap();
    let sr = source.sample_rate();
    let env_rate = sr / BLOCK_LEN as f32;

    let dit_ticks = 1.2 * env_rate / seed_wpm;
    let mut goertzel = Goertzel::new(TONE_HZ, sr, BLOCK_LEN);
    let mut smoother = MovingAverage::new(((dit_ticks / 4.0).round() as usize).clamp(2, 16));
    let mut threshold = Threshold::new(env_rate, 1.0, 0.005);
    let mut rle = RunLengthEncoder::new();
    let mut debouncer = Debouncer::new(((dit_ticks / 5.0) as u32).max(2));
    let mut decoder = BootstrapDecoder::new(TimingEstimator::from_wpm(seed_wpm, env_rate));

    let mut text = String::new();
    let push_decoded = |ev: Decoded, text: &mut String| match ev {
        Decoded::Char(c) => text.push(c),
        Decoded::WordBreak => text.push(' '),
        Decoded::Unknown => text.push('?'),
    };

    let mut buf = vec![0.0_f32; 4_096];
    loop {
        let n = source.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        for &sample in &buf[..n] {
            if let Some(env) = goertzel.push(sample) {
                let mark = threshold.push(smoother.push(env));
                if let Some(run) = rle.push(mark).and_then(|r| debouncer.push(r)) {
                    for ev in decoder.push(run.mark, run.duration) {
                        push_decoded(ev, &mut text);
                    }
                }
            }
        }
    }
    let tail = [
        rle.finish().and_then(|r| debouncer.push(r)),
        debouncer.finish(),
    ];
    for run in tail.into_iter().flatten() {
        for ev in decoder.push(run.mark, run.duration) {
            push_decoded(ev, &mut text);
        }
    }
    for ev in decoder.finish() {
        push_decoded(ev, &mut text);
    }
    text
}

#[test]
fn decodes_through_moderate_noise() {
    let text = "CQ CQ DE W1AW W1AW K";
    let wav = noisy_wav(text, 20.0, 20.0, 3.0);
    assert_eq!(decode(&wav, 20.0), text);
}

#[test]
fn decodes_through_heavy_noise() {
    // 6 dB full-band at 8 kHz is ~20 dB in the ~250 Hz detection
    // bandwidth — plainly audible by ear, previously undecodable.
    let text = "CQ CQ DE W1AW W1AW K";
    let wav = noisy_wav(text, 20.0, 6.0, 3.0);
    assert_eq!(decode(&wav, 20.0), text);
}

#[test]
fn noise_only_lead_in_stays_silent() {
    // A short message after a long noisy lead-in: nothing at all may be
    // decoded before the keying starts, so the output is exactly the text.
    let text = "TEST";
    let wav = noisy_wav(text, 25.0, 10.0, 5.0);
    assert_eq!(decode(&wav, 25.0), text);
}

#[test]
fn survives_wrong_wpm_seed_in_noise() {
    // Bootstrap must still rescue a 2x-off seed when the input is noisy.
    let text = "CQ DE W1AW";
    let wav = noisy_wav(text, 28.0, 12.0, 2.0);
    assert_eq!(decode(&wav, 14.0), text);
}
