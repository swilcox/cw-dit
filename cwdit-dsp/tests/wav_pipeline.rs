//! End-to-end test for the WAV-file decode pipeline (staging step 2).
//!
//! Synthesises a tone-modulated Morse recording in memory via
//! `cwdit-synth`, runs it through `WavSource` → `Goertzel` → `Threshold`
//! → `RunLengthEncoder` → `cwdit_morse::Decoder`, and asserts the decoded
//! text matches the original input.

use std::io::Cursor;

use cwdit_dsp::{Goertzel, RunLengthEncoder, Threshold};
use cwdit_morse::{Decoded, Decoder, TimingEstimator};
use cwdit_source::{Source, WavSource};
use cwdit_synth::{SynthOptions, Track, synth_bytes};

fn synth_wav(text: &str, wpm: f32, tone_hz: f32, sample_rate: u32, lead_s: f32) -> Vec<u8> {
    synth_bytes(
        &[Track::new(text, wpm, tone_hz)],
        &SynthOptions {
            sample_rate,
            lead_silence_s: lead_s,
            ..SynthOptions::default()
        },
    )
    .expect("synth")
}

/// Run the full step-2 pipeline and return the decoded string.
fn decode_wav(
    wav_bytes: &[u8],
    tone_hz: f32,
    wpm: f32,
    block_len: u32,
    peak_half_life_s: f32,
) -> String {
    let mut source = WavSource::from_reader(Cursor::new(wav_bytes.to_vec())).unwrap();
    let sr = source.sample_rate();
    let env_rate = sr / block_len as f32;

    let mut goertzel = Goertzel::new(tone_hz, sr, block_len);
    let mut threshold = Threshold::new(env_rate, peak_half_life_s, 0.005);
    let mut rle = RunLengthEncoder::new();
    let mut decoder =
        Decoder::new(TimingEstimator::from_wpm(wpm, env_rate)).with_adapt(false);

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
                let mark = threshold.push(env);
                if let Some(run) = rle.push(mark) {
                    for ev in decoder.push(run.mark, run.duration) {
                        push_decoded(ev, &mut text);
                    }
                }
            }
        }
    }
    if let Some(run) = rle.finish() {
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
fn decodes_tone_modulated_wav_cq_de_w1aw() {
    let text = "CQ DE W1AW";
    let wav = synth_wav(text, 18.0, 700.0, 8_000, 0.20);
    assert_eq!(decode_wav(&wav, 700.0, 18.0, 32, 1.0), text);
}

#[test]
fn decodes_numbers_and_word_breaks() {
    let text = "73 ES CUL";
    let wav = synth_wav(text, 20.0, 600.0, 8_000, 0.10);
    assert_eq!(decode_wav(&wav, 600.0, 20.0, 32, 1.0), text);
}

#[test]
fn decodes_at_higher_sample_rate() {
    let text = "HELLO WORLD";
    let wav = synth_wav(text, 15.0, 750.0, 22_050, 0.15);
    // Block length scales with sample rate to preserve envelope resolution.
    assert_eq!(decode_wav(&wav, 750.0, 15.0, 64, 1.0), text);
}
