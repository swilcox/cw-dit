//! End-to-end decoder tests that exercise only the crate's public API.
//!
//! Each test synthesises a run-length stream of `(mark, duration)` pairs,
//! feeds it to a [`Decoder`], and compares the decoded string against the
//! input text.

use cwdit_morse::{alphabet, Decoded, Decoder, TimingEstimator};

/// Synthesise a perfectly-timed run-length stream for `text`.
///
/// Returns a sequence of `(mark, duration)` pairs where durations are in
/// multiples of the dot-unit `t`.
fn synth(text: &str, t: u32) -> Vec<(bool, u32)> {
    let mut out = Vec::new();
    let mut first_word = true;
    for word in text.split(' ') {
        if !first_word {
            out.push((false, 7 * t));
        }
        first_word = false;

        let mut first_char = true;
        for ch in word.chars() {
            if !first_char {
                out.push((false, 3 * t));
            }
            first_char = false;

            let pattern = alphabet::pattern_for_char(ch)
                .unwrap_or_else(|| panic!("no pattern for char {ch:?}"));
            let mut first_elem = true;
            for glyph in pattern.chars() {
                if !first_elem {
                    out.push((false, t));
                }
                first_elem = false;
                let dur = match glyph {
                    '.' => t,
                    '-' => 3 * t,
                    _ => unreachable!(),
                };
                out.push((true, dur));
            }
        }
    }
    out
}

/// Collect the decoded string from a stream, rendering [`Decoded::WordBreak`]
/// as a space and [`Decoded::Unknown`] as `'?'`.
fn decode(dec: &mut Decoder, events: &[(bool, u32)]) -> String {
    let mut s = String::new();
    for &(m, d) in events {
        for event in dec.push(m, d) {
            match event {
                Decoded::Char(c) => s.push(c),
                Decoded::WordBreak => s.push(' '),
                Decoded::Unknown => s.push('?'),
            }
        }
    }
    for tail in dec.finish() {
        match tail {
            Decoded::Char(c) => s.push(c),
            Decoded::WordBreak => s.push(' '),
            Decoded::Unknown => s.push('?'),
        }
    }
    s
}

/// Apply bounded symmetric jitter to every duration in `events`.
///
/// A deterministic xorshift PRNG keeps tests reproducible without pulling in
/// `rand`. Jitter is expressed as a fraction of each duration.
fn jitter(events: &[(bool, u32)], max_fraction: f32, seed: u64) -> Vec<(bool, u32)> {
    let mut state = seed;
    let mut rng = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // Map to a float in [-1.0, 1.0]
        let bits = (state >> 40) as u32;
        (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
    };

    events
        .iter()
        .map(|&(m, d)| {
            let delta = rng() * max_fraction * d as f32;
            let jittered = (d as f32 + delta).round().max(1.0) as u32;
            (m, jittered)
        })
        .collect()
}

#[test]
fn perfect_timing_at_t1_decodes_cq_de_w1aw() {
    let text = "CQ DE W1AW";
    let events = synth(text, 1);
    let mut dec = Decoder::new(TimingEstimator::from_unit(1)).with_adapt(false);
    assert_eq!(decode(&mut dec, &events), text);
}

#[test]
fn perfect_timing_at_realistic_sample_rate() {
    // 20 WPM at 48 kHz → dot-unit is 2880 samples.
    let t = 2_880;
    let text = "HELLO WORLD";
    let events = synth(text, t);
    let mut dec =
        Decoder::new(TimingEstimator::from_wpm(20.0, 48_000.0)).with_adapt(false);
    assert_eq!(decode(&mut dec, &events), text);
}

#[test]
fn decodes_digits_and_punctuation() {
    let text = "DE W1AW 73 ES CUL";
    let events = synth(text, 10);
    let mut dec = Decoder::new(TimingEstimator::from_unit(10)).with_adapt(false);
    assert_eq!(decode(&mut dec, &events), text);
}

#[test]
fn survives_ten_percent_jitter() {
    let text = "CQ CQ CQ DE W1AW W1AW K";
    let clean = synth(text, 100);
    let noisy = jitter(&clean, 0.10, 0x00C0_FFEE);
    let mut dec = Decoder::new(TimingEstimator::from_unit(100));
    assert_eq!(decode(&mut dec, &noisy), text);
}

#[test]
fn adapts_from_wrong_initial_estimate() {
    // True T = 100, but seed the decoder thinking T = 60.
    // Adaptation should converge quickly enough to decode after a preamble.
    let preamble = synth("EEEEEEEEEE EEEEE", 100); // many dits to train on
    let message = synth("TEST DE N0CALL", 100);

    let mut dec = Decoder::new(TimingEstimator::from_unit(60));
    // Drive the preamble; we don't assert on its output — it may contain
    // errors while the estimator converges.
    let _ = decode(&mut dec, &preamble);
    let decoded = decode(&mut dec, &message);
    assert_eq!(decoded, "TEST DE N0CALL");
}

#[test]
fn different_wpm_rates_decode_cleanly() {
    for wpm in [12.0_f32, 18.0, 25.0, 35.0] {
        let sample_rate = 48_000.0;
        let t = (1.2 * sample_rate / wpm).round() as u32;
        let text = "CQ TEST";
        let events = synth(text, t);
        let mut dec =
            Decoder::new(TimingEstimator::from_wpm(wpm, sample_rate)).with_adapt(false);
        assert_eq!(decode(&mut dec, &events), text, "failed at {wpm} WPM");
    }
}
