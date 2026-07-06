//! Decode-quality harness — scores each backend's character error rate
//! (CER) against ground truth on synthesised fixtures at controlled SNRs.
//!
//! The `#[test]` functions assert regression guardrails: thresholds are
//! set from measured baselines with headroom, so they fail on a real
//! regression rather than on run-to-run noise (renders are deterministic
//! per seed).
//!
//! Two `#[ignore]`d reports print the full quality matrix for tuning work:
//!
//! ```text
//! cargo test -p cwdit-cli --test decode_quality -- --ignored --nocapture
//! ```
//!
//! `real_sample_report` additionally decodes an off-air recording when
//! `CWDIT_SAMPLE_WAV` points at a mono WAV (skips silently otherwise).
//!
//! SNRs below are full-band (white noise power vs one tone's power across
//! the whole Nyquist band, see `SynthOptions::noise_snr_db`). At 48 kHz a
//! full-band 0 dB leaves roughly +20 dB in a 170 Hz detection bandwidth,
//! so the ladder's low end is genuinely hard, not token noise.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;

use cwdit_synth::{SynthOptions, Track, synth_to_path};

const SAMPLE_RATE: u32 = 48_000;
/// Contest-style exchange: letters, digits, and both call-sign shapes.
const CORPUS_TEXT: &str = "CQ TEST DE W1AW W1AW 5NN 073 TU K2D";

fn tmp_wav(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("cwdit-quality-{}-{name}.wav", std::process::id()));
    p
}

/// Minimal Levenshtein distance; the strings here are tens of characters,
/// so the `O(len_a * len_b)` matrix is fine.
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ca != cb);
            cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Character error rate of `got` against `expected`, whitespace-normalised.
fn cer(expected: &str, got: &str) -> f32 {
    let norm = |s: &str| -> String {
        s.split_whitespace().collect::<Vec<_>>().join(" ")
    };
    let e = norm(expected);
    let g = norm(got);
    levenshtein(&e, &g) as f32 / e.chars().count() as f32
}

/// Channel summary lines (`[  620 Hz, 22.1 WPM] (full) TEXT`, or
/// `(closed)` for a reaped channel) parsed into (freq, text) pairs.
fn parse_channels(out: &str) -> Vec<(f32, String)> {
    out.lines()
        .filter_map(|l| {
            let marker = ["(full)", "(closed)"].iter().find(|m| l.contains(**m))?;
            let freq: f32 = l
                .trim_start_matches('[')
                .split("Hz")
                .next()?
                .trim()
                .parse()
                .ok()?;
            Some((freq, l.split(marker).nth(1)?.trim().to_string()))
        })
        .collect()
}

fn synth_fixture(name: &str, wpm: f32, snr_db: Option<f32>, seed: u64) -> PathBuf {
    let wav = tmp_wav(name);
    synth_to_path(
        &wav,
        &[Track::new(CORPUS_TEXT, wpm, 700.0)],
        &SynthOptions {
            sample_rate: SAMPLE_RATE,
            noise_snr_db: snr_db,
            noise_seed: seed,
            ..SynthOptions::default()
        },
    )
    .expect("synth fixture");
    wav
}

/// Run the cwdit binary on `wav` with `extra` args and return stdout.
fn decode(wav: &PathBuf, extra: &[&str]) -> String {
    let bin = env!("CARGO_BIN_EXE_cwdit");
    let out = Command::new(bin).arg(wav).args(extra).output().expect("run cwdit");
    assert!(
        out.status.success(),
        "cwdit exited non-zero; stderr={}",
        String::from_utf8_lossy(&out.stderr),
    );
    String::from_utf8(out.stdout).expect("utf8 stdout")
}

fn score(name: &str, wpm: f32, snr_db: Option<f32>, seed: u64, extra: &[&str]) -> f32 {
    let wav = synth_fixture(name, wpm, snr_db, seed);
    let mut args = vec!["--tone", "700"];
    let wpm_s = format!("{wpm}");
    args.extend_from_slice(&["--wpm", &wpm_s]);
    args.extend_from_slice(extra);
    let got = decode(&wav, &args);
    let _ = std::fs::remove_file(&wav);
    cer(CORPUS_TEXT, &got)
}

#[test]
fn levenshtein_and_cer_sanity() {
    assert_eq!(levenshtein("KITTEN", "SITTING"), 3);
    assert_eq!(levenshtein("", "ABC"), 3);
    assert!(cer("CQ TEST", "CQ TEST").abs() < f32::EPSILON);
    assert!((cer("CQ  TEST ", "CQ TEST") - 0.0).abs() < f32::EPSILON);
    assert!(cer("AAAA", "") > 0.99);
}

/// Goertzel path must stay near-perfect down to 0 dB full-band and usable
/// at -6 dB. Baselines measured at 0.00 / 0.00 / 0.00 CER; thresholds
/// leave headroom for future tuning without letting a collapse through.
#[test]
fn goertzel_cer_holds_across_snr_ladder() {
    for (snr, max_cer) in [(None, 0.02), (Some(0.0), 0.05), (Some(-6.0), 0.15)] {
        let c = score("g-ladder", 20.0, snr, 11, &[]);
        assert!(
            c <= max_cer,
            "goertzel CER {c:.3} > {max_cer} at snr {snr:?}"
        );
    }
}

/// FFT path must track the Goertzel path in noise, not just on clean
/// fixtures — this is the regression that motivated the harness: the
/// auto-sized window used to span a full dit, which decoded clean synth
/// audio but smeared keying into garbage on noisy input.
#[test]
fn fft_cer_holds_across_snr_ladder() {
    for (snr, max_cer) in [(None, 0.02), (Some(0.0), 0.05), (Some(-6.0), 0.15)] {
        let c = score("f-ladder", 20.0, snr, 11, &["--fft"]);
        assert!(c <= max_cer, "fft CER {c:.3} > {max_cer} at snr {snr:?}");
    }
}

/// Both paths at contest speed in moderate noise.
#[test]
fn both_backends_decode_contest_speed_in_noise() {
    let g = score("g-30", 30.0, Some(3.0), 23, &[]);
    let f = score("f-30", 30.0, Some(3.0), 23, &["--fft"]);
    assert!(g <= 0.05, "goertzel CER {g:.3} at 30 WPM / 3 dB");
    assert!(f <= 0.10, "fft CER {f:.3} at 30 WPM / 3 dB");
}

#[test]
#[ignore = "report for tuning work: --ignored --nocapture"]
fn quality_report() {
    // Full-band SNR at 48 kHz keeps ~21 dB more than the in-band figure in
    // a 170 Hz detection bandwidth, so the interesting cliff is far below
    // 0 dB. Narrow FFT bins keep even more, which is why long windows win
    // the deep end of this ladder while losing on real off-air signals.
    let snrs: [Option<f32>; 7] = [
        None,
        Some(0.0),
        Some(-6.0),
        Some(-10.0),
        Some(-14.0),
        Some(-18.0),
        Some(-20.0),
    ];
    println!("\nCER by backend / full-band SNR / WPM — 0.00 is perfect copy");
    println!(
        "{:<22} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}",
        "backend", "clean", "0dB", "-6dB", "-10dB", "-14dB", "-18dB", "-20dB"
    );
    for wpm in [20.0, 30.0] {
        for (label, extra) in [
            ("goertzel", vec![]),
            ("fft auto", vec!["--fft"]),
            ("fft 512", vec!["--fft", "--fft-size", "512"]),
            ("fft 1024", vec!["--fft", "--fft-size", "1024"]),
            ("fft 2048", vec!["--fft", "--fft-size", "2048"]),
        ] {
            let mut row = format!("{label:<13} {wpm:>3.0} WPM ");
            for snr in snrs {
                let c = score("report", wpm, snr, 11, &extra);
                let _ = write!(row, " {c:>6.2}");
            }
            println!("{row}");
        }
    }
}

/// Two stations 80 Hz apart, keying simultaneously at -6 dB, discovered
/// and decoded end-to-end by `--scan`: the long-window detection FFT
/// resolves the pair, peak interpolation puts each channel on its
/// station, and the per-tone Goertzel decode copies both. This is the
/// scenario the detect/decode split exists for — it must not regress.
#[test]
fn scan_separates_and_decodes_close_stations() {
    let a = Track::new("CQ TEST DE K2D K2D", 22.0, 620.0);
    let b = Track::new("WD8DSV 5NN CT TU", 25.0, 700.0);
    let wav = tmp_wav("scan-close");
    synth_to_path(
        &wav,
        &[a.clone(), b.clone()],
        &SynthOptions {
            sample_rate: SAMPLE_RATE,
            noise_snr_db: Some(-6.0),
            noise_seed: 11,
            ..SynthOptions::default()
        },
    )
    .expect("synth close-station fixture");
    let out = decode(&wav, &["--scan"]);
    let _ = std::fs::remove_file(&wav);

    let channels = parse_channels(&out);
    for track in [&a, &b] {
        let (freq, got) = channels
            .iter()
            .min_by(|x, y| {
                (x.0 - track.tone_hz)
                    .abs()
                    .total_cmp(&(y.0 - track.tone_hz).abs())
            })
            .unwrap_or_else(|| panic!("no decoded channels in output:\n{out}"));
        assert!(
            (freq - track.tone_hz).abs() < 15.0,
            "nearest channel to {} Hz is {freq} Hz:\n{out}",
            track.tone_hz
        );
        let c = cer(&track.text, got);
        assert!(
            c <= 0.1,
            "CER {c:.2} for {} Hz station; got: {got}",
            track.tone_hz
        );
    }
}

/// A station that keys up only after the first calibration interval must
/// still get a channel: the skimmer re-runs detection every interval and
/// replays the discovery interval into the new channel, so the late
/// station's first transmission is decoded from its start. (Leading
/// spaces in a track's text delay its keying — each one inserts a
/// word gap — which is how the fixture staggers the two stations.)
#[test]
fn skim_spawns_station_appearing_after_first_interval() {
    let a = Track::new("CQ CQ TEST DE K2D K2D K2D K", 20.0, 620.0);
    // 8 leading spaces = 56 dit-times ≈ 3.4 s at 20 WPM: into interval 2.
    let b = Track::new("        WD8DSV 5NN CT TU", 20.0, 700.0);
    let b_text = b.text.trim_start();
    let wav = tmp_wav("skim-late");
    synth_to_path(
        &wav,
        &[a.clone(), b.clone()],
        &SynthOptions {
            sample_rate: SAMPLE_RATE,
            noise_snr_db: Some(-6.0),
            noise_seed: 11,
            ..SynthOptions::default()
        },
    )
    .expect("synth staggered fixture");
    let out = decode(&wav, &["--scan"]);
    let _ = std::fs::remove_file(&wav);

    let channels = parse_channels(&out);
    for (tone, text) in [(a.tone_hz, a.text.as_str()), (b.tone_hz, b_text)] {
        let (freq, got) = channels
            .iter()
            .min_by(|x, y| (x.0 - tone).abs().total_cmp(&(y.0 - tone).abs()))
            .unwrap_or_else(|| panic!("no decoded channels in output:\n{out}"));
        assert!(
            (freq - tone).abs() < 15.0,
            "nearest channel to {tone} Hz is {freq} Hz:\n{out}"
        );
        let c = cer(text, got);
        assert!(c <= 0.1, "CER {c:.2} for {tone} Hz station; got: {got}");
    }
}

/// A channel that stops being detected gets closed after the timeout,
/// flushing its text with a `(closed)` summary; nothing is left to flush
/// at end of input.
#[test]
fn skim_closes_idle_channel() {
    let wav = tmp_wav("skim-reap");
    let text = "CQ DE K2D";
    synth_to_path(
        &wav,
        &[Track::new(text, 20.0, 620.0)],
        &SynthOptions {
            sample_rate: SAMPLE_RATE,
            noise_snr_db: Some(-6.0),
            noise_seed: 11,
            // Long noise-only tail: the station leaves the air and the
            // channel must idle out rather than decode noise forever.
            tail_silence_s: 18.0,
            ..SynthOptions::default()
        },
    )
    .expect("synth reap fixture");
    let out = decode(&wav, &["--scan", "--channel-timeout", "5"]);
    let _ = std::fs::remove_file(&wav);

    let closed: Vec<&str> = out.lines().filter(|l| l.contains("(closed)")).collect();
    assert_eq!(closed.len(), 1, "expected one closed channel:\n{out}");
    let (_, got) = &parse_channels(&out)[0];
    let c = cer(text, got);
    assert!(c <= 0.3, "CER {c:.2} on closed channel; got: {got}");
    assert!(
        !out.contains("(full)"),
        "reaped channel must not reappear at EOF:\n{out}"
    );
}

/// Same two-station fixture, decoded per-backend at fixed tones — keeps
/// the raw comparison visible for tuning work. Goertzel's default block
/// is wide enough to hear a strong neighbour 80 Hz away via sidelobes;
/// the FFT's narrow bins separate cleanly when they sit on the station.
#[test]
#[ignore = "report for tuning work: --ignored --nocapture"]
fn close_stations_report() {
    let a = Track::new("CQ TEST DE K2D K2D", 22.0, 620.0);
    let b = Track::new("WD8DSV 5NN CT TU", 25.0, 700.0);
    let wav = tmp_wav("close");
    synth_to_path(
        &wav,
        &[a.clone(), b.clone()],
        &SynthOptions {
            sample_rate: SAMPLE_RATE,
            noise_snr_db: Some(-6.0),
            noise_seed: 11,
            ..SynthOptions::default()
        },
    )
    .expect("synth close-station fixture");

    println!("\nTwo simultaneous stations, 620 vs 700 Hz, -6 dB full-band");
    for (label, extra) in [
        ("goertzel", vec![]),
        ("fft auto", vec!["--fft"]),
        ("fft 2048", vec!["--fft", "--fft-size", "2048"]),
    ] {
        for track in [&a, &b] {
            let tone = format!("{}", track.tone_hz);
            let wpm = format!("{}", track.wpm);
            let mut args = vec!["--tone", &tone, "--wpm", &wpm];
            args.extend_from_slice(&extra);
            let got = decode(&wav, &args);
            println!(
                "[{label:<8} @ {tone} Hz] CER {:.2}: {}",
                cer(&track.text, &got),
                got.trim()
            );
        }
    }
    let _ = std::fs::remove_file(&wav);
}

#[test]
#[ignore = "needs CWDIT_SAMPLE_WAV pointing at a mono off-air recording"]
fn real_sample_report() {
    let Ok(path) = std::env::var("CWDIT_SAMPLE_WAV") else {
        println!("CWDIT_SAMPLE_WAV not set — skipping");
        return;
    };
    let wav = PathBuf::from(path);
    // Best-known transcript of the bundled 13-Colonies recording; edit to
    // match your own sample. Off-air ground truth is approximate — treat
    // the CER as a trend indicator, not an absolute.
    let expected = std::env::var("CWDIT_SAMPLE_TEXT").unwrap_or_else(|_| {
        "W AGN S E WD8DSV 5NN CT WD8DRV TU 5NN CT RV? WD8KRV 5NN CT TU K2D CQ K2D K2D".into()
    });
    for (label, extra) in [
        ("goertzel", vec!["--tone", "680"]),
        ("fft auto", vec!["--fft", "--tone", "680"]),
        ("skim", vec!["--scan"]),
    ] {
        let got = decode(&wav, &extra);
        let channels = parse_channels(&got);
        if channels.is_empty() {
            println!("[{label}] CER {:.2}: {}", cer(&expected, &got), got.trim());
        } else {
            // Skim output is per-channel; score each against the (single
            // best-known) transcript and let the reader match sides.
            for (freq, text) in &channels {
                println!("[{label} @ {freq:.0} Hz] CER {:.2}: {text}", cer(&expected, text));
            }
        }
    }
}
