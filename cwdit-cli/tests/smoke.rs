//! Smoke test — invoke the `cwdit` binary against a synthesised WAV and
//! check the decoded stdout.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use cwdit_synth::{SynthOptions, Track, synth_to_path};

fn tmp_wav(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("cwdit-cli-smoke-{}-{name}.wav", std::process::id()));
    p
}

#[test]
fn help_prints_usage_and_exits_zero() {
    let bin = env!("CARGO_BIN_EXE_cwdit");
    let out = Command::new(bin).arg("--help").output().unwrap();
    assert!(out.status.success());
    let s = String::from_utf8(out.stdout).unwrap();
    assert!(s.contains("Decode narrow-band CW"));
    assert!(s.contains("--tone"));
    assert!(s.contains("--live"));
}

#[test]
fn decodes_synthesised_wav_via_cli() {
    let wav = tmp_wav("cq");
    let text = "CQ DE W1AW";
    synth_to_path(
        &wav,
        &[Track::new(text, 18.0, 700.0)],
        &SynthOptions::default(),
    )
    .unwrap();

    let bin = env!("CARGO_BIN_EXE_cwdit");
    let out = Command::new(bin)
        .arg(&wav)
        .args(["--tone", "700"])
        .args(["--wpm", "18"])
        .output()
        .unwrap();

    let _ = fs::remove_file(&wav);

    assert!(
        out.status.success(),
        "cwdit exited non-zero; stderr={}",
        String::from_utf8_lossy(&out.stderr),
    );
    let decoded = String::from_utf8(out.stdout).unwrap();
    assert_eq!(decoded.trim(), text);
}

#[test]
fn decodes_multi_channel_with_channels_flag() {
    let wav = tmp_wav("multi");
    let tracks = [
        Track::new("CQ DE W1AW", 18.0, 600.0),
        Track::new("QRZ", 18.0, 1_400.0),
    ];
    synth_to_path(
        &wav,
        &tracks,
        &SynthOptions {
            lead_silence_s: 0.1,
            tail_silence_s: 0.1,
            ..SynthOptions::default()
        },
    )
    .unwrap();

    let bin = env!("CARGO_BIN_EXE_cwdit");
    let out = Command::new(bin)
        .arg(&wav)
        .args(["--channels", "600,1400"])
        .args(["--wpm", "18"])
        .output()
        .unwrap();

    let _ = fs::remove_file(&wav);

    assert!(
        out.status.success(),
        "cwdit exited non-zero; stderr={}",
        String::from_utf8_lossy(&out.stderr),
    );
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(
        stdout.contains("600 Hz") && stdout.contains("CQ DE W1AW"),
        "missing ch0 in stdout: {stdout}",
    );
    assert!(
        stdout.contains("1400 Hz") && stdout.contains("QRZ"),
        "missing ch1 in stdout: {stdout}",
    );
}

#[test]
fn fft_mode_decodes_at_contest_speed() {
    // 30 WPM is normal for CW contests; at 40 WPM a dit is only 30 ms, so
    // the auto-selected FFT size has to shrink below the default to keep
    // its window shorter than a dit.
    let wav = tmp_wav("fft-contest");
    let text = "CQ TEST DE W1AW K";
    synth_to_path(
        &wav,
        &[Track::new(text, 30.0, 700.0)],
        &SynthOptions::default(),
    )
    .unwrap();

    let bin = env!("CARGO_BIN_EXE_cwdit");
    let out = Command::new(bin)
        .arg(&wav)
        .args(["--fft", "--tone", "700", "--wpm", "30"])
        .output()
        .unwrap();
    let _ = fs::remove_file(&wav);

    assert!(
        out.status.success(),
        "cwdit exited non-zero; stderr={}",
        String::from_utf8_lossy(&out.stderr),
    );
    let decoded = String::from_utf8(out.stdout).unwrap();
    assert_eq!(decoded.trim(), text);
}

#[test]
fn fft_mode_decodes_multi_channel() {
    let wav = tmp_wav("fft-multi");
    let tracks = [
        Track::new("CQ DE W1AW", 25.0, 600.0),
        Track::new("QRZ", 25.0, 1_400.0),
    ];
    synth_to_path(
        &wav,
        &tracks,
        &SynthOptions {
            lead_silence_s: 0.1,
            tail_silence_s: 0.1,
            ..SynthOptions::default()
        },
    )
    .unwrap();

    let bin = env!("CARGO_BIN_EXE_cwdit");
    let out = Command::new(bin)
        .arg(&wav)
        .args(["--fft", "--channels", "600,1400", "--wpm", "25"])
        .output()
        .unwrap();
    let _ = fs::remove_file(&wav);

    assert!(
        out.status.success(),
        "cwdit exited non-zero; stderr={}",
        String::from_utf8_lossy(&out.stderr),
    );
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(
        stdout.contains("CQ DE W1AW"),
        "missing first signal: {stdout}",
    );
    assert!(stdout.contains("QRZ"), "missing second signal: {stdout}");
}

#[test]
fn fft_scan_auto_detects_signals() {
    // Three simultaneous CW signals at different tones — don't pass
    // --channels, let --scan discover them from the first few seconds.
    let wav = tmp_wav("fft-scan");
    let tracks = [
        Track::new("CQ DE W1AW", 20.0, 600.0),
        Track::new("599 TU", 20.0, 1_100.0),
        Track::new("QRZ DE N0CALL", 20.0, 1_800.0),
    ];
    synth_to_path(
        &wav,
        &tracks,
        &SynthOptions {
            lead_silence_s: 0.1,
            tail_silence_s: 0.1,
            ..SynthOptions::default()
        },
    )
    .unwrap();

    let bin = env!("CARGO_BIN_EXE_cwdit");
    let out = Command::new(bin)
        .arg(&wav)
        .args(["--fft", "--scan", "--wpm", "20"])
        .output()
        .unwrap();
    let _ = fs::remove_file(&wav);

    assert!(
        out.status.success(),
        "cwdit exited non-zero; stderr={}",
        String::from_utf8_lossy(&out.stderr),
    );
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(
        stderr.contains("scan detected 3 signal"),
        "expected 3 signals detected, got: {stderr}",
    );
    let stdout = String::from_utf8(out.stdout).unwrap();
    for track in &tracks {
        assert!(
            stdout.contains(&track.text),
            "missing {:?} in stdout:\n{stdout}",
            track.text,
        );
    }
}

#[test]
fn scan_adapts_to_wildly_different_per_channel_wpm() {
    // Two channels at very different speeds (15 and 35 WPM), with the
    // --wpm seed parked at neither (25). Without per-channel bootstrap
    // both decodes would fail: the 15-WPM signal's intra-character gaps
    // look like word gaps at a 25-WPM seed, and the 35-WPM signal's dahs
    // look like dits. With bootstrap each channel re-seeds from its own
    // first few marks.
    let wav = tmp_wav("scan-mixed-wpm");
    let tracks = [
        Track::new("CQ DE W1AW", 15.0, 600.0),
        Track::new("599 TU", 35.0, 1_400.0),
    ];
    synth_to_path(
        &wav,
        &tracks,
        &SynthOptions {
            lead_silence_s: 0.1,
            tail_silence_s: 0.1,
            ..SynthOptions::default()
        },
    )
    .unwrap();

    let bin = env!("CARGO_BIN_EXE_cwdit");
    let out = Command::new(bin)
        .arg(&wav)
        .args(["--fft", "--scan", "--wpm", "25"])
        .output()
        .unwrap();
    let _ = fs::remove_file(&wav);

    assert!(
        out.status.success(),
        "cwdit exited non-zero; stderr={}",
        String::from_utf8_lossy(&out.stderr),
    );
    let stdout = String::from_utf8(out.stdout).unwrap();
    for track in &tracks {
        assert!(
            stdout.contains(&track.text),
            "missing {:?} in stdout:\n{stdout}",
            track.text,
        );
    }
    // Per-channel WPM reported in the multi-channel header.
    assert!(
        stdout.contains("WPM"),
        "expected per-channel WPM in output:\n{stdout}",
    );
}

#[test]
fn missing_file_reports_error() {
    let bin = env!("CARGO_BIN_EXE_cwdit");
    let out = Command::new(bin)
        .arg("/definitely/does/not/exist.wav")
        .output()
        .unwrap();
    assert!(!out.status.success());
    let err = String::from_utf8(out.stderr).unwrap();
    assert!(err.contains("cwdit:"), "unexpected stderr: {err}");
}
