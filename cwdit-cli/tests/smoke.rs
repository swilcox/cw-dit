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
