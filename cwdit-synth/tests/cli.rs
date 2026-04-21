//! Smoke test for the `cwdit-synth` binary. Generates a WAV and confirms
//! that `cwdit-source::WavSource` can parse it back.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use cwdit_source::{Source, WavSource};

fn tmp_wav(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("cwdit-synth-smoke-{}-{tag}.wav", std::process::id()));
    p
}

#[test]
fn single_track_roundtrip_via_cli() {
    let wav = tmp_wav("single");
    let bin = env!("CARGO_BIN_EXE_cwdit-synth");
    let out = Command::new(bin)
        .args(["-o", wav.to_str().unwrap()])
        .args(["-t", "CQ DE W1AW"])
        .args(["-f", "700"])
        .args(["-w", "18"])
        .args(["-s", "8000"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "cwdit-synth failed: {}",
        String::from_utf8_lossy(&out.stderr),
    );

    let src = WavSource::from_path(&wav).expect("WavSource::from_path");
    assert_eq!(src.sample_rate() as u32, 8_000);
    let _ = fs::remove_file(&wav);
}

#[test]
fn multi_channel_accepts_colon_spec() {
    let wav = tmp_wav("multi");
    let bin = env!("CARGO_BIN_EXE_cwdit-synth");
    let out = Command::new(bin)
        .args(["-o", wav.to_str().unwrap()])
        .args(["-c", "CQ DE W1AW:18:600"])
        .args(["-c", "QRZ:20:1400"])
        .args(["-s", "8000"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "cwdit-synth failed: {}",
        String::from_utf8_lossy(&out.stderr),
    );

    let src = WavSource::from_path(&wav).expect("WavSource::from_path");
    assert_eq!(src.sample_rate() as u32, 8_000);
    let _ = fs::remove_file(&wav);
}

#[test]
fn unknown_character_reports_error() {
    let wav = tmp_wav("bad");
    let bin = env!("CARGO_BIN_EXE_cwdit-synth");
    let out = Command::new(bin)
        .args(["-o", wav.to_str().unwrap()])
        // '#' has no Morse entry in the alphabet this project ships.
        .args(["-t", "A#B"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("cwdit-synth:"), "unexpected stderr: {err}");
    let _ = fs::remove_file(&wav);
}
