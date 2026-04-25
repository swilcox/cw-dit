//! Smoke tests — start the server in-process against a synthesised WAV,
//! open a WebSocket, and confirm decoded events reproduce the input.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use base64::Engine as _;
use cwdit_server::{ServerConfig, build_app};
use cwdit_synth::{SynthOptions, Track, synth_to_path};
use futures_util::StreamExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_tungstenite::tungstenite::Message;

fn tmp_wav(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("cwdit-server-ws-{}-{tag}.wav", std::process::id()));
    p
}

fn write_synth(path: &Path, tracks: &[Track], sample_rate: u32) {
    synth_to_path(
        path,
        tracks,
        &SynthOptions {
            sample_rate,
            ..SynthOptions::default()
        },
    )
    .expect("synth");
}

fn base_config(input: PathBuf) -> ServerConfig {
    ServerConfig {
        input,
        tone: 700.0,
        channels: None,
        wpm: 20.0,
        fft: false,
        scan: false,
        scan_duration: 3.0,
        scan_snr_db: 12.0,
        scan_max_channels: 32,
        scan_nms_radius: 3,
        scan_min_freq: 300.0,
        scan_max_freq: 3000.0,
        // Tests drive the WS directly and don't care about HTML assets;
        // `None` means axum serves the embedded stub at `/`.
        web_dir: None,
        // Push samples through ~100x faster than real-time so tests finish
        // in a fraction of a second.
        pace_factor: 100.0,
    }
}

/// Decoded per-channel text plus whichever `session` / `channel_open`
/// metadata the test cares about.
#[derive(Default)]
struct Collected {
    session_mode: Option<String>,
    channels: HashMap<u64, ChannelInfo>,
    got_done: bool,
    spectrum_count: usize,
    spectrum_bin_count: Option<usize>,
    spectrum_f_max: Option<f64>,
}

struct ChannelInfo {
    freq_hz: f64,
    text: String,
}

async fn collect_events(url: String, timeout: Duration) -> Collected {
    let (mut ws, _resp) = tokio_tungstenite::connect_async(&url).await.expect("connect");
    let mut out = Collected::default();
    let run = async {
        while let Some(Ok(msg)) = ws.next().await {
            let text = match msg {
                Message::Text(t) => t,
                Message::Close(_) => break,
                _ => continue,
            };
            let ev: serde_json::Value = match serde_json::from_str(&text) {
                Ok(v) => v,
                Err(_) => continue,
            };
            match ev["type"].as_str().unwrap_or("") {
                "session" => {
                    out.session_mode =
                        ev["mode"].as_str().map(std::string::ToString::to_string);
                }
                "channel_open" => {
                    let id = ev["id"].as_u64().unwrap();
                    out.channels.insert(
                        id,
                        ChannelInfo {
                            freq_hz: ev["freq_hz"].as_f64().unwrap(),
                            text: String::new(),
                        },
                    );
                }
                "char" => {
                    let id = ev["channel"].as_u64().unwrap();
                    let ch = ev["ch"].as_str().unwrap().chars().next().unwrap();
                    out.channels.get_mut(&id).unwrap().text.push(ch);
                }
                "word_break" => {
                    let id = ev["channel"].as_u64().unwrap();
                    out.channels.get_mut(&id).unwrap().text.push(' ');
                }
                "unknown" => {
                    let id = ev["channel"].as_u64().unwrap();
                    out.channels.get_mut(&id).unwrap().text.push('?');
                }
                "spectrum" => {
                    out.spectrum_count += 1;
                    if out.spectrum_bin_count.is_none() {
                        let b64 = ev["bins"].as_str().unwrap();
                        // Decoded length = 3/4 of base64 length when no
                        // padding is needed; we just want a non-zero count
                        // and consistency across frames.
                        let decoded =
                            base64::engine::general_purpose::STANDARD.decode(b64).unwrap();
                        out.spectrum_bin_count = Some(decoded.len());
                        out.spectrum_f_max = ev["f_max"].as_f64();
                    }
                }
                "done" => {
                    out.got_done = true;
                    break;
                }
                _ => {}
            }
        }
    };
    tokio::time::timeout(timeout, run)
        .await
        .expect("timed out waiting for decode stream");
    out
}

async fn bind_app(app: axum::Router) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    // Give axum a moment to be ready to accept.
    tokio::time::sleep(Duration::from_millis(20)).await;
    (addr, server)
}

#[tokio::test]
async fn ws_streams_decode_events_for_synth_wav() {
    let wav = tmp_wav("single");
    write_synth(&wav, &[Track::new("TEST", 25.0, 700.0)], 8_000);

    let app = build_app(&ServerConfig {
        wpm: 25.0,
        ..base_config(wav.clone())
    })
    .expect("build_app");

    let (addr, server) = bind_app(app).await;
    let out = collect_events(format!("ws://{addr}/ws"), Duration::from_secs(10)).await;

    let _ = std::fs::remove_file(&wav);
    server.abort();

    assert_eq!(out.session_mode.as_deref(), Some("fixed"));
    assert!(out.got_done, "never received done event");
    assert_eq!(out.channels.len(), 1);
    let ch0 = out.channels.get(&0).expect("channel 0");
    assert!(
        (ch0.freq_hz - 700.0).abs() < 0.001,
        "channel_open freq mismatch: {}",
        ch0.freq_hz
    );
    assert_eq!(ch0.text, "TEST");
}

#[tokio::test]
async fn ws_streams_multi_channel_with_fixed_tones() {
    let wav = tmp_wav("multi");
    write_synth(
        &wav,
        &[
            Track::new("CQ DE W1AW", 22.0, 600.0),
            Track::new("QRZ DE K5ABC", 22.0, 1400.0),
        ],
        8_000,
    );

    let app = build_app(&ServerConfig {
        channels: Some(vec![600.0, 1400.0]),
        wpm: 22.0,
        fft: true,
        ..base_config(wav.clone())
    })
    .expect("build_app");

    let (addr, server) = bind_app(app).await;
    let out = collect_events(format!("ws://{addr}/ws"), Duration::from_secs(15)).await;

    let _ = std::fs::remove_file(&wav);
    server.abort();

    assert_eq!(out.session_mode.as_deref(), Some("fixed"));
    assert!(out.got_done);
    assert_eq!(out.channels.len(), 2, "expected two channel_open events");

    let ch0 = out.channels.get(&0).expect("channel 0");
    let ch1 = out.channels.get(&1).expect("channel 1");
    // FFT bin centres are near the requested tones but not exact.
    assert!((ch0.freq_hz - 600.0).abs() < 50.0, "ch0 freq {}", ch0.freq_hz);
    assert!((ch1.freq_hz - 1400.0).abs() < 50.0, "ch1 freq {}", ch1.freq_hz);
    assert!(ch0.text.contains("W1AW"), "ch0 decoded: {:?}", ch0.text);
    assert!(ch1.text.contains("K5ABC"), "ch1 decoded: {:?}", ch1.text);
}

#[tokio::test]
async fn ws_emits_spectrum_when_fft_enabled() {
    let wav = tmp_wav("spectrum");
    write_synth(&wav, &[Track::new("CQ TEST", 22.0, 800.0)], 8_000);

    let app = build_app(&ServerConfig {
        wpm: 22.0,
        fft: true,
        ..base_config(wav.clone())
    })
    .expect("build_app");

    let (addr, server) = bind_app(app).await;
    let out = collect_events(format!("ws://{addr}/ws"), Duration::from_secs(15)).await;

    let _ = std::fs::remove_file(&wav);
    server.abort();

    assert!(out.got_done, "never received done event");
    assert!(
        out.spectrum_count >= 4,
        "expected several spectrum frames, got {}",
        out.spectrum_count,
    );
    let bin_count = out.spectrum_bin_count.unwrap();
    assert!(bin_count > 16, "implausibly few bins: {bin_count}");
    // f_max should match the Nyquist of an 8 kHz source.
    let f_max = out.spectrum_f_max.unwrap();
    assert!((f_max - 4_000.0).abs() < 1.0, "unexpected f_max: {f_max}");
}

#[tokio::test]
async fn ws_does_not_emit_spectrum_in_goertzel_mode() {
    let wav = tmp_wav("nospectrum");
    write_synth(&wav, &[Track::new("E", 25.0, 700.0)], 8_000);

    let app = build_app(&ServerConfig {
        wpm: 25.0,
        // fft defaults to false; Goertzel backend has no spectrum view.
        ..base_config(wav.clone())
    })
    .expect("build_app");

    let (addr, server) = bind_app(app).await;
    let out = collect_events(format!("ws://{addr}/ws"), Duration::from_secs(10)).await;

    let _ = std::fs::remove_file(&wav);
    server.abort();

    assert!(out.got_done);
    assert_eq!(out.spectrum_count, 0, "Goertzel mode must not emit spectrum");
}

#[tokio::test]
async fn index_route_serves_fallback_when_web_assets_missing() {
    let wav = tmp_wav("index");
    write_synth(&wav, &[Track::new("E", 25.0, 700.0)], 8_000);

    let app = build_app(&ServerConfig {
        wpm: 25.0,
        web_dir: None, // force the "assets missing" fallback.
        ..base_config(wav.clone())
    })
    .expect("build_app");

    let (addr, server) = bind_app(app).await;

    // Raw HTTP GET — avoids pulling in a full http client.
    let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    stream
        .write_all(
            format!("GET / HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n").as_bytes(),
        )
        .await
        .unwrap();
    let mut body = Vec::new();
    stream.read_to_end(&mut body).await.unwrap();
    let text = String::from_utf8_lossy(&body);

    let _ = std::fs::remove_file(&wav);
    server.abort();

    assert!(text.starts_with("HTTP/1.1 200"));
    assert!(text.contains("web assets missing"));
    assert!(text.contains("npm run build"));
}
