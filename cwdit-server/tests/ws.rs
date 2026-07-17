//! Smoke tests — start the server in-process against a synthesised WAV,
//! open a WebSocket, and confirm decoded events reproduce the input.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use base64::Engine as _;
use cwdit_server::{
    Input, ServerConfig, build_app, build_app_from_iq_source, build_app_from_source,
};
use cwdit_source::{Source, SourceError};
use rustfft::num_complex::Complex32;
use cwdit_synth::{SynthOptions, Track, synth_bytes, synth_to_path};
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

/// Like [`write_synth`] but with band noise and a long noise-only tail,
/// for scan-mode tests that need channels to idle out.
fn write_synth_noisy(path: &Path, tracks: &[Track], sample_rate: u32, tail_silence_s: f32) {
    synth_to_path(
        path,
        tracks,
        &SynthOptions {
            sample_rate,
            noise_snr_db: Some(0.0),
            noise_seed: 11,
            tail_silence_s,
            ..SynthOptions::default()
        },
    )
    .expect("synth");
}

fn base_config(input: PathBuf) -> ServerConfig {
    ServerConfig {
        input: Input::Wav(input),
        tone: 700.0,
        channels: None,
        wpm: 20.0,
        fft: false,
        scan: false,
        scan_duration: 3.0,
        scan_snr_db: 8.0,
        scan_max_channels: 32,
        scan_nms_radius: 3,
        scan_min_freq: None,
        scan_max_freq: None,
        channel_timeout: 30.0,
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
    closed_channels: Vec<u64>,
    got_done: bool,
    spectrum_count: usize,
    spectrum_bin_count: Option<usize>,
    spectrum_f_min: Option<f64>,
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
                "channel_close" => {
                    out.closed_channels.push(ev["id"].as_u64().unwrap());
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
                        out.spectrum_f_min = ev["f_min"].as_f64();
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
async fn ws_scan_mode_skims_dynamic_channels_with_waterfall() {
    let wav = tmp_wav("skim");
    // Station A keys from the start; station B (8 leading spaces ≈ 3.4 s
    // of word gaps at 20 WPM) appears after the first calibration
    // interval and must still get its own channel. A long noisy tail
    // lets the channels idle out, exercising channel_close.
    write_synth_noisy(
        &wav,
        &[
            Track::new("CQ CQ DE K2D", 20.0, 620.0),
            Track::new("        WD8DSV 5NN CT", 20.0, 900.0),
        ],
        8_000,
        18.0,
    );

    let app = build_app(&ServerConfig {
        scan: true,
        channel_timeout: 4.0,
        ..base_config(wav.clone())
    })
    .expect("build_app");

    let (addr, server) = bind_app(app).await;
    let out = collect_events(format!("ws://{addr}/ws"), Duration::from_secs(20)).await;

    let _ = std::fs::remove_file(&wav);
    server.abort();

    assert_eq!(out.session_mode.as_deref(), Some("scan"));
    assert!(out.got_done, "never received done event");
    assert_eq!(out.channels.len(), 2, "expected two dynamic channels: {:?}",
        out.channels.keys().collect::<Vec<_>>());

    let by_freq = |target: f64| {
        out.channels
            .values()
            .min_by(|a, b| {
                (a.freq_hz - target).abs().total_cmp(&(b.freq_hz - target).abs())
            })
            .expect("channel")
    };
    let a = by_freq(620.0);
    assert!((a.freq_hz - 620.0).abs() < 20.0, "A freq {}", a.freq_hz);
    assert!(a.text.contains("K2D"), "A decoded: {:?}", a.text);
    let b = by_freq(900.0);
    assert!((b.freq_hz - 900.0).abs() < 20.0, "B freq {}", b.freq_hz);
    assert!(b.text.contains("5NN"), "B decoded: {:?}", b.text);

    // Both stations leave the air well before the tail ends; with a 4 s
    // timeout both channels must close before done.
    assert_eq!(out.closed_channels.len(), 2, "closed: {:?}", out.closed_channels);

    // Scan mode drives the waterfall from the detection channelizer,
    // cropped to the scanned band — not the full 0..Nyquist span.
    assert!(
        out.spectrum_count >= 4,
        "expected waterfall frames in scan mode, got {}",
        out.spectrum_count
    );
    let f_min = out.spectrum_f_min.unwrap();
    let f_max = out.spectrum_f_max.unwrap();
    assert!((250.0..=350.0).contains(&f_min), "f_min {f_min}");
    assert!((2900.0..=3100.0).contains(&f_max), "f_max {f_max}");
}

/// A live-style source for tests: loops a synthesised sample buffer
/// forever, sleeping in `read` to simulate a device delivering samples at
/// `pace` × real time.
struct LoopingPacedSource {
    samples: Vec<f32>,
    pos: usize,
    sample_rate: f32,
    pace: f32,
}

impl Source for LoopingPacedSource {
    type Sample = f32;

    fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    fn read(&mut self, buf: &mut [f32]) -> Result<usize, SourceError> {
        let n = buf.len().min(512);
        for slot in &mut buf[..n] {
            *slot = self.samples[self.pos];
            self.pos = (self.pos + 1) % self.samples.len();
        }
        std::thread::sleep(Duration::from_secs_f32(
            n as f32 / self.sample_rate / self.pace,
        ));
        Ok(n)
    }
}

/// Render tracks to an in-memory sample buffer via `cwdit-synth`.
/// `noise_snr_db` adds calibrated band noise — scan-mode tests need a
/// realistic noise floor or keying sidebands clear the detector's SNR gate.
fn synth_samples(
    tracks: &[Track],
    sample_rate: u32,
    tail_silence_s: f32,
    noise_snr_db: Option<f32>,
) -> Vec<f32> {
    let bytes = synth_bytes(
        tracks,
        &SynthOptions {
            sample_rate,
            tail_silence_s,
            noise_snr_db,
            noise_seed: 11,
            ..SynthOptions::default()
        },
    )
    .expect("synth");
    let mut source =
        cwdit_source::WavSource::from_reader(std::io::Cursor::new(bytes)).expect("wav");
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

#[tokio::test]
async fn ws_live_feed_lets_clients_join_mid_stream() {
    // "CQ TEST" repeating forever with a 1 s silent gap, delivered by a
    // paced source as if from a soundcard (8× real time to keep the test
    // quick). The client connects at some arbitrary point in the loop and
    // must lock on and decode a later repetition.
    let samples = synth_samples(&[Track::new("CQ TEST", 25.0, 700.0)], 8_000, 1.0, None);
    let source = LoopingPacedSource {
        samples,
        pos: 0,
        sample_rate: 8_000.0,
        pace: 8.0,
    };

    // `input` in the config is ignored by build_app_from_source.
    let app = build_app_from_source(
        &ServerConfig {
            wpm: 25.0,
            ..base_config(PathBuf::new())
        },
        "test live".to_owned(),
        move || Ok(source),
    )
    .expect("build_app_from_source");

    let (addr, server) = bind_app(app).await;
    let (mut ws, _resp) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws"))
        .await
        .expect("connect");

    let mut session_mode = None;
    let mut freq_hz = None;
    let mut text = String::new();
    let run = async {
        while let Some(Ok(msg)) = ws.next().await {
            let Message::Text(raw) = msg else { continue };
            let ev: serde_json::Value = serde_json::from_str(&raw).expect("json");
            match ev["type"].as_str().unwrap_or("") {
                "session" => {
                    session_mode = ev["mode"].as_str().map(std::string::ToString::to_string);
                }
                "channel_open" => freq_hz = ev["freq_hz"].as_f64(),
                "char" => text.push(ev["ch"].as_str().unwrap().chars().next().unwrap()),
                "word_break" => text.push(' '),
                _ => {}
            }
            // A mid-transmission join may garble the first repetition;
            // any later clean copy ends the test.
            if text.contains("CQ TEST") {
                break;
            }
        }
    };
    tokio::time::timeout(Duration::from_secs(20), run)
        .await
        .expect("timed out waiting for live decode");
    server.abort();

    assert_eq!(session_mode.as_deref(), Some("fixed"));
    let freq = freq_hz.expect("channel_open");
    assert!((freq - 700.0).abs() < 0.001, "freq {freq}");
    assert!(text.contains("CQ TEST"), "decoded: {text:?}");
}

/// IQ counterpart of [`LoopingPacedSource`]: plays a synthesised buffer
/// once as complex samples (Q = 0, so a real tone at `f` appears at both
/// `center ± f`), paced like a live SDR, then ends the stream.
struct FinitePacedIqSource {
    samples: Vec<Complex32>,
    pos: usize,
    sample_rate: f32,
    pace: f32,
}

impl Source for FinitePacedIqSource {
    type Sample = Complex32;

    fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    fn read(&mut self, buf: &mut [Complex32]) -> Result<usize, SourceError> {
        let n = buf.len().min(512).min(self.samples.len() - self.pos);
        buf[..n].copy_from_slice(&self.samples[self.pos..self.pos + n]);
        self.pos += n;
        std::thread::sleep(Duration::from_secs_f32(
            n as f32 / self.sample_rate / self.pace,
        ));
        Ok(n)
    }
}

#[tokio::test]
async fn ws_sdr_scan_skims_iq_source_in_rf_hz() {
    const CENTER: f32 = 7_035_000.0;
    // 700 Hz audio tone → RF station at CENTER + 700. Scanning positive
    // offsets only keeps the real-signal mirror image at CENTER - 700 out
    // of the detector's band.
    let samples = synth_samples(
        &[Track::new("CQ CQ DE K2D", 25.0, 700.0)],
        8_000,
        2.0,
        Some(0.0),
    );
    let iq: Vec<Complex32> = samples.iter().map(|&s| Complex32::new(s, 0.0)).collect();
    let source = FinitePacedIqSource {
        samples: iq,
        pos: 0,
        sample_rate: 8_000.0,
        pace: 8.0,
    };

    let app = build_app_from_iq_source(
        &ServerConfig {
            wpm: 25.0,
            scan: true,
            scan_min_freq: Some(CENTER + 300.0),
            scan_max_freq: Some(CENTER + 3_000.0),
            ..base_config(PathBuf::new())
        },
        "test sdr".to_owned(),
        CENTER,
        move || Ok(source),
    )
    .expect("build_app_from_iq_source");

    let (addr, server) = bind_app(app).await;
    let out = collect_events(format!("ws://{addr}/ws"), Duration::from_secs(20)).await;
    server.abort();

    assert_eq!(out.session_mode.as_deref(), Some("scan"));
    assert!(out.got_done, "never received done event");
    assert_eq!(
        out.channels.len(),
        1,
        "expected one skimmed channel: {:?}",
        out.channels.keys().collect::<Vec<_>>()
    );
    let ch = out.channels.values().next().expect("channel");
    // The channel must be reported in absolute RF Hz, not baseband offset.
    assert!(
        (ch.freq_hz - f64::from(CENTER + 700.0)).abs() < 20.0,
        "channel freq {} not near {}",
        ch.freq_hz,
        CENTER + 700.0
    );
    assert!(ch.text.contains("K2D"), "decoded: {:?}", ch.text);

    // The waterfall spans the scanned RF band, also in absolute RF Hz.
    assert!(
        out.spectrum_count >= 4,
        "expected waterfall frames, got {}",
        out.spectrum_count
    );
    let f_min = out.spectrum_f_min.unwrap();
    let f_max = out.spectrum_f_max.unwrap();
    assert!(
        (f_min - f64::from(CENTER + 300.0)).abs() < 50.0,
        "f_min {f_min}"
    );
    assert!(
        (f_max - f64::from(CENTER + 3_000.0)).abs() < 50.0,
        "f_max {f_max}"
    );
    // Wire frames stay within the display cap regardless of FFT width.
    let bins = out.spectrum_bin_count.unwrap();
    assert!(bins <= 2_048, "spectrum frame too wide: {bins}");
}

#[tokio::test]
async fn iq_input_without_scan_is_rejected() {
    let err = build_app_from_iq_source(
        &base_config(PathBuf::new()), // scan: false
        "test sdr".to_owned(),
        7_035_000.0,
        move || {
            Ok(FinitePacedIqSource {
                samples: vec![Complex32::new(0.0, 0.0); 8],
                pos: 0,
                sample_rate: 8_000.0,
                pace: 1.0,
            })
        },
    )
    .map(|_| ())
    .expect_err("IQ without scan must be rejected");
    assert!(err.to_string().contains("--scan"), "unexpected error: {err}");
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
