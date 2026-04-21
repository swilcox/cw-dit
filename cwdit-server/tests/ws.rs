//! Smoke test — start the server in-process against a synthesised WAV,
//! open a WebSocket, and confirm the decoded events reproduce the input.

use std::path::{Path, PathBuf};
use std::time::Duration;

use cwdit_server::{ServerConfig, build_app};
use cwdit_synth::{SynthOptions, Track, synth_to_path};
use futures_util::StreamExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_tungstenite::tungstenite::Message;

fn tmp_wav(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "cwdit-server-ws-{}-{tag}.wav",
        std::process::id()
    ));
    p
}

fn write_synth(path: &Path, text: &str, wpm: f32, tone: f32, sample_rate: u32) {
    synth_to_path(
        path,
        &[Track::new(text, wpm, tone)],
        &SynthOptions {
            sample_rate,
            ..SynthOptions::default()
        },
    )
    .expect("synth");
}

#[tokio::test]
async fn ws_streams_decode_events_for_synth_wav() {
    let wav = tmp_wav("test");
    write_synth(&wav, "TEST", 25.0, 700.0, 8_000);

    let app = build_app(&ServerConfig {
        input: wav.clone(),
        tone: 700.0,
        wpm: 25.0,
        // Push samples through ~100x faster than real-time so the test
        // finishes in a fraction of a second.
        pace_factor: 100.0,
    })
    .expect("build_app");

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    // Give axum a moment to be ready to accept.
    tokio::time::sleep(Duration::from_millis(20)).await;

    let url = format!("ws://{addr}/ws");
    let (mut ws, _resp) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("connect");

    let mut decoded = String::new();
    let mut got_meta = false;
    let collect = async {
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
                "meta" => {
                    got_meta = true;
                    assert_eq!(ev["tone"].as_f64().unwrap() as i32, 700);
                    assert_eq!(ev["wpm"].as_f64().unwrap() as i32, 25);
                }
                "char" => {
                    let s = ev["char"].as_str().unwrap();
                    decoded.push(s.chars().next().unwrap());
                }
                "word_break" => decoded.push(' '),
                "unknown" => decoded.push('?'),
                "done" => break,
                _ => {}
            }
        }
    };

    // Cap the overall wait so a hang surfaces as a failure, not a timeout.
    tokio::time::timeout(Duration::from_secs(10), collect)
        .await
        .expect("timed out waiting for decode stream");

    let _ = std::fs::remove_file(&wav);
    server.abort();

    assert!(got_meta, "never received meta event");
    assert_eq!(decoded, "TEST");
}

#[tokio::test]
async fn index_route_serves_html() {
    let wav = tmp_wav("index");
    write_synth(&wav, "E", 25.0, 700.0, 8_000);

    let app = build_app(&ServerConfig {
        input: wav.clone(),
        tone: 700.0,
        wpm: 25.0,
        pace_factor: 100.0,
    })
    .expect("build_app");

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    tokio::time::sleep(Duration::from_millis(20)).await;

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
    assert!(text.contains("<title>cw-dit</title>"));
    assert!(text.contains("new WebSocket"));
}
