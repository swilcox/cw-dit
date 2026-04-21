//! Decode pipeline task. One of these runs per WebSocket connection.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use cwdit_dsp::{GoertzelBank, RunLengthEncoder, Threshold};
use cwdit_morse::{Decoded, Decoder, TimingEstimator};
use cwdit_source::{Source, SourceError, WavSource};
use serde::Serialize;
use tokio::sync::mpsc;

/// Goertzel block length. At 8 kHz this gives a 4 ms envelope step, which
/// is plenty fine for CW timing up to ~40 WPM.
const BLOCK_LEN: u32 = 32;

/// Connection-independent context describing the stream being decoded.
#[derive(Clone, Debug)]
pub struct Meta {
    pub input: String,
    pub sample_rate: u32,
    pub tone: f32,
    pub wpm: f32,
}

/// Events sent to a connected WebSocket client. Serialised as
/// `{"type": "...", ...}` with `snake_case` type tags.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    Meta {
        input: String,
        sample_rate: u32,
        tone: f32,
        wpm: f32,
    },
    Char {
        r#char: char,
    },
    WordBreak,
    Unknown,
    Done,
}

impl From<&Meta> for Event {
    fn from(m: &Meta) -> Self {
        Event::Meta {
            input: m.input.clone(),
            sample_rate: m.sample_rate,
            tone: m.tone,
            wpm: m.wpm,
        }
    }
}

/// Read an entire mono WAV file into memory and return the sample buffer
/// alongside the file's sample rate.
pub fn load(path: &Path) -> Result<(Vec<f32>, f32), SourceError> {
    let mut source = WavSource::from_path(path)?;
    let sr = source.sample_rate();
    let mut samples = Vec::new();
    let mut buf = vec![0.0_f32; 4_096];
    loop {
        let n = source.read(&mut buf)?;
        if n == 0 {
            break;
        }
        samples.extend_from_slice(&buf[..n]);
    }
    Ok((samples, sr))
}

/// Stream `samples` through a fresh decode pipeline, publishing one
/// [`Event`] per decoded element. Returns when the sample buffer is
/// exhausted or the receiver is dropped.
pub async fn pump(
    samples: Arc<Vec<f32>>,
    sample_rate: f32,
    tone: f32,
    wpm: f32,
    pace_factor: f32,
    tx: mpsc::Sender<Event>,
) {
    let env_rate = sample_rate / BLOCK_LEN as f32;
    let mut bank = GoertzelBank::new(&[tone], sample_rate, BLOCK_LEN);
    let mut threshold = Threshold::new(env_rate, 1.0, 0.005);
    let mut rle = RunLengthEncoder::new();
    let mut decoder =
        Decoder::new(TimingEstimator::from_wpm(wpm, env_rate)).with_adapt(false);

    // Pace in roughly 20 ms chunks of source audio. `pace_factor` > 1.0
    // accelerates the timer so tests don't have to wait real-time.
    let chunk_samples = ((sample_rate * 0.020) as usize).max(64);
    let effective_rate = (sample_rate * pace_factor.max(0.01)).max(1.0);
    let chunk_period = Duration::from_secs_f64(f64::from(chunk_samples as u32) / f64::from(effective_rate));
    let mut interval = tokio::time::interval(chunk_period);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    for chunk in samples.chunks(chunk_samples) {
        interval.tick().await;
        for &sample in chunk {
            if let Some(envs) = bank.push(sample) {
                let mark = threshold.push(envs[0]);
                if let Some(run) = rle.push(mark) {
                    for ev in decoder.push(run.mark, run.duration) {
                        if !send_decoded(&tx, ev).await {
                            return;
                        }
                    }
                }
            }
        }
    }
    if let Some(run) = rle.finish() {
        for ev in decoder.push(run.mark, run.duration) {
            if !send_decoded(&tx, ev).await {
                return;
            }
        }
    }
    for ev in decoder.finish() {
        if !send_decoded(&tx, ev).await {
            return;
        }
    }
    let _ = tx.send(Event::Done).await;
}

async fn send_decoded(tx: &mpsc::Sender<Event>, ev: Decoded) -> bool {
    let event = match ev {
        Decoded::Char(c) => Event::Char { r#char: c },
        Decoded::WordBreak => Event::WordBreak,
        Decoded::Unknown => Event::Unknown,
    };
    tx.send(event).await.is_ok()
}
