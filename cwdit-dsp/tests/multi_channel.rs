//! End-to-end test of a multi-channel decode — two CW messages transmitted
//! simultaneously at different tone frequencies inside one WAV, each with
//! its own per-channel decoder chain fed from a single [`GoertzelBank`].

use std::io::Cursor;

use cwdit_dsp::{GoertzelBank, RunLengthEncoder, Threshold};
use cwdit_morse::{Decoded, Decoder, TimingEstimator};
use cwdit_source::{Source, WavSource};
use cwdit_synth::{SynthOptions, Track, synth_bytes};

/// A single decode chain downstream of one `GoertzelBank` channel.
struct ChannelChain {
    threshold: Threshold,
    rle: RunLengthEncoder,
    decoder: Decoder,
    text: String,
}

impl ChannelChain {
    fn new(env_rate: f32, wpm: f32) -> Self {
        Self {
            threshold: Threshold::new(env_rate, 1.0, 0.005)
                .with_absolute_on_floor(0.08),
            rle: RunLengthEncoder::new(),
            decoder: Decoder::new(TimingEstimator::from_wpm(wpm, env_rate)).with_adapt(false),
            text: String::new(),
        }
    }

    fn feed_envelope(&mut self, env: f32) {
        let mark = self.threshold.push(env);
        if let Some(run) = self.rle.push(mark) {
            for ev in self.decoder.push(run.mark, run.duration) {
                self.accumulate(ev);
            }
        }
    }

    fn finish(&mut self) {
        if let Some(run) = self.rle.finish() {
            for ev in self.decoder.push(run.mark, run.duration) {
                self.accumulate(ev);
            }
        }
        for ev in self.decoder.finish() {
            self.accumulate(ev);
        }
    }

    fn accumulate(&mut self, ev: Decoded) {
        match ev {
            Decoded::Char(c) => self.text.push(c),
            Decoded::WordBreak => self.text.push(' '),
            Decoded::Unknown => self.text.push('?'),
        }
    }
}

#[test]
fn two_simultaneous_signals_decode_independently() {
    let tracks = [
        Track::new("CQ DE W1AW", 18.0, 600.0),
        Track::new("QRZ DE K5ABC", 20.0, 1_400.0),
    ];
    let sample_rate = 8_000_u32;
    let wav = synth_bytes(
        &tracks,
        &SynthOptions {
            sample_rate,
            lead_silence_s: 0.1,
            tail_silence_s: 0.1,
            ..SynthOptions::default()
        },
    )
    .unwrap();

    let mut source = WavSource::from_reader(Cursor::new(wav)).unwrap();
    let sr = source.sample_rate();
    let block_len = 32_u32;
    let env_rate = sr / block_len as f32;

    let tones: Vec<f32> = tracks.iter().map(|t| t.tone_hz).collect();
    let mut bank = GoertzelBank::new(&tones, sr, block_len);
    let mut chains: Vec<ChannelChain> = tracks
        .iter()
        .map(|t| ChannelChain::new(env_rate, t.wpm))
        .collect();

    let mut buf = vec![0.0_f32; 4_096];
    loop {
        let n = source.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        for &sample in &buf[..n] {
            if let Some(envs) = bank.push(sample) {
                for (i, chain) in chains.iter_mut().enumerate() {
                    chain.feed_envelope(envs[i]);
                }
            }
        }
    }
    for chain in &mut chains {
        chain.finish();
    }

    assert_eq!(chains[0].text, tracks[0].text);
    assert_eq!(chains[1].text, tracks[1].text);
}

#[test]
fn silent_channel_produces_no_false_characters() {
    // One real signal on 1000 Hz; also watch 2000 Hz, which should stay
    // empty throughout.
    let tracks = [Track::new("TEST", 20.0, 1_000.0)];
    let sample_rate = 8_000_u32;
    let wav = synth_bytes(
        &tracks,
        &SynthOptions {
            sample_rate,
            lead_silence_s: 0.1,
            tail_silence_s: 0.1,
            ..SynthOptions::default()
        },
    )
    .unwrap();

    let mut source = WavSource::from_reader(Cursor::new(wav)).unwrap();
    let sr = source.sample_rate();
    let block_len = 32_u32;
    let env_rate = sr / block_len as f32;

    let mut bank = GoertzelBank::new(&[1_000.0, 2_000.0], sr, block_len);
    let mut chains = [
        ChannelChain::new(env_rate, 20.0),
        ChannelChain::new(env_rate, 20.0),
    ];

    let mut buf = vec![0.0_f32; 4_096];
    while let Ok(n) = source.read(&mut buf) {
        if n == 0 {
            break;
        }
        for &sample in &buf[..n] {
            if let Some(envs) = bank.push(sample) {
                for (i, chain) in chains.iter_mut().enumerate() {
                    chain.feed_envelope(envs[i]);
                }
            }
        }
    }
    for chain in &mut chains {
        chain.finish();
    }

    assert_eq!(chains[0].text, "TEST");
    assert!(
        chains[1].text.is_empty(),
        "silent channel decoded: {:?}",
        chains[1].text,
    );
}
