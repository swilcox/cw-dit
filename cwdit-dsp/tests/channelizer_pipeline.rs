//! End-to-end test for the FFT channelizer path: synthesised CW WAV →
//! `FftChannelizer` → `|z|` on the target bin → `Threshold` →
//! `RunLengthEncoder` → `cwdit_morse::Decoder`. Parallels the
//! `GoertzelBank`-based tests and asserts the FFT channelizer lands at the
//! same decoded output.

use std::io::Cursor;

use cwdit_dsp::{FftChannelizer, RunLengthEncoder, Threshold};
use cwdit_morse::{Decoded, Decoder, TimingEstimator};
use cwdit_source::{Source, WavSource};
use cwdit_synth::{SynthOptions, Track, synth_bytes};

/// Everything downstream of one bin of the channelizer.
struct ChannelChain {
    threshold: Threshold,
    rle: RunLengthEncoder,
    decoder: Decoder,
    text: String,
}

impl ChannelChain {
    fn new(env_rate: f32, wpm: f32) -> Self {
        Self {
            threshold: Threshold::new(env_rate, 1.0, 0.005).with_absolute_on_floor(0.08),
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

fn synth_wav(text: &str, wpm: f32, tone_hz: f32, sample_rate: u32) -> Vec<u8> {
    synth_bytes(
        &[Track::new(text, wpm, tone_hz)],
        &SynthOptions {
            sample_rate,
            lead_silence_s: 0.1,
            tail_silence_s: 0.1,
            ..SynthOptions::default()
        },
    )
    .expect("synth")
}

#[test]
fn decodes_single_cw_signal_via_channelizer() {
    // fft_size=1024, SR=8 kHz → bin spacing 7.8125 Hz. hop=64 gives an
    // envelope rate of 125 Hz, comfortably above the dit rate for CW up to
    // around 30 WPM. Round-tripping the tone through bin_index_for /
    // bin_frequency keeps it on a bin centre.
    let sample_rate = 8_000_u32;
    let fft_size = 1024_usize;
    let hop = 64_usize;
    let chan_probe = FftChannelizer::new(fft_size, hop, sample_rate as f32);
    let target_bin = chan_probe.bin_index_for(703.0);
    let tone_hz = chan_probe.bin_frequency(target_bin);

    let text = "CQ DE W1AW";
    let wpm = 18.0;
    let wav = synth_wav(text, wpm, tone_hz, sample_rate);

    let mut source = WavSource::from_reader(Cursor::new(wav)).unwrap();
    let sr = source.sample_rate();
    let mut channelizer = FftChannelizer::new(fft_size, hop, sr);
    let env_rate = channelizer.output_sample_rate();
    let mut chain = ChannelChain::new(env_rate, wpm);

    let mut buf = vec![0.0_f32; 4_096];
    loop {
        let n = source.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        for &sample in &buf[..n] {
            if let Some(bins) = channelizer.push(sample) {
                chain.feed_envelope(bins[target_bin].norm());
            }
        }
    }
    chain.finish();

    assert_eq!(chain.text, text);
}

#[test]
fn decodes_two_simultaneous_signals_via_channelizer() {
    let sample_rate = 8_000_u32;
    let fft_size = 1024_usize;
    let hop = 64_usize;
    let probe = FftChannelizer::new(fft_size, hop, sample_rate as f32);
    let bin_a = probe.bin_index_for(600.0);
    let bin_b = probe.bin_index_for(1_400.0);
    let fa = probe.bin_frequency(bin_a);
    let fb = probe.bin_frequency(bin_b);

    let tracks = [Track::new("CQ DE W1AW", 18.0, fa), Track::new("QRZ DE K5ABC", 20.0, fb)];
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
    let mut channelizer = FftChannelizer::new(fft_size, hop, sr);
    let env_rate = channelizer.output_sample_rate();
    let mut chains = [
        ChannelChain::new(env_rate, tracks[0].wpm),
        ChannelChain::new(env_rate, tracks[1].wpm),
    ];
    let bin_indices = [bin_a, bin_b];

    let mut buf = vec![0.0_f32; 4_096];
    loop {
        let n = source.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        for &sample in &buf[..n] {
            if let Some(bins) = channelizer.push(sample) {
                for (chain, &idx) in chains.iter_mut().zip(&bin_indices) {
                    chain.feed_envelope(bins[idx].norm());
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
