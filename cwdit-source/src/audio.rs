//! Live audio-input source via cpal.
//!
//! [`AudioSource`] opens a system audio input (microphone, line-in, loopback)
//! and yields mono `f32` samples normalised to `[-1.0, 1.0]`. Multi-channel
//! devices are down-mixed by taking channel 0.
//!
//! cpal drives audio on its own real-time thread; samples are forwarded to
//! [`AudioSource::read`] through a bounded channel. If the decoder falls
//! behind, the callback drops new samples and bumps an overrun counter
//! rather than blocking the audio thread.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, StreamConfig};

use crate::{Source, SourceError};

/// Live audio-input source.
pub struct AudioSource {
    _stream: Stream,
    receiver: Receiver<f32>,
    sample_rate: f32,
    overruns: Arc<AtomicUsize>,
}

impl fmt::Debug for AudioSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AudioSource")
            .field("sample_rate", &self.sample_rate)
            .field("overruns", &self.overruns.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl AudioSource {
    /// Open the default system input device.
    pub fn default_input() -> Result<Self, SourceError> {
        Self::with_device(None)
    }

    /// Open an input device by name, or the default device when `None`.
    pub fn with_device(name: Option<&str>) -> Result<Self, SourceError> {
        let host = cpal::default_host();
        let device = match name {
            Some(wanted) => host
                .input_devices()
                .map_err(|e| SourceError::Decode(format!("enumerate input devices: {e}")))?
                .find(|d| d.name().ok().as_deref() == Some(wanted))
                .ok_or_else(|| {
                    SourceError::UnsupportedFormat(format!("no input device named {wanted:?}"))
                })?,
            None => host
                .default_input_device()
                .ok_or_else(|| SourceError::UnsupportedFormat("no default input device".into()))?,
        };
        Self::from_device(&device)
    }

    fn from_device(device: &Device) -> Result<Self, SourceError> {
        let config = device
            .default_input_config()
            .map_err(|e| SourceError::Decode(format!("default input config: {e}")))?;
        let sample_format = config.sample_format();
        let stream_config: StreamConfig = config.into();
        let sample_rate = stream_config.sample_rate.0 as f32;
        let channels = stream_config.channels as usize;

        // One second of head-room at the negotiated rate — plenty for the
        // decoder to absorb scheduling jitter, bounded so a stuck consumer
        // doesn't balloon memory.
        let capacity = stream_config.sample_rate.0 as usize;
        let (tx, rx) = sync_channel::<f32>(capacity);
        let overruns = Arc::new(AtomicUsize::new(0));
        let overruns_cb = Arc::clone(&overruns);

        let err_fn = |err| eprintln!("cwdit-source: audio stream error: {err}");

        let stream = match sample_format {
            SampleFormat::F32 => {
                build_stream::<f32>(device, &stream_config, channels, tx, overruns_cb, err_fn)?
            }
            SampleFormat::I16 => {
                build_stream::<i16>(device, &stream_config, channels, tx, overruns_cb, err_fn)?
            }
            SampleFormat::U16 => {
                build_stream::<u16>(device, &stream_config, channels, tx, overruns_cb, err_fn)?
            }
            other => {
                return Err(SourceError::UnsupportedFormat(format!(
                    "audio sample format {other:?}"
                )));
            }
        };
        stream
            .play()
            .map_err(|e| SourceError::Decode(format!("start stream: {e}")))?;

        Ok(Self {
            _stream: stream,
            receiver: rx,
            sample_rate,
            overruns,
        })
    }

    /// Number of samples the audio callback has dropped because the consumer
    /// fell behind. Monotonically increasing for the life of the source.
    #[must_use]
    pub fn overruns(&self) -> usize {
        self.overruns.load(Ordering::Relaxed)
    }
}

impl Source for AudioSource {
    type Sample = f32;

    fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    fn read(&mut self, buf: &mut [f32]) -> Result<usize, SourceError> {
        if buf.is_empty() {
            return Ok(0);
        }
        // Block until the stream delivers its first sample, then drain
        // whatever else is immediately available without waiting. If the
        // sender is gone, treat it as end-of-stream.
        let Ok(first) = self.receiver.recv() else {
            return Ok(0);
        };
        buf[0] = first;
        let mut n = 1;
        while n < buf.len() {
            match self.receiver.try_recv() {
                Ok(s) => {
                    buf[n] = s;
                    n += 1;
                }
                Err(_) => break,
            }
        }
        Ok(n)
    }
}

/// Spin up a cpal input stream whose callback forwards mono samples to `tx`.
fn build_stream<T>(
    device: &Device,
    config: &StreamConfig,
    channels: usize,
    tx: SyncSender<f32>,
    overruns: Arc<AtomicUsize>,
    err_fn: impl FnMut(cpal::StreamError) + Send + 'static,
) -> Result<Stream, SourceError>
where
    T: cpal::SizedSample + ToFloat + Send + 'static,
{
    let callback = move |data: &[T], _: &cpal::InputCallbackInfo| {
        forward_frames(data, channels, &tx, &overruns);
    };
    device
        .build_input_stream(config, callback, err_fn, None)
        .map_err(|e| SourceError::Decode(format!("build input stream: {e}")))
}

/// For each interleaved frame, take channel 0, normalise to `f32`, and
/// best-effort enqueue. Full queue → count as overrun and drop; dropped
/// sender → stop processing the batch.
fn forward_frames<T: ToFloat>(
    data: &[T],
    channels: usize,
    tx: &SyncSender<f32>,
    overruns: &AtomicUsize,
) {
    let step = channels.max(1);
    for frame in data.chunks(step) {
        let v = frame[0].to_float();
        match tx.try_send(v) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                overruns.fetch_add(1, Ordering::Relaxed);
            }
            Err(TrySendError::Disconnected(_)) => return,
        }
    }
}

/// Normalise a cpal sample type to `f32` in `[-1.0, 1.0]`.
trait ToFloat: Copy {
    fn to_float(self) -> f32;
}

impl ToFloat for f32 {
    fn to_float(self) -> f32 {
        self
    }
}

impl ToFloat for i16 {
    fn to_float(self) -> f32 {
        f32::from(self) / f32::from(i16::MAX)
    }
}

impl ToFloat for u16 {
    fn to_float(self) -> f32 {
        // Unsigned PCM centres silence at 0x8000.
        (i32::from(self) - 0x8000) as f32 / 32_768.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_passthrough() {
        assert!((0.25_f32.to_float() - 0.25).abs() < 1e-6);
        assert!((-0.5_f32.to_float() + 0.5).abs() < 1e-6);
    }

    #[test]
    fn i16_normalises_to_unit_range() {
        assert!((i16::MAX.to_float() - 1.0).abs() < 1e-6);
        assert!(i16::MIN.to_float() <= -1.0);
        assert!(0_i16.to_float().abs() < 1e-6);
    }

    #[test]
    fn u16_centres_on_zero() {
        assert!((0x8000_u16.to_float()).abs() < 1e-6);
        assert!((u16::MAX.to_float() - (32_767.0 / 32_768.0)).abs() < 1e-6);
        assert!((0_u16.to_float() + 1.0).abs() < 1e-6);
    }

    #[test]
    fn forward_frames_picks_channel_zero() {
        let (tx, rx) = sync_channel::<f32>(16);
        let overruns = AtomicUsize::new(0);
        // Stereo frames: [left, right, left, right, ...]. Picking ch0 should
        // yield only the "left" values.
        let data: [f32; 6] = [0.1, 0.9, 0.2, 0.9, 0.3, 0.9];
        forward_frames(&data, 2, &tx, &overruns);
        drop(tx);
        let got: Vec<f32> = rx.iter().collect();
        assert_eq!(got, vec![0.1, 0.2, 0.3]);
        assert_eq!(overruns.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn forward_frames_counts_overruns_when_channel_full() {
        let (tx, rx) = sync_channel::<f32>(2);
        let overruns = AtomicUsize::new(0);
        // Five mono frames into a queue of 2 → 3 overruns.
        let data: [f32; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
        forward_frames(&data, 1, &tx, &overruns);
        assert_eq!(overruns.load(Ordering::Relaxed), 3);
        drop(tx);
        let got: Vec<f32> = rx.iter().collect();
        assert_eq!(got, vec![1.0, 2.0]);
    }
}
