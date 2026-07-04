//! Live IQ source via `SoapySDR`.
//!
//! [`SoapySource`] opens a `SoapySDR` device (`RTL-SDR`, `SDRplay`, `HackRF`,
//! …), tunes it, and yields `Complex<f32>` IQ samples through the [`Source`]
//! trait so the FFT channelizer in `cwdit-dsp` can skim every CW signal
//! across the device's bandwidth in one pass.
//!
//! Soapy drives the radio on a background thread; samples reach the consumer
//! through a bounded chunk channel. If the consumer stalls, new chunks are
//! dropped and an overrun counter increments rather than blocking the radio
//! thread.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, TrySendError, sync_channel};
use std::thread::{self, JoinHandle};

use num_complex::Complex32;
use soapysdr::{Device, Direction, ErrorCode};

use crate::{Source, SourceError};

/// Default Soapy driver args when the caller passes an empty string.
pub const DEFAULT_DRIVER_ARGS: &str = "driver=sdrplay";

/// Samples per RX read call. Sized so a 1 Msps RTL-SDR yields ~64 ms of
/// audio per chunk — short enough that scan calibration windows respond
/// promptly, long enough that the per-chunk overhead is negligible.
const RX_BUF_SAMPLES: usize = 65_536;

/// Maximum chunks queued between the radio thread and the consumer. With
/// the default `RX_BUF_SAMPLES` this is roughly 2 seconds of head-room at
/// 1 Msps — enough for scan calibration and brief consumer stalls without
/// letting a stuck consumer balloon memory.
const QUEUE_CHUNKS: usize = 32;

/// Soapy `RxStream::read` timeout. Long enough that timeouts are rare under
/// normal flow but short enough that shutdown joins quickly.
const READ_TIMEOUT_US: i64 = 100_000;

/// Live IQ source from a SoapySDR-supported radio.
pub struct SoapySource {
    receiver: Receiver<Vec<Complex32>>,
    sample_rate: f32,
    overruns: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    leftover: Vec<Complex32>,
    leftover_idx: usize,
}

impl fmt::Debug for SoapySource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SoapySource")
            .field("sample_rate", &self.sample_rate)
            .field("overruns", &self.overruns.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl SoapySource {
    /// Open a `SoapySDR` device, tune it, and start streaming IQ.
    ///
    /// `args` is a `SoapySDR` device-args string such as `"driver=rtlsdr"` or
    /// `"driver=sdrplay,serial=12345"`. An empty string falls back to
    /// [`DEFAULT_DRIVER_ARGS`]. `gain_db = None` enables hardware AGC when
    /// the driver supports it; otherwise hardware defaults apply.
    pub fn open(
        args: &str,
        center_freq_hz: f32,
        sample_rate_hz: f32,
        gain_db: Option<f32>,
    ) -> Result<Self, SourceError> {
        let args = if args.trim().is_empty() {
            DEFAULT_DRIVER_ARGS
        } else {
            args
        };

        let dev = Device::new(args).map_err(|e| {
            SourceError::Decode(format!("open SoapySDR device {args:?}: {e}"))
        })?;

        dev.set_sample_rate(Direction::Rx, 0, f64::from(sample_rate_hz))
            .map_err(|e| SourceError::Decode(format!("set sample rate: {e}")))?;
        dev.set_frequency(Direction::Rx, 0, f64::from(center_freq_hz), "")
            .map_err(|e| SourceError::Decode(format!("set frequency: {e}")))?;

        match gain_db {
            Some(g) => {
                if let Ok(true) = dev.has_gain_mode(Direction::Rx, 0) {
                    let _ = dev.set_gain_mode(Direction::Rx, 0, false);
                }
                dev.set_gain(Direction::Rx, 0, f64::from(g))
                    .map_err(|e| SourceError::Decode(format!("set gain: {e}")))?;
            }
            None => {
                if let Ok(true) = dev.has_gain_mode(Direction::Rx, 0) {
                    let _ = dev.set_gain_mode(Direction::Rx, 0, true);
                }
            }
        }

        let mut rx = dev
            .rx_stream::<Complex32>(&[0])
            .map_err(|e| SourceError::Decode(format!("create rx stream: {e}")))?;
        rx.activate(None)
            .map_err(|e| SourceError::Decode(format!("activate rx stream: {e}")))?;

        let (tx, rx_chan) = sync_channel::<Vec<Complex32>>(QUEUE_CHUNKS);
        let overruns = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));

        let overruns_w = Arc::clone(&overruns);
        let stop_w = Arc::clone(&stop);
        let worker = thread::Builder::new()
            .name("cwdit-soapy".into())
            .spawn(move || {
                let mut buf = vec![Complex32::new(0.0, 0.0); RX_BUF_SAMPLES];
                while !stop_w.load(Ordering::Relaxed) {
                    let mut bufs: [&mut [Complex32]; 1] = [&mut buf[..]];
                    let n = match rx.read(&mut bufs, READ_TIMEOUT_US) {
                        Ok(n) => n,
                        Err(e) if e.code == ErrorCode::Timeout => continue,
                        Err(e) => {
                            eprintln!("cwdit-source: soapy rx error: {e}");
                            break;
                        }
                    };
                    if n == 0 {
                        continue;
                    }
                    let chunk: Vec<Complex32> = buf[..n].to_vec();
                    match tx.try_send(chunk) {
                        Ok(()) => {}
                        Err(TrySendError::Full(dropped)) => {
                            overruns_w.fetch_add(dropped.len(), Ordering::Relaxed);
                        }
                        Err(TrySendError::Disconnected(_)) => break,
                    }
                }
                let _ = rx.deactivate(None);
                drop(rx);
                drop(dev);
            })
            .map_err(|e| SourceError::Decode(format!("spawn rx worker: {e}")))?;

        Ok(Self {
            receiver: rx_chan,
            sample_rate: sample_rate_hz,
            overruns,
            stop,
            worker: Some(worker),
            leftover: Vec::new(),
            leftover_idx: 0,
        })
    }

    /// Number of samples the radio thread has dropped because the consumer
    /// fell behind. Monotonically increasing for the life of the source.
    #[must_use]
    pub fn overruns(&self) -> usize {
        self.overruns.load(Ordering::Relaxed)
    }
}

impl Drop for SoapySource {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.worker.take() {
            let _ = h.join();
        }
    }
}

impl Source for SoapySource {
    type Sample = Complex32;

    fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    fn read(&mut self, buf: &mut [Complex32]) -> Result<usize, SourceError> {
        if buf.is_empty() {
            return Ok(0);
        }

        let mut written = 0;
        while written < buf.len() {
            if self.leftover_idx < self.leftover.len() {
                let avail = self.leftover.len() - self.leftover_idx;
                let want = (buf.len() - written).min(avail);
                buf[written..written + want].copy_from_slice(
                    &self.leftover[self.leftover_idx..self.leftover_idx + want],
                );
                written += want;
                self.leftover_idx += want;
                continue;
            }

            // Need a fresh chunk. Block on the first one so callers always
            // see at least one sample before returning; subsequent chunks
            // are best-effort to avoid stalling the loop on a partial fill.
            let chunk = if written == 0 {
                match self.receiver.recv() {
                    Ok(c) => c,
                    Err(_) => return Ok(0),
                }
            } else {
                match self.receiver.try_recv() {
                    Ok(c) => c,
                    Err(_) => break,
                }
            };
            self.leftover = chunk;
            self.leftover_idx = 0;
        }
        Ok(written)
    }
}
