# cw-dit

Cross-platform multi-channel CW / Morse decoder in Rust.

## Workspace

- **cwdit-morse** — streaming Morse decoder with adaptive timing.
- **cwdit-dsp** — Goertzel bank, envelope smoothing, noise-floor-tracking
  slicer with SNR squelch, run-length encoder, glitch debouncer.
- **cwdit-source** — `Source` trait, a mono PCM WAV reader, live audio via cpal, and (with `--features soapy`) live IQ via SoapySDR.
- **cwdit-cli** — `cwdit` command-line decoder (single or multi-channel).
- **cwdit-server** — Axum + WebSocket back-end for the SvelteKit web UI in `web/`.
- **cwdit-synth** — CW audio synthesiser (library + `cwdit-synth` binary) for generating fixtures and demos, with optional calibrated noise (`--noise-snr-db`).

## Quick start

Generate a WAV and decode it:

```sh
cargo run -p cwdit-synth -- -o /tmp/cq.wav -t "CQ DE W1AW" -f 700 -w 18
cargo run -p cwdit-cli   -- /tmp/cq.wav --tone 700 --wpm 18
```

Serve the web UI. The frontend is a SvelteKit SPA in `web/`; build it once,
then start the Rust server:

```sh
(cd web && npm install && npm run build)
cargo run -p cwdit-server -- /tmp/cq.wav -t 700 -w 18
```

Open `http://127.0.0.1:3000`. Pass `--web-dir path/to/build` if the built
assets live somewhere other than `web/build/`.

The server skims too: with `--scan` (instead of `-t`/`--channels`) it
re-detects stations every `--scan-duration` seconds and the UI shows
channels opening and closing as stations come and go, with the waterfall
cropped to the scanned band:

```sh
cargo run -p cwdit-server -- /tmp/recording.wav --scan --wpm 25
```

Live audio instead of a file — feed a receiver into the soundcard and
skim it from the browser. One capture is shared by every connected
client; each client decodes the stream from the moment it joins
(`--device "Name"` picks a specific input; `--pace-factor` is file-only):

```sh
cargo run -p cwdit-server -- --live --scan --wpm 25
```

For frontend hot reload run Vite separately — it proxies `/ws` to the Rust
server automatically:

```sh
cargo run -p cwdit-server -- /tmp/cq.wav -t 700 -w 18 &
(cd web && npm run dev)   # serves the UI at http://127.0.0.1:5173
```

Multi-channel:

```sh
cargo run -p cwdit-synth -- -o /tmp/two.wav \
  -c "CQ DE W1AW:18:600" -c "QRZ DE K5ABC:20:1400"
cargo run -p cwdit-cli   -- /tmp/two.wav --channels 600,1400 --wpm 18
```

FFT channelizer mode — one FFT drives every channel, so decoding many
simultaneous signals scales better than one Goertzel per tone. The FFT size
and hop are auto-selected from `--wpm` to cover slow CW through contest
speeds (40 WPM+); override with `--fft-size` / `--hop` if you want to pin
them:

```sh
cargo run -p cwdit-cli -- /tmp/two.wav --fft --channels 600,1400 --wpm 30
```

Skim the passband — no `--channels` list needed. Detection re-runs every
`--scan-duration` seconds (long-window FFT, local-noise-floor SNR gate,
sidelobe / keying-sideband suppression): a decode channel opens when a
station keys up and closes after `--channel-timeout` (default 30 s) of
silence, so stations that come and go over the recording are all caught:

```sh
cargo run -p cwdit-cli -- /tmp/recording.wav --scan --wpm 25
```

Adding `--fft` keeps the older one-shot flow — calibrate on the opening
interval, then decode that fixed channel list via the FFT channelizer.

Live audio from the default system input (e.g. feed a receiver's audio output
into the soundcard):

```sh
cargo run -p cwdit-cli -- --live --tone 700 --wpm 18
```

Pick a specific input device with `--device "Name"`.

## SDR (live IQ via SoapySDR)

Skim every CW signal across the radio's full sampled bandwidth. Like the
audio path, `--sdr --scan` skims *continuously*: detection re-runs every
`--scan-duration` seconds on an RF bin grid and decode channels open and
close as stations come and go (add `--fft` for the older one-shot
calibrate-then-decode flow). Built behind the `soapy` cargo feature so the
SoapySDR linkage is opt-in.

Install the host bits once (macOS examples — pick the driver modules that
match your hardware):

```sh
brew install soapysdr soapyrtlsdr soapysdrplay3
SoapySDRUtil --probe="driver=sdrplay"   # smoke-test once each radio is plugged in
SoapySDRUtil --probe="driver=rtlsdr"
```

`soapysdrplay3` requires the closed-source SDRplay API service to be running
for the SDRplay to enumerate; install it from sdrplay.com if `--probe` can't
find the device.

Scan a CW segment of 40 m on an SDRplay (default driver):

```sh
cargo run -p cwdit-cli --features soapy -- \
    --sdr --freq 7035000 --rf-rate 2000000 --scan --wpm 25
```

RTL-SDR with explicit driver args and manual gain:

```sh
cargo run -p cwdit-cli --features soapy -- \
    --sdr "driver=rtlsdr" --freq 7040000 --rf-rate 1024000 \
    --rf-gain 30 --scan --wpm 25
```

Decode a fixed list of RF tones instead of scanning:

```sh
cargo run -p cwdit-cli --features soapy -- \
    --sdr --freq 7035000 --rf-rate 2000000 \
    --channels 7035500,7038200,7041000 --wpm 22
```

Defaults: `--sdr` alone uses `driver=sdrplay`; `--rf-rate` defaults to
1.024 Msps; `--rf-gain` is omitted to enable hardware AGC. Scan covers the
whole sampled passband minus a 5 % guard at each edge unless
`--scan-min-freq` / `--scan-max-freq` (in absolute RF Hz) override.

## Development

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

## License

Dual-licensed under MIT or Apache-2.0 at your option.
