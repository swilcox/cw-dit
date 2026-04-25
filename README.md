# cw-dit

Cross-platform multi-channel CW / Morse decoder in Rust.

## Workspace

- **cwdit-morse** — streaming Morse decoder with adaptive timing.
- **cwdit-dsp** — Goertzel bank, hysteretic slicer, run-length encoder.
- **cwdit-source** — `Source` trait and a mono PCM WAV reader.
- **cwdit-cli** — `cwdit` command-line decoder (single or multi-channel).
- **cwdit-server** — Axum + WebSocket back-end for the SvelteKit web UI in `web/`.
- **cwdit-synth** — CW audio synthesiser (library + `cwdit-synth` binary) for generating fixtures and demos.

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

Auto-detect every CW signal in the passband — no `--channels` list needed.
The first few seconds of audio are analysed for occupied bins (keyed peaks
above the noise floor, with sidelobe / keying-sideband suppression), then
every detected signal gets its own decoder:

```sh
cargo run -p cwdit-cli -- /tmp/recording.wav --fft --scan --wpm 25
```

Live audio from the default system input (e.g. feed a receiver's audio output
into the soundcard):

```sh
cargo run -p cwdit-cli -- --live --tone 700 --wpm 18
```

Pick a specific input device with `--device "Name"`.

## Development

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

## License

Dual-licensed under MIT or Apache-2.0 at your option.
