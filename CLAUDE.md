# cw-dit — collaboration notes

Rust 2024 Cargo workspace; six crates (see README).

## Before saying work is done

- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`

Both must be clean. The workspace enables `clippy::pedantic`; the four `cast_*` lints are allowed workspace-wide because DSP code is full of `as` conversions. `unsafe_code` is forbidden.

## Test fixtures

Use `cwdit-synth` (as a dev-dependency) to generate CW WAVs in tests. Do not re-introduce inline WAV synthesis — that duplication was consolidated onto `cwdit-synth` on purpose.
