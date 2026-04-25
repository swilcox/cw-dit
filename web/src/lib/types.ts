// Event schema mirrored from cwdit-server/src/pipeline.rs::Event.
// Serde emits the tag under `type` with snake_case variants.

export type SessionMode = 'fixed' | 'scan';
export type ScanState = 'calibrating' | 'ready';

export interface SessionEvent {
	type: 'session';
	input: string;
	sample_rate: number;
	mode: SessionMode;
}

export interface ScanStatusEvent {
	type: 'scan_status';
	state: ScanState;
	detected?: number;
}

export interface ChannelOpenEvent {
	type: 'channel_open';
	id: number;
	freq_hz: number;
	wpm: number;
}

export interface CharEvent {
	type: 'char';
	channel: number;
	ch: string;
}

export interface WordBreakEvent {
	type: 'word_break';
	channel: number;
}

export interface UnknownEvent {
	type: 'unknown';
	channel: number;
}

export interface WpmEvent {
	type: 'wpm';
	channel: number;
	wpm: number;
}

export interface DoneEvent {
	type: 'done';
}

export interface SpectrumEvent {
	type: 'spectrum';
	/** base64 of a Uint8Array — one byte per FFT bin (DC … Nyquist). */
	bins: string;
	/** Centre frequency of the first bin (Hz). */
	f_min: number;
	/** Centre frequency of the last bin (Hz). */
	f_max: number;
	/** dB value mapped to byte 0. */
	db_floor: number;
	/** dB value mapped to byte 255. */
	db_ceiling: number;
}

export type DecodeEvent =
	| SessionEvent
	| ScanStatusEvent
	| ChannelOpenEvent
	| CharEvent
	| WordBreakEvent
	| UnknownEvent
	| WpmEvent
	| SpectrumEvent
	| DoneEvent;

// The text shown for one channel is a stream of tokens: regular characters,
// word-break spaces, and unknown-symbol markers. Kept as discrete tokens so
// the unknown marker can be styled without reparsing the string.
export type Token =
	| { kind: 'char'; value: string }
	| { kind: 'space' }
	| { kind: 'unknown' };
