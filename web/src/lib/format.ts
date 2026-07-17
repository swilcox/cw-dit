// Frequency formatting shared by the waterfall axis and channel labels.
// Audio-path frequencies are a few kHz; SDR-path frequencies are absolute
// RF Hz (e.g. 7.0355 MHz), so pick the unit from the magnitude.

/** Format a frequency with a magnitude-appropriate unit. */
export function fmtFreq(hz: number): string {
	if (Math.abs(hz) >= 1_000_000) return `${(hz / 1_000_000).toFixed(4)} MHz`;
	if (Math.abs(hz) >= 10_000) return `${(hz / 1_000).toFixed(1)} kHz`;
	return `${hz.toFixed(1)} Hz`;
}

/** Compact variant for waterfall marker tags (no unit-word padding). */
export function fmtFreqShort(hz: number): string {
	if (Math.abs(hz) >= 1_000_000) return `${(hz / 1_000_000).toFixed(4)}M`;
	if (Math.abs(hz) >= 10_000) return `${(hz / 1_000).toFixed(1)}k`;
	return hz.toFixed(0);
}
