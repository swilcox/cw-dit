<script lang="ts">
	import { onMount } from 'svelte';
	import { fmtFreq, fmtFreqShort } from './format';

	interface Marker {
		id: number;
		freqHz: number;
	}

	interface Props {
		frame: Uint8Array | null;
		fMin: number;
		fMax: number;
		/** Live decode channels, drawn as ticks over the display. */
		markers?: Marker[];
	}

	let { frame, fMin, fMax, markers = [] }: Props = $props();

	// Horizontal position of a frequency as a 0..100% offset.
	function toPercent(freqHz: number): number {
		const span = fMax - fMin;
		if (span <= 0) return 0;
		return Math.min(100, Math.max(0, ((freqHz - fMin) / span) * 100));
	}

	const HEIGHT = 256;
	let canvas: HTMLCanvasElement;
	let ctx: CanvasRenderingContext2D | null = null;
	let canvasWidth = $state(0);

	onMount(() => {
		ctx = canvas.getContext('2d');
	});

	$effect(() => {
		if (!frame || !ctx) return;
		if (canvasWidth !== frame.length) {
			canvas.width = frame.length;
			canvas.height = HEIGHT;
			ctx.fillStyle = '#000';
			ctx.fillRect(0, 0, canvas.width, canvas.height);
			canvasWidth = frame.length;
		}
		// Scroll everything down one row, then write the new row at the top.
		ctx.drawImage(canvas, 0, 1);
		const row = ctx.createImageData(canvas.width, 1);
		for (let i = 0; i < frame.length; i++) {
			const [r, g, b] = colormap(frame[i]);
			const o = i * 4;
			row.data[o] = r;
			row.data[o + 1] = g;
			row.data[o + 2] = b;
			row.data[o + 3] = 255;
		}
		ctx.putImageData(row, 0, 0);
	});

	// Three-stop gradient: deep blue → cyan → white. Reads well on a dark
	// background and gives keyed CW signals an obvious vertical streak.
	function colormap(v: number): [number, number, number] {
		const t = v / 255;
		if (t < 0.5) {
			const k = t / 0.5;
			return [0, Math.round(120 * k), Math.round(60 + 195 * k)];
		}
		const k = (t - 0.5) / 0.5;
		return [Math.round(255 * k), Math.round(120 + 135 * k), 255];
	}
</script>

<div class="waterfall">
	<div class="display">
		<canvas bind:this={canvas}></canvas>
		{#each markers as m (m.id)}
			<div class="marker" style:left="{toPercent(m.freqHz)}%">
				<span class="tag">ch {m.id} · {fmtFreqShort(m.freqHz)}</span>
			</div>
		{/each}
	</div>
	<div class="axis">
		<span>{fmtFreq(fMin)}</span>
		<span>{fmtFreq(fMax)}</span>
	</div>
</div>

<style>
	.waterfall {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}
	.display {
		position: relative;
	}
	canvas {
		width: 100%;
		height: 256px;
		image-rendering: pixelated;
		background: #000;
		border: 1px solid var(--panel-border);
		border-radius: 6px;
		display: block;
	}
	.marker {
		position: absolute;
		top: 0;
		bottom: 0;
		width: 0;
		border-left: 1px dashed color-mix(in srgb, var(--accent) 65%, transparent);
		pointer-events: none;
	}
	.tag {
		position: absolute;
		top: 2px;
		left: 3px;
		font-family: ui-monospace, 'SF Mono', Menlo, monospace;
		font-size: 0.65rem;
		color: var(--accent);
		background: rgba(0, 0, 0, 0.65);
		padding: 0 3px;
		border-radius: 3px;
		white-space: nowrap;
	}
	.axis {
		display: flex;
		justify-content: space-between;
		font-family: ui-monospace, 'SF Mono', Menlo, monospace;
		font-size: 0.75rem;
		color: var(--mute);
	}
</style>
