<script lang="ts">
	import { onMount } from 'svelte';

	interface Props {
		frame: Uint8Array | null;
		fMin: number;
		fMax: number;
	}

	let { frame, fMin, fMax }: Props = $props();

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
	<canvas bind:this={canvas}></canvas>
	<div class="axis">
		<span>{fMin.toFixed(0)} Hz</span>
		<span>{fMax.toFixed(0)} Hz</span>
	</div>
</div>

<style>
	.waterfall {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
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
	.axis {
		display: flex;
		justify-content: space-between;
		font-family: ui-monospace, 'SF Mono', Menlo, monospace;
		font-size: 0.75rem;
		color: var(--mute);
	}
</style>
