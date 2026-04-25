<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import Channel from '$lib/Channel.svelte';
	import Waterfall from '$lib/Waterfall.svelte';
	import type { DecodeEvent, SessionMode, Token } from '$lib/types';

	interface ChannelState {
		freqHz: number;
		wpm: number;
		tokens: Token[];
	}

	let connectionStatus = $state<'connecting' | 'connected' | 'disconnected' | 'error' | 'done'>(
		'connecting'
	);
	let sessionInput = $state<string | null>(null);
	let sampleRate = $state<number | null>(null);
	let sessionMode = $state<SessionMode | null>(null);
	let scanNote = $state<string | null>(null);
	let channels = $state<Record<number, ChannelState>>({});
	let done = $state(false);
	let spectrumFrame = $state<Uint8Array | null>(null);
	let spectrumFMin = $state(0);
	let spectrumFMax = $state(0);
	let ws: WebSocket | null = null;

	const channelList = $derived(
		Object.entries(channels)
			.map(([id, c]) => ({ id: Number(id), ...c }))
			.sort((a, b) => a.id - b.id)
	);

	onMount(() => {
		const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
		ws = new WebSocket(`${proto}//${location.host}/ws`);
		ws.addEventListener('open', () => {
			connectionStatus = 'connected';
		});
		ws.addEventListener('close', () => {
			if (!done) connectionStatus = 'disconnected';
		});
		ws.addEventListener('error', () => {
			connectionStatus = 'error';
		});
		ws.addEventListener('message', handleMessage);
	});

	onDestroy(() => ws?.close());

	function handleMessage(msg: MessageEvent<string>) {
		let ev: DecodeEvent;
		try {
			ev = JSON.parse(msg.data) as DecodeEvent;
		} catch {
			return;
		}
		switch (ev.type) {
			case 'session':
				sessionInput = ev.input;
				sampleRate = ev.sample_rate;
				sessionMode = ev.mode;
				break;
			case 'scan_status':
				if (ev.state === 'calibrating') {
					scanNote = 'scanning for signals…';
				} else {
					const n = ev.detected ?? 0;
					scanNote = n === 0 ? 'scan found no signals' : `scan found ${n} signal${n === 1 ? '' : 's'}`;
				}
				break;
			case 'channel_open':
				channels[ev.id] = { freqHz: ev.freq_hz, wpm: ev.wpm, tokens: [] };
				break;
			case 'char':
				channels[ev.channel]?.tokens.push({ kind: 'char', value: ev.ch });
				break;
			case 'word_break':
				channels[ev.channel]?.tokens.push({ kind: 'space' });
				break;
			case 'unknown':
				channels[ev.channel]?.tokens.push({ kind: 'unknown' });
				break;
			case 'wpm': {
				const c = channels[ev.channel];
				if (c) c.wpm = ev.wpm;
				break;
			}
			case 'spectrum':
				spectrumFrame = base64ToBytes(ev.bins);
				spectrumFMin = ev.f_min;
				spectrumFMax = ev.f_max;
				break;
			case 'done':
				done = true;
				connectionStatus = 'done';
				break;
		}
	}

	function base64ToBytes(s: string): Uint8Array {
		const bin = atob(s);
		const out = new Uint8Array(bin.length);
		for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
		return out;
	}
</script>

<svelte:head>
	<title>cw-dit</title>
</svelte:head>

<main>
	<h1>cw-dit</h1>
	<div class="status">{scanNote ?? connectionStatus}</div>
	{#if sessionInput}
		<div class="meta">
			{sessionInput} · {sampleRate} Hz · mode: {sessionMode}
		</div>
	{/if}
	{#if spectrumFrame}
		<div class="waterfall-wrap">
			<Waterfall frame={spectrumFrame} fMin={spectrumFMin} fMax={spectrumFMax} />
		</div>
	{/if}
	<div class="channels">
		{#each channelList as ch (ch.id)}
			<Channel id={ch.id} freqHz={ch.freqHz} wpm={ch.wpm} tokens={ch.tokens} {done} />
		{/each}
		{#if done && channelList.length === 0}
			<div class="empty">no channels</div>
		{/if}
	</div>
</main>

<style>
	:global(:root) {
		color-scheme: dark;
		--bg: #0c0d10;
		--fg: #e6e6e6;
		--mute: #7a828c;
		--accent: #4ea1ff;
		--warn: #f88;
		--panel: #141519;
		--panel-border: #1e2128;
	}
	:global(body) {
		font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
		background: var(--bg);
		color: var(--fg);
		margin: 0;
	}
	main {
		padding: 2rem;
		max-width: 64rem;
		margin-inline: auto;
	}
	h1 {
		margin: 0 0 0.25rem;
		font-weight: 500;
	}
	.status {
		color: var(--mute);
		font-size: 0.9rem;
		margin-bottom: 0.5rem;
	}
	.meta {
		color: var(--mute);
		font-size: 0.85rem;
		font-family: ui-monospace, 'SF Mono', Menlo, monospace;
		margin-bottom: 1rem;
	}
	.channels {
		display: grid;
		gap: 0.75rem;
	}
	.waterfall-wrap {
		margin-bottom: 1rem;
	}
	.empty {
		color: var(--mute);
		font-style: italic;
		padding: 1rem;
	}
</style>
