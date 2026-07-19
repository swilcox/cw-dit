<script lang="ts">
	import { fmtFreq } from './format';
	import type { Token } from './types';

	interface Props {
		id: number;
		freqHz: number;
		wpm: number;
		tokens: Token[];
		done: boolean;
		/** Retired by the skimmer; text stays, card dims. */
		closed?: boolean;
	}

	let { id, freqHz, wpm, tokens, done, closed = false }: Props = $props();
</script>

<div class="channel" class:done class:closed>
	<header>
		<span class="label">ch {id} · {fmtFreq(freqHz)}</span>
		<span class="wpm">{closed ? 'closed · ' : ''}{wpm.toFixed(1)} WPM</span>
	</header>
	<div class="text">
		{#each tokens as tok, i (i)}
			{#if tok.kind === 'char'}{tok.value}{:else if tok.kind === 'space'}&nbsp;{:else}<span
					class="unknown">?</span
				>{/if}
		{/each}
	</div>
</div>

<style>
	.channel {
		background: var(--panel);
		border: 1px solid var(--panel-border);
		border-radius: 6px;
		padding: 0.5rem 0.75rem 0.6rem;
	}
	.channel.done {
		border-color: var(--accent);
	}
	.channel.closed {
		opacity: 0.55;
	}
	header {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		font-family: ui-monospace, 'SF Mono', Menlo, monospace;
		font-size: 0.75rem;
		color: var(--mute);
		margin-bottom: 0.3rem;
	}
	.label {
		color: var(--fg);
	}
	/* Compact: many channels need to share the screen with the waterfall. */
	.text {
		font-family: ui-monospace, 'SF Mono', Menlo, monospace;
		font-size: 0.95rem;
		white-space: pre-wrap;
		word-break: break-word;
		background: #000;
		border-radius: 4px;
		padding: 0.35rem 0.55rem;
		min-height: 1.4rem;
	}
	.unknown {
		color: var(--warn);
	}
</style>
