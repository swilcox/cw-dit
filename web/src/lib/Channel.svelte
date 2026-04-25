<script lang="ts">
	import type { Token } from './types';

	interface Props {
		id: number;
		freqHz: number;
		wpm: number;
		tokens: Token[];
		done: boolean;
	}

	let { id, freqHz, wpm, tokens, done }: Props = $props();
</script>

<div class="channel" class:done>
	<header>
		<span class="label">ch {id} · {freqHz.toFixed(1)} Hz</span>
		<span class="wpm">{wpm.toFixed(1)} WPM</span>
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
		padding: 0.75rem 1rem 1rem;
	}
	.channel.done {
		border-color: var(--accent);
	}
	header {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		font-family: ui-monospace, 'SF Mono', Menlo, monospace;
		font-size: 0.85rem;
		color: var(--mute);
		margin-bottom: 0.5rem;
	}
	.label {
		color: var(--fg);
	}
	.text {
		font-family: ui-monospace, 'SF Mono', Menlo, monospace;
		font-size: 1.4rem;
		white-space: pre-wrap;
		word-break: break-word;
		background: #000;
		border-radius: 4px;
		padding: 0.6rem 0.8rem;
		min-height: 3rem;
	}
	.unknown {
		color: var(--warn);
	}
</style>
