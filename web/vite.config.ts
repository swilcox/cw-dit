import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

// In dev (`npm run dev`), proxy the decode-event WebSocket through to a
// locally-running `cwdit-server` on :3000. Production builds serve the
// static assets directly from `cwdit-server` itself, so no proxy applies.
export default defineConfig({
	plugins: [sveltekit()],
	server: {
		proxy: {
			'/ws': {
				target: 'ws://127.0.0.1:3000',
				ws: true,
				changeOrigin: true
			}
		}
	}
});
