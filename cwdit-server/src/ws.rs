//! WebSocket connection handler. Spawns a per-connection pipeline and
//! forwards its events to the client as JSON text frames.

use axum::extract::ws::{Message, WebSocket};
use tokio::sync::mpsc;

use crate::AppState;
use crate::pipeline::{self, Event};

/// Handle one upgraded WebSocket. Runs the decode pipeline (which emits
/// its own `Session` event first) and streams events until the stream
/// finishes or the client disconnects.
pub async fn handle(mut socket: WebSocket, state: AppState) {
    let (tx, mut rx) = mpsc::channel::<Event>(256);
    tokio::spawn(pipeline::pump(
        state.input.clone(),
        state.samples.clone(),
        state.sample_rate,
        state.cfg.clone(),
        state.pace_factor,
        tx,
    ));

    while let Some(ev) = rx.recv().await {
        if !send_event(&mut socket, &ev).await {
            break;
        }
    }

    let _ = socket.close().await;
}

async fn send_event(socket: &mut WebSocket, ev: &Event) -> bool {
    let Ok(json) = serde_json::to_string(ev) else {
        return false;
    };
    socket.send(Message::Text(json)).await.is_ok()
}
