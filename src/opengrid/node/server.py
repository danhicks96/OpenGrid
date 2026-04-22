"""
Worker WebSocket server — listens for incoming work packets from the coordinator,
dispatches them to Worker.handle_packet(), and returns results.

This is the missing piece that makes worker nodes reachable on the network.
Without this, workers exist but nobody can talk to them.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict
from typing import Optional

import websockets
from websockets.asyncio.server import serve, ServerConnection

from opengrid.node.worker import Worker, WorkPacket, WorkResult

log = logging.getLogger(__name__)

# Protocol constants
HEARTBEAT_INTERVAL = 10.0   # seconds between ping/pong
HEARTBEAT_TIMEOUT = 5.0     # seconds to wait for pong before declaring dead
MAX_MESSAGE_SIZE = 256 * 1024 * 1024  # 256 MB (large activation tensors)


class WorkerServer:
    """
    Async WebSocket server that accepts work packets from coordinators.

    Lifecycle:
        server = WorkerServer(worker, host, port)
        await server.start()
        ...
        await server.stop()
    """

    def __init__(self, worker: Worker, host: str = "0.0.0.0", port: int = 7600):
        self._worker = worker
        self._host = host
        self._port = port
        self._server = None
        self._active_connections: set[ServerConnection] = set()
        self._jobs_processed = 0
        self._nonces_seen: set[str] = set()  # replay protection
        self._nonce_max = 10_000

    async def start(self) -> None:
        self._server = await serve(
            self._handle_connection,
            self._host,
            self._port,
            max_size=MAX_MESSAGE_SIZE,
            ping_interval=HEARTBEAT_INTERVAL,
            ping_timeout=HEARTBEAT_TIMEOUT,
        )
        log.info("Worker WebSocket server listening on ws://%s:%d", self._host, self._port)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            log.info("Worker server stopped.")

    async def _handle_connection(self, ws: ServerConnection) -> None:
        remote = ws.remote_address
        log.info("New connection from %s", remote)
        self._active_connections.add(ws)
        try:
            async for raw_msg in ws:
                if isinstance(raw_msg, bytes):
                    raw_msg = raw_msg.decode()
                try:
                    result = await self._dispatch(raw_msg, remote)
                    await ws.send(json.dumps(asdict(result)))
                except Exception as e:
                    log.exception("Error dispatching message from %s", remote)
                    error_result = WorkResult(
                        job_id="unknown", sequence_id=0, shard_range=[0, 0],
                        output_activations_b64="", toploc_proof="",
                        latency_ms=0.0, error=str(e),
                    )
                    await ws.send(json.dumps(asdict(error_result)))
        except websockets.exceptions.ConnectionClosed:
            log.debug("Connection from %s closed", remote)
        finally:
            self._active_connections.discard(ws)

    async def _dispatch(self, raw_msg: str, remote) -> WorkResult:
        data = json.loads(raw_msg)

        # Replay protection: reject duplicate job_ids
        job_id = data.get("job_id", "")
        nonce_key = f"{job_id}:{data.get('sequence_id', 0)}"
        if nonce_key in self._nonces_seen:
            log.warning("Replay rejected: %s from %s", nonce_key, remote)
            return WorkResult(
                job_id=job_id, sequence_id=data.get("sequence_id", 0),
                shard_range=data.get("shard_range", [0, 0]),
                output_activations_b64="", toploc_proof="",
                latency_ms=0.0, error="replay_rejected",
            )
        self._nonces_seen.add(nonce_key)
        # Evict old nonces to prevent unbounded growth
        if len(self._nonces_seen) > self._nonce_max:
            # Remove oldest half (set doesn't preserve order, so just clear)
            self._nonces_seen.clear()

        packet = WorkPacket(
            packet_type=data.get("packet_type", "inference_forward"),
            job_id=job_id,
            sequence_id=data.get("sequence_id", 0),
            model_id=data.get("model_id", ""),
            shard_range=data.get("shard_range", [0, 0]),
            input_activations_b64=data.get("input_activations_b64", ""),
            kv_cache_token=data.get("kv_cache_token", ""),
            return_address=data.get("return_address", ""),
            deadline_ms=data.get("deadline_ms", 2000),
        )

        # Enforce deadline as a timeout
        try:
            result = await asyncio.wait_for(
                self._worker.handle_packet(packet),
                timeout=packet.deadline_ms / 1000.0,
            )
        except asyncio.TimeoutError:
            log.warning("Job %s timed out after %d ms", packet.job_id, packet.deadline_ms)
            result = WorkResult(
                job_id=packet.job_id, sequence_id=packet.sequence_id,
                shard_range=packet.shard_range,
                output_activations_b64="", toploc_proof="",
                latency_ms=float(packet.deadline_ms), error="deadline_exceeded",
            )

        self._jobs_processed += 1
        return result

    @property
    def connection_count(self) -> int:
        return len(self._active_connections)

    @property
    def jobs_processed(self) -> int:
        return self._jobs_processed
