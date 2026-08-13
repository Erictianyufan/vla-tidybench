"""Synchronous OpenPI-compatible policy client for the Isaac process."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import websockets.sync.client

from . import wire_protocol


class PolicyClient:
    """One persistent WebSocket connection to a policy service."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8000, timeout_s: float = 5.0) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        uri = host if host.startswith("ws://") or host.startswith("wss://") else f"ws://{host}:{port}"
        self._connection = websockets.sync.client.connect(
            uri, compression=None, max_size=None, open_timeout=timeout_s, close_timeout=timeout_s
        )
        metadata = self._connection.recv(timeout=timeout_s)
        if isinstance(metadata, str):
            raise RuntimeError(f"Policy server returned text during handshake: {metadata}")
        self.metadata: dict[str, Any] = wire_protocol.unpackb(metadata)
        self.timeout_s = timeout_s

    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        self._connection.send(wire_protocol.packb(dict(observation)))
        response = self._connection.recv(timeout=self.timeout_s)
        if isinstance(response, str):
            raise RuntimeError(f"Policy server inference failed:\n{response}")
        result = wire_protocol.unpackb(response)
        if not isinstance(result, dict):
            raise RuntimeError(f"Expected response dict, got {type(result).__name__}")
        return result

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "PolicyClient":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()

