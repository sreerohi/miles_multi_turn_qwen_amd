"""
Custom agent function for ``agentic_tool_call.generate``.

Dispatches to a Harbor-based agent server and returns env metadata
as a plain dict. The generate layer merges this into sample.metadata so
downstream reward models (--custom-rm-path) can extract reward, eval
reports, etc.

Task-type agnostic — the server + Harbor task directory handle all
differentiation (environment, grading harness, agent selection).
"""

import asyncio
import json
import logging
import os
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlsplit, urlunparse

import httpx

from miles.utils.http_utils import post

logger = logging.getLogger(__name__)

# Backstop for an unreachable agent server; its own --agent-timeout should fire first.
_DEFAULT_AGENT_TRIAL_TIMEOUT_S = 7200

_agent_server_client: httpx.AsyncClient | None = None


def _agent_trial_timeout_s() -> int:
    """Per-trial ceiling for the agent-server call, overridable via AGENT_TRIAL_TIMEOUT."""
    return int(os.environ.get("AGENT_TRIAL_TIMEOUT", _DEFAULT_AGENT_TRIAL_TIMEOUT_S))


def _get_agent_server_client() -> httpx.AsyncClient:
    """Return a client whose long-running requests survive idle network paths."""
    global _agent_server_client
    if _agent_server_client is None:
        socket_options = [
            (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
            (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPIDLE", 4), 60),
            (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPINTVL", 5), 30),
            (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPCNT", 6), 5),
        ]
        transport = httpx.AsyncHTTPTransport(socket_options=socket_options)
        _agent_server_client = httpx.AsyncClient(
            transport=transport,
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
            timeout=None,
        )
    return _agent_server_client


async def _post_agent_server(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    client = _get_agent_server_client()
    response = await client.post(url, json=payload)
    response.raise_for_status()
    return response.json()


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | None:
    """Run a single task instance via the Harbor agent server."""
    metadata = metadata or {}
    request_kwargs = request_kwargs or {}

    agent_server_url = os.getenv(
        "AGENT_SERVER_URL",
        os.getenv("SWE_AGENT_URL", "http://localhost:11000"),
    )
    model_name = os.getenv(
        "AGENT_MODEL_NAME",
        os.getenv("SWE_AGENT_MODEL_NAME", "model"),
    )

    session_url = f"{base_url}/v1"
    external_host = os.getenv("MILES_ROUTER_EXTERNAL_HOST")
    if external_host:
        parsed = urlparse(session_url)
        port = parsed.port
        netloc = f"{external_host}:{port}" if port else external_host
        session_url = urlunparse(parsed._replace(netloc=netloc))

    request: dict[str, Any] = {
        **metadata,
        "base_url": session_url,
        "model": f"openai/{model_name}",
        "sampling_params": request_kwargs,
    }

    max_seq_len = metadata.get("max_seq_len")
    if max_seq_len is not None:
        request["max_seq_len"] = int(max_seq_len)

    # Tag each /run request with the current rollout_id (set by generate.py per rollout)
    # so build_timeline_data.py can assign trials to steps exactly instead of by wall-clock.
    rollout_id = os.getenv("MILES_ROLLOUT_ID")
    if rollout_id is not None:
        request["rollout_id"] = int(rollout_id)

    session_server_id = metadata.get("session_server_id")
    if session_server_id is not None:
        if external_host:
            port = urlsplit(f"http://{session_server_id}").port
            session_server_id = f"{external_host}:{port}"
        request["session_server_id"] = session_server_id

    session_server_instance_id = metadata.get("session_server_instance_id")
    if session_server_instance_id is not None:
        request["session_server_instance_id"] = session_server_instance_id

    trial_timeout_s = _agent_trial_timeout_s()
    try:
        response = await asyncio.wait_for(
            _post_agent_server(f"{agent_server_url}/run", request),
            timeout=trial_timeout_s,
        )
    except asyncio.TimeoutError:
        logger.error(f"Agent server call timed out after {trial_timeout_s}s")
        return None
    except asyncio.CancelledError:
        logger.warning("Agent server call cancelled (sibling task failure?)")
        return None
    except Exception as e:
        logger.error(f"Agent server call failed: {e}")
        return None

    result = {
        "reward": response.get("reward", 0.0),
        "exit_status": response.get("exit_status", ""),
        "eval_report": response.get("eval_report", {}),
        "agent_metrics": response.get("agent_metrics", {}),
    }

    # Write miles_rollout_id.json sidecar into the trial dir so build_timeline_data.py
    # can map each trial to its exact rollout step and session_id for GPU gen time.
    trial_uri = response.get("trial_uri") or response.get("trial_name", "")
    trials_dir = os.getenv("MILES_TRIALS_DIR", "")
    if trial_uri and trials_dir and rollout_id is not None:
        sidecar_path = Path(trials_dir) / trial_uri / "miles_rollout_id.json"
        try:
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            sidecar_path.write_text(json.dumps({
                "rollout_id": int(rollout_id),
                "session_id": response.get("session_id"),
            }))
        except Exception as e:
            logger.debug(f"Could not write rollout sidecar {sidecar_path}: {e}")

    return result


async def abort(args) -> None:
    """Teardown hook for oversampling abort (called by sglang_rollout.abort).

    When Miles has enough samples and aborts SGLang, the in-flight Harbor trials
    keep looping and hitting SGLang until they hit their own max_seq_len/timeout.
    Flush the agent server so it cancels those ``/run`` tasks and releases their
    containers. No-op unless AGENT_SERVER_URL and session_server_instance_id are
    available.
    """
    agent_server_url = os.getenv("AGENT_SERVER_URL", os.getenv("SWE_AGENT_URL"))
    instance_id = getattr(args, "session_server_instance_id", None)
    if not agent_server_url or not instance_id:
        return

    headers = None
    admin_secret = os.getenv("HARBOR_ADMIN_SECRET")
    if admin_secret:
        headers = {"Authorization": f"Bearer {admin_secret}"}

    try:
        result = await post(
            f"{agent_server_url.rstrip('/')}/flush",
            {"session_server_instance_id": instance_id},
            max_retries=3,
            headers=headers,
        )
        logger.info(f"Flushed agent server {agent_server_url}: {result}")
    except Exception as e:
        logger.warning(f"Failed to flush agent server {agent_server_url}: {e}")

    # Force-close the shared httpx client so any pending /run POSTs get an
    # immediate connection error instead of waiting for TCP keepalive (~210s)
    # or the 7200s asyncio.wait_for backstop to fire.
    global _agent_server_client
    if _agent_server_client is not None:
        try:
            await _agent_server_client.aclose()
        except Exception:
            pass
        _agent_server_client = None
