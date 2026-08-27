"""Anthropic Messages API request preparation and response rendering.

Handles the wire format for the session server's native Anthropic endpoint
(``POST /sessions/{id}/v1/messages``).  No translation to/from OpenAI format
— requests are forwarded verbatim to SGLang's own ``/v1/messages`` endpoint,
and responses are returned in Anthropic format.

The only modifications made to the request before forwarding:
  - ``stream`` is popped (backend call is always non-streaming; the session
    server fakes the SSE stream so it can store the full response as a record
    before replying).

Sample assembly from Anthropic-format records lives in
``samples/anthropic_merge.py``.
"""

import json

from starlette.responses import Response


def prepare_anthropic_request(body: dict) -> tuple[dict, bool]:
    """Strip ``stream`` from an Anthropic Messages API body.

    Returns ``(body_without_stream, client_stream)``.  The backend call must
    stay non-streaming so the session server can store the complete response
    as a ``SessionRecord`` before replying to the client.
    """
    body = dict(body)
    client_stream = bool(body.pop("stream", False))
    return body, client_stream


def _render_json(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _anthropic_sse(response: dict) -> bytes:
    """Render a complete Anthropic response dict as a minimal SSE stream.

    Emits the minimal event sequence that Anthropic-SDK clients expect:
    ``message_start`` → per-block ``content_block_*`` triplets →
    ``message_delta`` → ``message_stop``.  All content is delivered in a
    single delta per block (fake streaming — same strategy as the OpenAI path).
    """
    lines: list[bytes] = []

    def event(name: str, data: dict) -> None:
        lines.append(f"event: {name}\n".encode())
        lines.append(b"data: " + _render_json(data) + b"\n\n")

    content_blocks = response.get("content", [])

    event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": response.get("id", ""),
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": response.get("model", ""),
                "stop_reason": None,
                "stop_sequence": None,
                "usage": response.get("usage", {}),
            },
        },
    )

    for idx, block in enumerate(content_blocks):
        btype = block.get("type", "text")
        event("content_block_start", {"type": "content_block_start", "index": idx, "content_block": block})

        if btype == "thinking":
            event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {
                        "type": "thinking_delta",
                        "thinking": block.get("thinking", ""),
                    },
                },
            )
            event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {
                        "type": "signature_delta",
                        "signature": block.get("signature", ""),
                    },
                },
            )
        elif btype == "text":
            event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "text_delta", "text": block.get("text", "")},
                },
            )
        elif btype == "tool_use":
            event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(block.get("input", {}), separators=(",", ":")),
                    },
                },
            )

        event("content_block_stop", {"type": "content_block_stop", "index": idx})

    event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": response.get("stop_reason"),
                "stop_sequence": response.get("stop_sequence"),
            },
            "usage": {"output_tokens": (response.get("usage") or {}).get("output_tokens", 0)},
        },
    )
    event("message_stop", {"type": "message_stop"})

    return b"".join(lines)


def render_anthropic_response(response: dict, status_code: int, client_stream: bool) -> Response:
    """Build a Starlette ``Response`` for an Anthropic Messages API reply.

    ``response`` is the parsed JSON dict returned by SGLang's ``/v1/messages``
    endpoint.  ``client_stream`` controls whether to emit Anthropic SSE events
    or a plain JSON body.
    """
    if client_stream:
        return Response(
            content=_anthropic_sse(response),
            status_code=status_code,
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
            media_type="text/event-stream",
        )
    return Response(
        content=_render_json(response),
        status_code=status_code,
        media_type="application/json",
    )
