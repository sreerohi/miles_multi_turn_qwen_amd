"""Training-sample assembly from Anthropic Messages API session records.

Parallel to ``merge.py`` (which handles OpenAI-format records).  Called by
``SessionCore.collect_samples`` when all records in a session carry
``path="/v1/messages"`` — i.e. the session used the Anthropic endpoint.

Current limitations (eval-only mode):
  - ``tokens`` and ``rollout_log_probs`` are left empty because SGLang's
    ``/v1/messages`` endpoint does not yet surface ``output_token_logprobs``
    in the Anthropic response format.  ``merge_samples`` requires cumulative
    token IDs across turns to stitch multi-turn trajectories, so multi-turn
    sessions return only the final-turn sample rather than a merged trajectory.
  - Single-turn sessions are fully supported: ``response``, ``response_length``
    (from ``usage.output_tokens``), ``loss_mask``, and ``status`` are all set.

When SGLang exposes per-token logprobs on ``/v1/messages`` (e.g. via a
``meta_info`` extension), populate ``tokens`` and ``rollout_log_probs`` here
and the multi-turn merge path will work automatically.
"""

import logging
from argparse import Namespace

from miles.rollout.session.types import SessionRecord
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def _content_to_text(content) -> str:
    """Extract plain text from an Anthropic ``content`` field (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def _stop_reason_to_status(stop_reason: str | None) -> Sample.Status:
    match stop_reason:
        case "end_turn" | "stop_sequence" | "tool_use":
            return Sample.Status.COMPLETED
        case "max_tokens":
            return Sample.Status.TRUNCATED
        case _:
            return Sample.Status.ABORTED


def _attach_anthropic_lifecycle_metadata(sample: Sample, record: SessionRecord, prev_record, turn: int) -> None:
    """Anthropic-format equivalent of ``miles.utils.lifecycle.attach_lifecycle_metadata``.

    The OpenAI version reads ``record.response["choices"][0]["meta_info"]["e2e_latency"]``
    which doesn't exist in Anthropic responses — those have ``content[]`` not ``choices[]``.
    ``t0`` is left None (no e2e_latency field in the Anthropic spec); the other
    timing boundaries (t1, req_ts, prev_t1) come from record fields, not the body.
    """
    segment: dict = dict(t0=None, t1=record.timestamp, turn=turn)
    if record.request_timestamp is not None:
        segment["req_ts"] = record.request_timestamp
    if prev_record is not None:
        segment["prev_t1"] = prev_record.timestamp
    sample.metadata["lifecycle"] = segment


def _sample_from_anthropic_record(record: SessionRecord) -> Sample:
    """Build a single-turn ``Sample`` from one Anthropic-format session record.

    ``tokens`` and ``rollout_log_probs`` are left empty (None / []) because
    the Anthropic response does not carry per-token logprobs.  ``loss_mask``
    is left None so ``merge_samples`` fills it with ones on merge.
    """
    response_body = record.response

    content = response_body.get("content", [])
    response_text = _content_to_text(content)

    usage = response_body.get("usage") or {}
    output_tokens = usage.get("output_tokens", 0)

    stop_reason = response_body.get("stop_reason")
    status = _stop_reason_to_status(stop_reason)

    sample = Sample()
    sample.response = response_text
    sample.response_length = output_tokens
    sample.status = status
    # tokens / rollout_log_probs / loss_mask intentionally left at defaults
    # ([] / None / None) — no logprobs available from /v1/messages yet.

    return sample


def compute_samples_from_anthropic_records(
    args: Namespace,
    records: list[SessionRecord],
    tokenizer,
) -> list[Sample]:
    """Convert Anthropic-format session records into per-turn ``Sample``s.

    For single-turn sessions the list has one element, ready for
    ``merge_samples``.  For multi-turn sessions each turn produces one
    ``Sample`` but the list is returned as-is — the caller (``collect_samples``)
    must be aware that ``merge_samples`` cannot stitch them without token IDs,
    and should call the single-sample path instead.
    """
    samples = []
    for i, record in enumerate(records):
        sample = _sample_from_anthropic_record(record)
        _attach_anthropic_lifecycle_metadata(sample, record, records[i - 1] if i else None, turn=i + 1)
        if i == len(records) - 1 and args.save_debug_trajectory_data is not None:
            # Reconstruct the full message list for debug dumps
            messages = record.request.get("messages", [])
            sample.metadata["messages"] = messages + [
                {"role": "assistant", "content": record.response.get("content", [])}
            ]
        samples.append(sample)
    return samples
