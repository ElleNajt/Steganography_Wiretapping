"""API clients for Schelling steganography experiments.

Three backends:
  - AnthropicBatch: Anthropic batch API (50% cheaper, slower)
  - AnthropicDirect: Anthropic direct API (full price, fast)
  - OpenRouter: OpenRouter API (for non-Claude models)

All Anthropic functions return the full message content list (dicts),
preserving thinking blocks, text blocks, and tool_use blocks.
Use extract_text() or extract_tool_input() to pull out what you need.
"""
import anthropic
import httpx
import json
import os
import time

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")


# === Response helpers ===

def _content_to_dicts(content_blocks):
    """Convert API content blocks to serializable dicts."""
    out = []
    for block in content_blocks:
        if hasattr(block, "model_dump"):
            out.append(block.model_dump())
        elif isinstance(block, dict):
            out.append(block)
        else:
            out.append({"type": "unknown", "text": str(block)})
    return out


def extract_text(content):
    """Extract first text block from content list."""
    if content is None:
        return None
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            return block["text"].strip()
    return None


def extract_tool_input(content):
    """Extract input dict from first tool_use block."""
    if content is None:
        return None
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            return block.get("input", {})
    return None


# === Anthropic batch helpers ===

def submit_batch(client, requests, chunk_size=10000):
    """Submit requests in chunks, return list of batch IDs."""
    batch_ids = []
    for start in range(0, len(requests), chunk_size):
        chunk = requests[start:start + chunk_size]
        print(f"    chunk {start}-{start + len(chunk) - 1}")
        for attempt in range(10):
            try:
                batch = client.messages.batches.create(requests=chunk)
                print(f"    batch={batch.id} ({len(chunk)} requests)")
                batch_ids.append(batch.id)
                break
            except Exception as e:
                delay = min(30 * (attempt + 1), 300)
                print(f"    attempt {attempt + 1} failed: {e} (retry in {delay}s)")
                time.sleep(delay)
        else:
            raise RuntimeError("batch submit failed after 10 attempts")
    return batch_ids


def poll_batch(client, batch_ids):
    """Wait for all batches to complete."""
    remaining = set(range(len(batch_ids)))
    while remaining:
        for i in list(remaining):
            for attempt in range(5):
                try:
                    batch = client.messages.batches.retrieve(batch_ids[i])
                    break
                except (anthropic.APITimeoutError, anthropic.APIConnectionError) as e:
                    delay = 10 * (attempt + 1)
                    print(f"    poll retry {attempt + 1}: {e} (retry in {delay}s)")
                    time.sleep(delay)
            else:
                raise RuntimeError(f"poll failed after 5 attempts for {batch_ids[i]}")
            if batch.processing_status == "ended":
                remaining.discard(i)
            else:
                print(f"    {batch_ids[i][:20]}... {batch.processing_status}")
        if remaining:
            time.sleep(30)


def collect_batch(client, batch_ids):
    """Collect results from completed batches. Returns {custom_id: content_list}."""
    results = {}
    for bid in batch_ids:
        for attempt in range(5):
            try:
                for result in client.messages.batches.results(bid):
                    if result.result.type == "succeeded" and result.result.message.content:
                        results[result.custom_id] = _content_to_dicts(result.result.message.content)
                    else:
                        results[result.custom_id] = None
                break
            except (anthropic.APITimeoutError, anthropic.APIConnectionError, json.JSONDecodeError) as e:
                delay = 10 * (attempt + 1)
                print(f"    collect retry {attempt + 1}: {e} (retry in {delay}s)")
                time.sleep(delay)
        else:
            raise RuntimeError(f"collect failed after 5 attempts for {bid}")
    return results


def run_batch(client, requests, chunk_size=10000):
    """Submit, poll, collect. Returns {custom_id: content_list}."""
    batch_ids = submit_batch(client, requests, chunk_size)
    poll_batch(client, batch_ids)
    return collect_batch(client, batch_ids)


# === Anthropic direct API ===

def anthropic_direct(client, model_id, prompt, thinking_budget=4096,
                     max_tokens=8000, tools=None, tool_choice=None):
    """Single direct API call with retry. Returns content list (dicts)."""
    for attempt in range(5):
        try:
            kwargs = {
                "model": model_id,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if thinking_budget:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
            if tools:
                kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
            resp = client.messages.create(**kwargs)
            return _content_to_dicts(resp.content)
        except Exception as e:
            delay = 5 * (attempt + 1)
            print(f"    attempt {attempt + 1} failed: {e} (retry in {delay}s)")
            time.sleep(delay)
    return None


# === OpenRouter API ===

def openrouter_call(client_http, model_id, prompt, system_prompt=None,
                    max_tokens=3000, reasoning=False):
    """Single OpenRouter call with retry. Returns content as list of dicts."""
    messages = [{"role": "user", "content": prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    body = {
        "model": model_id,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if reasoning:
        body["reasoning"] = {"effort": "high"}

    for attempt in range(5):
        try:
            resp = client_http.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            text = msg.get("content", "")
            result = [{"type": "text", "text": text.strip() if text else ""}]
            if msg.get("reasoning_details"):
                result.append({"type": "reasoning", "details": msg["reasoning_details"]})
            return result
        except Exception as e:
            delay = 5 * (attempt + 1)
            print(f"    attempt {attempt + 1} failed: {e} (retry in {delay}s)")
            time.sleep(delay)
    return None
