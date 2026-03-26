"""API clients for Schelling steganography experiments.

Three backends:
  - AnthropicBatch: Anthropic batch API (50% cheaper, slower)
  - AnthropicDirect: Anthropic direct API (full price, fast)
  - OpenRouter: OpenRouter API (for non-Claude models)
"""
import anthropic
import httpx
import os
import time

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")


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
    for bid in batch_ids:
        while True:
            batch = client.messages.batches.retrieve(bid)
            if batch.processing_status == "ended":
                break
            print(f"    {bid[:20]}... {batch.processing_status}")
            time.sleep(30)


def collect_batch(client, batch_ids):
    """Collect results from completed batches. Returns {custom_id: text}."""
    results = {}
    for bid in batch_ids:
        for result in client.messages.batches.results(bid):
            if result.result.type == "succeeded" and result.result.message.content:
                for block in result.result.message.content:
                    if block.type == "text":
                        results[result.custom_id] = block.text.strip()
                        break
                else:
                    results[result.custom_id] = None
            else:
                results[result.custom_id] = None
    return results


def run_batch(client, requests, chunk_size=10000):
    """Submit, poll, collect. Returns {custom_id: text}."""
    batch_ids = submit_batch(client, requests, chunk_size)
    poll_batch(client, batch_ids)
    return collect_batch(client, batch_ids)


# === Anthropic direct API ===

def anthropic_direct(client, model_id, prompt, thinking_budget=4096, max_tokens=8000):
    """Single direct API call with retry. Returns text response."""
    for attempt in range(5):
        try:
            kwargs = {
                "model": model_id,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if thinking_budget:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
            resp = client.messages.create(**kwargs)
            for block in resp.content:
                if block.type == "text":
                    return block.text.strip()
            return None
        except Exception as e:
            delay = 5 * (attempt + 1)
            print(f"    attempt {attempt + 1} failed: {e} (retry in {delay}s)")
            time.sleep(delay)
    return None


# === OpenRouter API ===

def openrouter_call(client_http, model_id, prompt, system_prompt=None, max_tokens=3000, reasoning=False):
    """Single OpenRouter call with retry. Returns text response."""
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
            text = data["choices"][0]["message"]["content"]
            return text.strip() if text else None
        except Exception as e:
            delay = 5 * (attempt + 1)
            print(f"    attempt {attempt + 1} failed: {e} (retry in {delay}s)")
            time.sleep(delay)
    return None
