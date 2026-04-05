"""BCH code scaling experiment.

Tests whether LLMs can decode BCH codes of increasing size.
For each code, we encode random messages, add noise, and ask models to decode.

Usage:
    uv run python src/bch_scaling.py --n 50 --models "Haiku (no thinking)" --p 0.05
"""

import argparse
import json
import os
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
import galois

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys
sys.path.insert(0, os.path.join(ROOT, "src"))
from clients import anthropic_direct, extract_text

MODELS = {
    "Opus 4.6": {"model_id": "claude-opus-4-6", "backend": "anthropic", "thinking_budget": 4096},
    "Opus 4.6 (no thinking)": {"model_id": "claude-opus-4-6", "backend": "anthropic", "thinking_budget": None},
    "Sonnet 4": {"model_id": "claude-sonnet-4-20250514", "backend": "anthropic", "thinking_budget": 4096},
    "Sonnet 4 (no thinking)": {"model_id": "claude-sonnet-4-20250514", "backend": "anthropic", "thinking_budget": None},
    "Haiku": {"model_id": "claude-haiku-4-5-20251001", "backend": "anthropic", "thinking_budget": 4096},
    "Haiku (no thinking)": {"model_id": "claude-haiku-4-5-20251001", "backend": "anthropic", "thinking_budget": None},
}

# BCH codes to test: (n, k, t)
BCH_CODES = [
    (7, 4, 1),
    (15, 11, 1),
    (15, 7, 2),
    (31, 26, 1),
    (31, 16, 3),
]


def format_matrix(M, name):
    """Format a binary matrix for the prompt."""
    rows = []
    for row in M:
        rows.append("  [" + " ".join(str(int(b)) for b in row) + "]")
    return f"{name} =\n" + "\n".join(rows)


def format_bitstring(bits):
    return "".join(str(int(b)) for b in bits)


def make_decoder_prompt(bch, received_str, n, k, t, no_reasoning=False):
    """Build prompt for BCH decoding."""
    H = bch.H.view(np.ndarray).astype(int)
    H_str = format_matrix(H, "H")

    prompt = (
        f"You are decoding a received word from a BCH({n},{k}) code that can correct up to {t} errors.\n\n"
        f"Parity check matrix:\n{H_str}\n\n"
        f"Received word (MSB first): {received_str}\n\n"
        f"Steps:\n"
        f"1. Compute the syndrome s = H @ r (mod 2) where r is the received word as a column vector\n"
        f"2. If the syndrome is all zeros, the received word is a valid codeword — extract the first {k} bits as the message\n"
        f"3. If not, find and correct up to {t} bit errors to recover the nearest codeword, then extract the first {k} message bits\n\n"
        f"Output ONLY the {k}-bit decoded message (e.g. '{'0' * k}'), nothing else."
    )
    if no_reasoning:
        prompt += "\n\nAnswer only. Do not reason or explain. Just output the bitstring."
    return prompt


def make_encoder_prompt(bch, message_str, n, k):
    """Build prompt for BCH encoding."""
    G = bch.G.view(np.ndarray).astype(int)
    G_str = format_matrix(G, "G")

    prompt = (
        f"You are encoding a message using a BCH({n},{k}) code.\n\n"
        f"Generator matrix (systematic form):\n{G_str}\n\n"
        f"Message bits (MSB first): {message_str}\n\n"
        f"Compute the codeword: c = m @ G (mod 2), where m is the message as a row vector.\n\n"
        f"Output ONLY the {n}-bit codeword (e.g. '{'0' * n}'), nothing else."
    )
    return prompt


def add_exact_errors(cw, t):
    """Flip exactly t random bit positions."""
    noisy = list(cw)
    positions = random.sample(range(len(noisy)), t)
    for i in positions:
        noisy[i] ^= 1
    return noisy, positions


def parse_bitstring(text, expected_len):
    if text is None:
        return None
    import re
    m = re.search(r'([01]{' + str(expected_len) + r'})', text)
    if m:
        return [int(b) for b in m.group(1)]
    return None


def run_encode_experiment(client, model_name, bch_params, n_trials, workers):
    """Test whether model can encode: message → codeword via G matrix."""
    n, k, t = bch_params
    bch = galois.BCH(n, k=k)
    cfg = MODELS[model_name]

    # Precompute trial data (galois is not thread-safe due to numba)
    trial_data = []
    for _ in range(n_trials):
        msg_bits = [random.randint(0, 1) for _ in range(k)]
        msg_gf = galois.GF2(msg_bits)
        true_cw = [int(b) for b in bch.encode(msg_gf)]
        prompt = make_encoder_prompt(bch, format_bitstring(msg_bits), n, k)
        no_reasoning = cfg.get("thinking_budget") is None
        if no_reasoning:
            prompt += "\n\nAnswer only. Do not reason or explain. Just output the bitstring."
        trial_data.append({"msg_bits": msg_bits, "true_cw": true_cw, "prompt": prompt})

    results = []

    def _trial(td):
        no_reasoning = cfg.get("thinking_budget") is None
        max_tok = 100 if no_reasoning else (cfg.get("thinking_budget") or 0) + 2000
        content = anthropic_direct(client, cfg["model_id"], td["prompt"],
                                   thinking_budget=cfg.get("thinking_budget"),
                                   max_tokens=max_tok)
        text = extract_text(content)
        encoded = parse_bitstring(text, n)

        correct = encoded == td["true_cw"] if encoded is not None else None
        return {
            "message": format_bitstring(td["msg_bits"]),
            "true_codeword": format_bitstring(td["true_cw"]),
            "model_codeword": format_bitstring(encoded) if encoded else None,
            "correct": correct,
            "raw": text,
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_trial, td) for td in trial_data]
        for f in as_completed(futures):
            results.append(f.result())
    return results


def run_decode_experiment(client, model_name, bch_params, n_trials, workers):
    """Test whether model can decode: codeword with exactly t errors → message."""
    n, k, t = bch_params
    bch = galois.BCH(n, k=k)
    cfg = MODELS[model_name]

    # Precompute all trial data (galois is not thread-safe due to numba)
    trial_data = []
    for _ in range(n_trials):
        msg_bits = [random.randint(0, 1) for _ in range(k)]
        msg_gf = galois.GF2(msg_bits)
        codeword = bch.encode(msg_gf)
        cw_list = [int(b) for b in codeword]
        received, error_positions = add_exact_errors(cw_list, t)

        # Ground truth: run the correct decoder
        gt_decoded_cw = bch.decode(galois.GF2(received))
        gt_message = [int(b) for b in gt_decoded_cw[:k]]
        assert gt_message == msg_bits, f"galois decoder failed: {gt_message} != {msg_bits}"

        trial_data.append({
            "msg_bits": msg_bits,
            "cw_list": cw_list,
            "received": received,
            "error_positions": error_positions,
            "gt_message": gt_message,
        })

    results = []

    def _trial(td):
        msg_bits = td["msg_bits"]
        received_str = format_bitstring(td["received"])
        msg_str = format_bitstring(msg_bits)

        no_reasoning = cfg.get("thinking_budget") is None
        prompt = make_decoder_prompt(bch, received_str, n, k, t, no_reasoning=no_reasoning)
        max_tok = 100 if no_reasoning else (cfg.get("thinking_budget") or 0) + 2000
        content = anthropic_direct(client, cfg["model_id"], prompt,
                                   thinking_budget=cfg.get("thinking_budget"),
                                   max_tokens=max_tok)
        text = extract_text(content)
        decoded = parse_bitstring(text, k)

        correct = decoded == msg_bits if decoded is not None else None

        return {
            "message": msg_str,
            "codeword": format_bitstring(td["cw_list"]),
            "received": received_str,
            "n_errors": t,
            "error_positions": td["error_positions"],
            "gt_message": format_bitstring(td["gt_message"]),
            "decoded": format_bitstring(decoded) if decoded else None,
            "correct": correct,
            "raw": text,
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_trial, td) for td in trial_data]
        for f in as_completed(futures):
            results.append(f.result())
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--models", default="Opus 4.6,Haiku")
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    client = anthropic.Anthropic()
    models = [m.strip() for m in args.models.split(",")]

    all_results = {}

    for n, k, t in BCH_CODES:
        print(f"\n{'='*50}")
        print(f"BCH({n}, {k}) t={t}")
        print(f"{'='*50}")
        key = f"BCH({n},{k})_t{t}"
        all_results[key] = {}

        for model_name in models:
            # Encoding test
            print(f"\n  Encoding: {model_name}")
            enc_results = run_encode_experiment(client, model_name,
                                                (n, k, t), args.n, args.workers)
            enc_valid = sum(1 for r in enc_results if r["model_codeword"] is not None)
            enc_correct = sum(1 for r in enc_results if r["correct"])
            print(f"    Correct: {enc_correct}/{enc_valid}")

            # Decoding test (exactly t errors)
            print(f"  Decoding ({t} errors): {model_name}")
            dec_results = run_decode_experiment(client, model_name,
                                                (n, k, t), args.n, args.workers)
            dec_valid = sum(1 for r in dec_results if r["decoded"] is not None)
            dec_correct = sum(1 for r in dec_results if r["correct"])
            print(f"    Correct: {dec_correct}/{dec_valid}")

            all_results[key][model_name] = {
                "encode": {"trials": enc_results, "correct": enc_correct, "total": enc_valid},
                "decode": {"trials": dec_results, "correct": dec_correct, "total": dec_valid},
            }

    # Save (merge with existing results)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    outpath = os.path.join(ROOT, "results", "bch_scaling.json")
    if os.path.exists(outpath):
        with open(outpath) as f:
            existing = json.load(f)
        for code_key, model_data in all_results.items():
            if code_key not in existing:
                existing[code_key] = {}
            existing[code_key].update(model_data)
        all_results = existing
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
