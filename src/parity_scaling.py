"""Parity/repetition coding experiment.

Tests whether LLMs can execute the naive parity+repetition scheme:
  - Encode: given W and n, produce a random n-bit string with parity(X^n) = W
  - Decode: given m noisy n-bit blocks, compute parity of each, majority vote

Uses the schedule n = ceil(alpha * ln(m)) from the wiretap analysis,
with p_B = 0.05, p_E = 0.15, alpha = 3.

Usage:
    uv run python src/parity_scaling.py --trials 50 --models "Haiku"
"""

import argparse
import json
import math
import os
import random
import re
import subprocess
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys
sys.path.insert(0, os.path.join(ROOT, "src"))
from clients import anthropic_direct, extract_text

MODELS = {
    "Opus 4.6": {"model_id": "claude-opus-4-6", "thinking_budget": 4096},
    "Opus 4.6 (no thinking)": {"model_id": "claude-opus-4-6", "thinking_budget": None},
    "Haiku": {"model_id": "claude-haiku-4-5-20251001", "thinking_budget": 4096},
    "Haiku (no thinking)": {"model_id": "claude-haiku-4-5-20251001", "thinking_budget": None},
}

P_B = 0.05
P_E = 0.15
ALPHA = 3.0

# (m, n) pairs along the schedule n = ceil(alpha * ln(m))
SCHEDULE = []
for m in [3, 5, 9, 15, 25, 51]:
    n = max(2, math.ceil(ALPHA * math.log(m)))
    SCHEDULE.append((m, n))


def format_bitstring(bits):
    return "".join(str(int(b)) for b in bits)


def add_noise(bits, p):
    return [b ^ (1 if random.random() < p else 0) for b in bits]


def random_parity_string(n, w):
    """Generate a random n-bit string with parity w."""
    bits = [random.randint(0, 1) for _ in range(n - 1)]
    current_parity = sum(bits) % 2
    bits.append((w - current_parity) % 2)
    random.shuffle(bits)
    return bits


def make_encode_prompt(w, n, random_bits=None, no_reasoning=False):
    if random_bits is not None:
        # External randomness mode: give n-1 bits, ask model to append parity bit
        bits_str = "".join(str(b) for b in random_bits)
        prompt = (
            f"You are given {n-1} random bits: {bits_str}\n\n"
            f"Append one bit so that the parity (XOR of all {n} bits) equals {w}.\n\n"
            f"Output ONLY the full {n}-bit string, nothing else."
        )
    else:
        prompt = (
            f"Generate a random {n}-bit binary string whose parity (XOR of all bits) is {w}.\n\n"
            f"The string should look random — not all zeros or a simple pattern.\n\n"
            f"Output ONLY the {n}-bit string (e.g. '{'0' * n}'), nothing else."
        )
    if no_reasoning:
        prompt += "\n\nAnswer only. Do not reason or explain. Just output the bitstring."
    return prompt


def make_decode_prompt(blocks, m, n, no_reasoning=False):
    blocks_str = "\n".join(f"Block {i+1}: {format_bitstring(b)}" for i, b in enumerate(blocks))
    prompt = (
        f"You are given {m} binary blocks, each of length {n}. "
        f"A single secret bit W was encoded as the parity of each block (before noise was added).\n\n"
        f"{blocks_str}\n\n"
        f"Steps:\n"
        f"1. Compute the parity (XOR of all bits) of each block\n"
        f"2. Take the majority vote of the {m} parities\n"
        f"3. Output the majority value\n\n"
        f"Output ONLY the single decoded bit (0 or 1), nothing else."
    )
    if no_reasoning:
        prompt += "\n\nAnswer only. Do not reason or explain. Just output the bit."
    return prompt


def parse_bit(text):
    if text is None:
        return None
    m = re.search(r'[01]', text)
    return int(m.group()) if m else None


def parse_bitstring(text, expected_len):
    if text is None:
        return None
    m = re.search(r'([01]{' + str(expected_len) + r'})', text)
    if m:
        return [int(b) for b in m.group(1)]
    return None


def call_model(client, model_name, prompt):
    cfg = MODELS[model_name]
    no_reasoning = cfg.get("thinking_budget") is None
    max_tok = 100 if no_reasoning else (cfg.get("thinking_budget") or 0) + 2000
    content = anthropic_direct(client, cfg["model_id"], prompt,
                               thinking_budget=cfg.get("thinking_budget"),
                               max_tokens=max_tok)
    return extract_text(content)


def run_encode_trials(client, model_name, schedule, n_trials, workers,
                      external_randomness=False):
    """Test encoding: can the model produce n-bit strings with correct parity?"""
    cfg = MODELS[model_name]
    no_reasoning = cfg.get("thinking_budget") is None
    results = {}

    for m, n in schedule:
        key = f"m{m}_n{n}"
        trials = []

        def _trial(trial_idx, _n=n):
            w = random.randint(0, 1)
            if external_randomness:
                rand_bits = [random.randint(0, 1) for _ in range(_n - 1)]
            else:
                rand_bits = None
            prompt = make_encode_prompt(w, _n, random_bits=rand_bits,
                                        no_reasoning=no_reasoning)
            text = call_model(client, model_name, prompt)
            bits = parse_bitstring(text, _n)
            if bits is not None:
                actual_parity = sum(bits) % 2
                correct = actual_parity == w
                # Check model preserved the random bits (if external randomness)
                if external_randomness and rand_bits is not None:
                    preserved = bits[:_n-1] == rand_bits
                else:
                    preserved = None
            else:
                correct = None
                preserved = None
            return {
                "target_parity": w,
                "n": _n,
                "random_bits": format_bitstring(rand_bits) if rand_bits else None,
                "model_output": format_bitstring(bits) if bits else None,
                "correct_parity": correct,
                "preserved_prefix": preserved,
                "raw": text,
            }

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_trial, i) for i in range(n_trials)]
            for f in as_completed(futures):
                trials.append(f.result())

        valid = sum(1 for t in trials if t["correct_parity"] is not None)
        correct = sum(1 for t in trials if t["correct_parity"])
        preserved = sum(1 for t in trials if t.get("preserved_prefix") is True)
        extra = f"  preserved_prefix={preserved}/{valid}" if external_randomness else ""
        print(f"    n={n}: {correct}/{valid}{extra}")
        results[key] = {"m": m, "n": n, "trials": trials,
                        "correct": correct, "valid": valid}
    return results


def run_decode_trials(client, model_name, schedule, n_trials, p_noise, workers):
    """Test decoding: given m noisy blocks, compute parities and majority vote."""
    cfg = MODELS[model_name]
    no_reasoning = cfg.get("thinking_budget") is None
    results = {}

    for m, n in schedule:
        key = f"m{m}_n{n}"
        trials = []

        def _trial(trial_idx, _m=m, _n=n):
            w = random.randint(0, 1)
            # Generate m blocks with parity w, add noise
            clean_blocks = [random_parity_string(_n, w) for _ in range(_m)]
            noisy_blocks = [add_noise(b, p_noise) for b in clean_blocks]

            # Ground truth parities (clean and noisy)
            clean_parities = [sum(b) % 2 for b in clean_blocks]
            noisy_parities = [sum(b) % 2 for b in noisy_blocks]
            n_flipped_parities = sum(c != n_ for c, n_ in zip(clean_parities, noisy_parities))
            true_majority = 1 if sum(noisy_parities) > _m / 2 else 0

            prompt = make_decode_prompt(noisy_blocks, _m, _n, no_reasoning=no_reasoning)
            text = call_model(client, model_name, prompt)
            decoded = parse_bit(text)

            return {
                "w": w,
                "m": _m,
                "n": _n,
                "clean_blocks": [format_bitstring(b) for b in clean_blocks],
                "noisy_blocks": [format_bitstring(b) for b in noisy_blocks],
                "clean_parities": clean_parities,
                "noisy_parities": noisy_parities,
                "n_flipped_parities": n_flipped_parities,
                "true_majority": true_majority,
                "correct_w": decoded == w if decoded is not None else None,
                "correct_majority": decoded == true_majority if decoded is not None else None,
                "decoded": decoded,
                "raw": text,
            }

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_trial, i) for i in range(n_trials)]
            for f in as_completed(futures):
                trials.append(f.result())

        valid = sum(1 for t in trials if t["correct_majority"] is not None)
        correct_maj = sum(1 for t in trials if t["correct_majority"])
        correct_w = sum(1 for t in trials if t["correct_w"])
        print(f"    m={m}, n={n}: correct_majority={correct_maj}/{valid}  correct_w={correct_w}/{valid}")
        results[key] = {"m": m, "n": n, "trials": trials,
                        "correct_majority": correct_maj, "correct_w": correct_w, "valid": valid}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--models", default="Haiku")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--p-bob", type=float, default=P_B)
    parser.add_argument("--p-eve", type=float, default=P_E)
    parser.add_argument("--external-randomness", action="store_true",
                        help="Provide n-1 random bits to the model; it only appends the parity bit")
    args = parser.parse_args()

    git_hash = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT
    ).decode().strip()
    date = datetime.date.today().isoformat()

    client = anthropic.Anthropic()
    models = [m.strip() for m in args.models.split(",")]

    all_results = {
        "config": {
            "git_hash": git_hash,
            "date": date,
            "alpha": ALPHA,
            "p_bob": args.p_bob,
            "p_eve": args.p_eve,
            "n_trials": args.trials,
            "external_randomness": args.external_randomness,
            "schedule": [{"m": m, "n": n} for m, n in SCHEDULE],
            "models": {name: MODELS[name] for name in models},
        },
        "results": {},
    }

    for model_name in models:
        print(f"\n{'='*50}")
        print(f"{model_name}")
        print(f"{'='*50}")

        label = "correct parity, external randomness" if args.external_randomness else "correct parity?"
        print(f"\n  Encoding ({label}):")
        enc = run_encode_trials(client, model_name, SCHEDULE, args.trials, args.workers,
                                external_randomness=args.external_randomness)

        print(f"\n  Decoding (Bob, p={args.p_bob}):")
        dec_bob = run_decode_trials(client, model_name, SCHEDULE, args.trials,
                                    args.p_bob, args.workers)

        print(f"\n  Decoding (Eve, p={args.p_eve}):")
        dec_eve = run_decode_trials(client, model_name, SCHEDULE, args.trials,
                                    args.p_eve, args.workers)

        all_results["results"][model_name] = {
            "encode": enc,
            "decode_bob": dec_bob,
            "decode_eve": dec_eve,
        }

    outdir = os.path.join(ROOT, "results", f"parity_{git_hash}_{date}")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "parity_scaling.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
