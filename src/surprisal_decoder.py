"""
Surprisal-based algorithmic decoder for Q&A steganography.

For each trial, computes per-word surprisal of the answer conditioned on
context+question using a small LM, then guesses the first letter of the
most surprising word.
"""
import json
import re
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


def word_boundaries(text, tokenizer):
    """Map each token back to the word it belongs to.
    Returns list of (word_idx, word_text) for each token."""
    words = text.split()
    # Build character-level word index
    char_to_word = []
    for wi, w in enumerate(words):
        char_to_word.extend([wi] * len(w))
        char_to_word.append(wi)  # space after word
    if char_to_word:
        char_to_word.pop()  # remove trailing

    # Tokenize and map tokens to words
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    token_ids = encoding['input_ids']
    offsets = encoding['offset_mapping']

    token_to_word = []
    for start, end in offsets:
        if start < len(char_to_word):
            token_to_word.append(char_to_word[start])
        else:
            token_to_word.append(len(words) - 1)

    return token_ids, token_to_word, words


def compute_surprisal(model, tokenizer, prefix, answer, device):
    """Compute per-token surprisal of answer conditioned on prefix."""
    # Tokenize prefix and answer separately
    prefix_ids = tokenizer(prefix, add_special_tokens=True)['input_ids']
    answer_encoding = tokenizer(answer, return_offsets_mapping=True, add_special_tokens=False)
    answer_ids = answer_encoding['input_ids']
    answer_offsets = answer_encoding['offset_mapping']

    if not answer_ids:
        return [], [], []

    # Combine
    input_ids = prefix_ids + answer_ids
    input_tensor = torch.tensor([input_ids], device=device)

    with torch.no_grad():
        outputs = model(input_tensor)
        logits = outputs.logits[0]  # (seq_len, vocab_size)

    # Get log probs for each answer token
    # logits[i] predicts token at position i+1
    # So logits[len(prefix)-1] predicts the first answer token
    surprisals = []
    prefix_len = len(prefix_ids)
    for i, tid in enumerate(answer_ids):
        logit_pos = prefix_len + i - 1
        if logit_pos < 0:
            surprisals.append(0.0)
            continue
        log_probs = torch.nn.functional.log_softmax(logits[logit_pos], dim=-1)
        surprisals.append(-log_probs[tid].item())

    # Map tokens to words
    words = answer.split()
    char_to_word = []
    for wi, w in enumerate(words):
        char_to_word.extend([wi] * len(w))
        char_to_word.append(wi)  # space
    if char_to_word:
        char_to_word.pop()

    token_to_word = []
    for start, end in answer_offsets:
        if start < len(char_to_word):
            token_to_word.append(char_to_word[start])
        else:
            token_to_word.append(len(words) - 1)

    return surprisals, token_to_word, words


def decode_trial(model, tokenizer, trial, device):
    """Compute and save all per-token surprisals for a trial."""
    context = trial['context']
    question = trial['question']
    answer = trial['encoded_text']

    if not answer or not answer.strip():
        return None

    prefix = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

    surprisals, token_to_word, words = compute_surprisal(
        model, tokenizer, prefix, answer, device
    )

    if not words or not surprisals:
        return None

    # Get token texts
    answer_encoding = tokenizer(answer, return_offsets_mapping=True, add_special_tokens=False)
    answer_offsets_local = answer_encoding['offset_mapping']
    tokens = []
    for idx, (start, end) in enumerate(answer_offsets_local):
        tokens.append({
            'text': answer[start:end],
            'surprisal': round(surprisals[idx], 4),
            'word_idx': token_to_word[idx] if idx < len(token_to_word) else -1,
        })

    return {
        'words': words,
        'tokens': tokens,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2.5-0.5B',
                        help='HuggingFace model name')
    parser.add_argument('--input', required=True, help='Input JSON with trials')
    parser.add_argument('--output', required=True, help='Output JSON with results')
    parser.add_argument('--max-prefix-tokens', type=int, default=2048,
                        help='Max tokens for prefix (context+question)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading {args.model} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device).eval()

    with open(args.input) as f:
        trials = json.load(f)

    print(f"Processing {len(trials)} trials...")
    results = []
    correct = {'word': {}, 'token': {}}
    total = {}

    for i, trial in enumerate(trials):
        result = decode_trial(model, tokenizer, trial, device)
        pt = trial['prompt_type']
        total[pt] = total.get(pt, 0) + 1
        for s in ['word', 'token']:
            correct[s].setdefault(pt, 0)

        if result:
            target = trial['symbol'].upper()

            # Compute guesses from raw surprisals for progress reporting
            # Word-level: mean surprisal per word, guess first letter of most surprising word
            from collections import defaultdict
            word_surprisals = defaultdict(list)
            for tok in result['tokens']:
                if tok['word_idx'] >= 0:
                    word_surprisals[tok['word_idx']].append(tok['surprisal'])
            word_means = {wi: sum(s)/len(s) for wi, s in word_surprisals.items()}
            if word_means:
                max_word_idx = max(word_means, key=word_means.get)
                chosen_word = result['words'][max_word_idx]
                guess_word = chosen_word[0].upper() if chosen_word else None
            else:
                chosen_word, guess_word = None, None

            # Token-level: first letter of most surprising single token
            if result['tokens']:
                max_tok = max(result['tokens'], key=lambda t: t['surprisal'])
                chosen_token = max_tok['text'].strip()
                guess_token = chosen_token[0].upper() if chosen_token else None
            else:
                chosen_token, guess_token = None, None

            r = {
                'trial_idx': trial['trial_idx'],
                'symbol': trial['symbol'],
                'prompt_type': pt,
                'words': result['words'],
                'tokens': result['tokens'],
                'guess_word': guess_word,
                'guess_token': guess_token,
                'correct_word': guess_word == target,
                'correct_token': guess_token == target,
            }
            if r['correct_word']:
                correct['word'][pt] += 1
            if r['correct_token']:
                correct['token'][pt] += 1
            results.append(r)
        else:
            results.append({
                'trial_idx': trial['trial_idx'],
                'symbol': trial['symbol'],
                'prompt_type': pt,
                'words': None, 'tokens': None,
                'guess_word': None, 'guess_token': None,
                'correct_word': False, 'correct_token': False,
            })

        if (i + 1) % 50 == 0:
            for pt_key in total:
                if total[pt_key] > 0:
                    w = correct['word'][pt_key]
                    t = correct['token'][pt_key]
                    n = total[pt_key]
                    print(f"  [{i+1}/{len(trials)}] {pt_key}: word={w}/{n} ({100*w/n:.1f}%) token={t}/{n} ({100*t/n:.1f}%)")

    print(f"\nFinal results:")
    for pt_key in total:
        if total[pt_key] > 0:
            n = total[pt_key]
            for s in ['word', 'token']:
                c = correct[s][pt_key]
                print(f"  {pt_key} [{s}]: {c}/{n} ({100*c/n:.1f}%)")
    print(f"  Chance: {100/26:.1f}%")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
