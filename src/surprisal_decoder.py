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
    """Decode a single trial using surprisal."""
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

    # Aggregate surprisal per word (sum of token surprisals)
    n_words = len(words)
    word_surprisal = [0.0] * n_words
    word_token_count = [0] * n_words
    for s, wi in zip(surprisals, token_to_word):
        if wi < n_words:
            word_surprisal[wi] += s
            word_token_count[wi] += 1

    # Normalize by token count (mean surprisal per token)
    word_mean_surprisal = []
    for i in range(n_words):
        if word_token_count[i] > 0:
            word_mean_surprisal.append(word_surprisal[i] / word_token_count[i])
        else:
            word_mean_surprisal.append(0.0)

    # Pick word with highest mean surprisal
    max_idx = max(range(n_words), key=lambda i: word_mean_surprisal[i])
    chosen_word = words[max_idx]

    # Extract first letter
    first_letter = re.sub(r'[^a-zA-Z]', '', chosen_word)
    if not first_letter:
        return None

    guess = first_letter[0].upper()

    return {
        'guess': guess,
        'chosen_word': chosen_word,
        'chosen_word_idx': max_idx,
        'word_surprisals': {words[i]: round(word_mean_surprisal[i], 3) for i in range(n_words)},
        'max_surprisal': round(word_mean_surprisal[max_idx], 3),
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
    correct = {'qa_alphabet_blind': 0, 'qa_alphabet_subtle2': 0}
    total = {'qa_alphabet_blind': 0, 'qa_alphabet_subtle2': 0}

    for i, trial in enumerate(trials):
        result = decode_trial(model, tokenizer, trial, device)
        pt = trial['prompt_type']
        if result:
            is_correct = result['guess'] == trial['symbol'].upper()
            total[pt] += 1
            if is_correct:
                correct[pt] += 1
            results.append({
                'trial_idx': trial['trial_idx'],
                'symbol': trial['symbol'],
                'prompt_type': pt,
                'guess': result['guess'],
                'correct': is_correct,
                'chosen_word': result['chosen_word'],
                'max_surprisal': result['max_surprisal'],
            })
        else:
            results.append({
                'trial_idx': trial['trial_idx'],
                'symbol': trial['symbol'],
                'prompt_type': pt,
                'guess': None,
                'correct': False,
            })
            total[pt] += 1

        if (i + 1) % 50 == 0:
            for pt_key in correct:
                if total[pt_key] > 0:
                    print(f"  [{i+1}/{len(trials)}] {pt_key}: {correct[pt_key]}/{total[pt_key]} ({100*correct[pt_key]/total[pt_key]:.1f}%)")

    print(f"\nFinal results:")
    for pt_key in correct:
        if total[pt_key] > 0:
            print(f"  {pt_key}: {correct[pt_key]}/{total[pt_key]} ({100*correct[pt_key]/total[pt_key]:.1f}%)")
    print(f"  Chance: {100/26:.1f}%")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
