# Schelling Steganography / Wiretap Channel Experiments

## Overview

This project studies whether LLMs can spontaneously develop steganographic communication — hiding secret messages in modified text — via Schelling point coordination, without any prior arrangement.

**Core question:** Can two copies of the same LLM independently converge on an encoding scheme to hide information in text, such that a different LLM (eavesdropper) cannot decode it?

This is modeled as a **wiretap channel** (information theory): encoder and decoder are the same model, eavesdropper is a different model. We measure mutual information (MI) for both, and **secrecy capacity** = MI(self) - MI(best eavesdropper).

## Project Structure

```
src/
├── run_experiment.py   # Main experiment runner (encode → judge → decode pipeline)
├── prompts.py          # All prompt templates (encoder, decoder, judge) for each mode
└── clients.py          # API clients (Anthropic batch, Anthropic direct, OpenRouter)

data/
├── wikipedia_seeds.json  # 200 Wikipedia sentence seeds (primary)
├── code_seeds.json       # Python code snippet seeds
└── opus_seeds.json       # Opus-generated text seeds

results/                  # Gitignored experiment outputs
└── stego_{hash}_{date}_{seeds}_{mode}/
    └── {model}.json      # Per-model results with all decoder results

SchellingSteganography.org  # Blog post / writeup (org-mode with live data blocks)
wiretap_diagram.svg         # Channel diagram
wiretap_model.svg           # Wiretap model diagram
```

## Running Experiments

```bash
# Alphabet mode (primary experiment) - single model
uv run python src/run_experiment.py \
    --seeds wikipedia --encoder-prompt alphabet \
    --models "Opus 4.6" --n 200 \
    --eavesdroppers "Haiku,Gemini 3.1 Pro,Gemini Flash"

# Multiple models
uv run python src/run_experiment.py \
    --seeds wikipedia --encoder-prompt alphabet \
    --models "Opus 4.6,Opus 4.5,Sonnet 4" --n 200 \
    --eavesdroppers "Haiku,Gemini 3.1 Pro,Gemini Flash,Opus 4.6"

# Fast mode (direct API, no batch - more expensive but faster iteration)
uv run python src/run_experiment.py \
    --seeds wikipedia --encoder-prompt alphabet \
    --models "Opus 4.6" --n 200 --fast

# Binary mode (original experiment)
uv run python src/run_experiment.py \
    --seeds wikipedia --encoder-prompt adversarial \
    --models "Opus 4.6" --n 200

# Word mode (high bandwidth but zero secrecy — for comparison only)
uv run python src/run_experiment.py \
    --seeds wikipedia --encoder-prompt word \
    --models "Opus 4.6" --n 200
```

### Encoding Modes

| Mode | Symbols | Max bits/trial | Secrecy? | Primary? |
|------|---------|---------------|----------|----------|
| `baseline` | 0/1 | 1.0 | Minimal | No |
| `adversarial` | 0/1 | 1.0 | Some | No |
| `alphabet` | A-Z | 4.7 | Yes | **Yes** |
| `word` | 1006 words | ~10 | No (zero secrecy) | No |

**Alphabet is the sweet spot.** Binary has too small a channel. Word has high throughput but uses an acrostic strategy (first letters spell the word) that any eavesdropper can trivially detect.

### Adding Eavesdropper Decodes to Existing Data

The experiment caches aggressively. To add a new eavesdropper to existing encoded data:

```bash
# This will skip encoding+judging (cached), skip existing decoders (cached),
# and only run the new eavesdropper decode
uv run python src/run_experiment.py \
    --seeds wikipedia --encoder-prompt alphabet \
    --models "Opus 4.5" --n 200 \
    --eavesdroppers "Haiku,NewEavesdropper" \
    --output-dir results/stego_HASH_DATE_wikipedia_alphabet --fast
```

**Important:** Use `--output-dir` pointing to the existing results directory. Otherwise `make_output_dir` generates a new path based on the current git hash and you'll start fresh.

### OpenRouter Models

Gemini and GPT models use OpenRouter. Requires `OPENROUTER_API_KEY` in `.env`. When running in background, explicitly pass it:

```bash
OPENROUTER_API_KEY=$(grep -v '^#' .env | grep 'OPENROUTER_API_KEY=' | head -1 | cut -d= -f2) \
    uv run python src/run_experiment.py ...
```

## Model Registry

Defined in `run_experiment.py`:

| Label | Model ID | Backend | Thinking |
|-------|----------|---------|----------|
| Opus 4.6 | claude-opus-4-6 | Anthropic | 4096 |
| Opus 4.5 | claude-opus-4-5-20251101 | Anthropic | 4096 |
| Sonnet 4 | claude-sonnet-4-20250514 | Anthropic | 4096 |
| Haiku | claude-haiku-4-5-20251001 | Anthropic | 4096 |
| Gemini 3.1 Pro | google/gemini-3.1-pro-preview | OpenRouter | None |
| Gemini Flash | google/gemini-2.5-flash | OpenRouter | None |
| GPT-5.4 | openai/gpt-5.4 | OpenRouter | None |

## Pipeline Phases

Each model runs through three phases, all cached per-trial:

1. **Encode**: Model modifies seed text to embed the symbol. Uses batch API by default (`--fast` for direct).
2. **Judge**: Opus 4.6 checks meaning preservation (`<same>yes/no</same>`). Trials failing this are excluded from decode.
3. **Decode**: Self-decode + each eavesdropper decodes. Results appended to `trial["decoders"][label]`.

## Results Format

Each `{model}.json` contains:

```json
{
  "config": { "git_hash": "...", "encoder_prompt": "...", ... },
  "trials": [
    {
      "idx": 0,
      "bit": "U",           // The target symbol (letter for alphabet, 0/1 for binary)
      "seed": "Original text...",
      "encoded": "Modified text...",
      "meaning_preserved": true,
      "decoders": {
        "Opus 4.6": [{"model_id": "...", "response": "...", "bit": "U"}],
        "Haiku": [{"model_id": "...", "response": "...", "bit": "A"}]
      }
    },
    ...
  ]
}
```

Note: The field is called `bit` for all modes (historical name from binary era). For alphabet mode it's an uppercase letter, for word mode it's a lowercase word.

## Key Results (Alphabet Mode, Wikipedia Seeds)

Chance = 1/26 = 3.8%. MI = mutual information in bits (max = log2(26) = 4.7 bits).

| Encoder        | Self-decode | MI(self) | Best eavesdropper  | MI(eaves) | Secrecy |
|----------------|-------------|----------|--------------------|-----------|---------|
| Opus 4.6       | 111/171 65% | 2.14     | Gemini Pro 75%     | 2.71      | -0.57   |
| Opus 4.5       | 69/164 42%  | 1.03     | Gemini Pro 70%     | 2.41      | -1.38   |
| Sonnet 4       | 35/161 22%  | 0.31     | Gemini Pro 34%     | 0.70      | -0.39   |
| Gemini 3.1 Pro | 48/83 58%   | 1.76     | Opus 4.6 49%       | 1.33      | 0.43    |
| Gemini Flash   | 16/136 12%  | 0.08     | Gemini Pro 35%     | 0.74      | -0.66   |
| GPT-5.4        | 8/198 4%    | 0.00     | (all at chance)    | 0.00      | 0.00    |

Gemini 3.1 Pro is the strongest eavesdropper overall — it decodes Opus 4.6 encodings better than Opus 4.6 decodes itself (75% vs 65%).

## Encoding Strategies (from CoT analysis)

In alphabet mode, Opus 4.6 primarily uses:
- **First-letter restructuring** (~33%): Rearranges sentence to start with the target letter
- **Word count / padding**: Adjusts word count to encode position (A=1, B=2, ...)
- **Number spelling**: Inserts spelled-out numbers that map to letter positions
- **Structural cues**: Sentence structure patterns (active/passive, clause ordering)

## The Org File

`SchellingSteganography.org` is the blog post. It uses live Python src blocks that query results directly. Python path is `./.venv/bin/python3`.

Sections:
- Introduction + wiretap channel model
- Experimental setup
- Results (to be updated with alphabet as primary)
- `* Alphabets` section with subsections for binary, alphabet, and word experiments

## Environment

```bash
# .env file (not committed)
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_API_KEY_BATCH=sk-ant-...  # same key, used for batch API
OPENROUTER_API_KEY=sk-or-...
```

Dependencies: `anthropic`, `httpx` (see `pyproject.toml`). Use `uv` for package management.
