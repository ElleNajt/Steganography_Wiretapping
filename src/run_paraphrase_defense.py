"""Paraphrase defense: paraphrase encoded text, then decode.

Stores paraphrased text in a `paraphrases` table, and decode results
in `decodes` with decoder_model like "Opus 4.6 via Haiku-paraphrase".

Usage:
    python -m src.run_paraphrase_defense \
        --encoder-prompt alphabet \
        --paraphraser Haiku \
        --decoders "Opus 4.6,Opus 4.5,Sonnet 4,Haiku"
"""
import anthropic
import argparse
import json
import os
import re
import sys

from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from src.clients import run_batch, extract_text, anthropic_direct
    from src.stego_db import get_db, save_decode
    from src.prompts import PARAPHRASE_PROMPT, DECODER_PROMPTS
    from src.run_experiment import MODELS, RESPONSE_PARSERS, strip_fences, _sanitize_id, detect_domain
except ImportError:
    from clients import run_batch, extract_text, anthropic_direct
    from stego_db import get_db, save_decode
    from prompts import PARAPHRASE_PROMPT, DECODER_PROMPTS
    from run_experiment import MODELS, RESPONSE_PARSERS, strip_fences, _sanitize_id, detect_domain


def ensure_paraphrases_table(db):
    db.execute("""
        CREATE TABLE IF NOT EXISTS paraphrases (
            trial_idx INTEGER NOT NULL,
            encoder_model TEXT NOT NULL,
            prompt_type TEXT NOT NULL,
            seed_set TEXT NOT NULL,
            paraphraser_model TEXT NOT NULL,
            paraphrased_text TEXT,
            raw_response TEXT,
            PRIMARY KEY (trial_idx, encoder_model, prompt_type, seed_set, paraphraser_model),
            FOREIGN KEY (trial_idx, encoder_model, prompt_type, seed_set)
                REFERENCES trials (trial_idx, encoder_model, prompt_type, seed_set)
        )
    """)
    db.commit()


def get_existing_paraphrases(db, encoder_model, prompt_type, seed_set, paraphraser_model):
    rows = db.execute(
        """SELECT trial_idx FROM paraphrases
           WHERE encoder_model = ? AND prompt_type = ? AND seed_set = ?
           AND paraphraser_model = ? AND paraphrased_text IS NOT NULL""",
        (encoder_model, prompt_type, seed_set, paraphraser_model),
    ).fetchall()
    return {r[0] for r in rows}


def get_paraphrased_trials(db, encoder_model, prompt_type, seed_set, paraphraser_model):
    rows = db.execute(
        """SELECT p.trial_idx, p.paraphrased_text FROM paraphrases p
           JOIN trials t ON p.trial_idx = t.trial_idx
               AND p.encoder_model = t.encoder_model
               AND p.prompt_type = t.prompt_type
               AND p.seed_set = t.seed_set
           WHERE p.encoder_model = ? AND p.prompt_type = ? AND p.seed_set = ?
           AND p.paraphraser_model = ? AND p.paraphrased_text IS NOT NULL
           AND t.meaning_preserved = 1""",
        (encoder_model, prompt_type, seed_set, paraphraser_model),
    ).fetchall()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-prompt", required=True)
    parser.add_argument("--paraphraser", default="Haiku")
    parser.add_argument("--decoders", required=True, help="Comma-separated decoder models")
    parser.add_argument("--seeds", default="wikipedia")
    parser.add_argument("--db", default=None)
    parser.add_argument("--fast", action="store_true",
                        help="Use direct API calls instead of batch API")
    parser.add_argument("--encoders", default=None,
                        help="Comma-separated encoder models to process (default: all)")
    args = parser.parse_args()

    prompt_type = args.encoder_prompt
    seed_set = args.seeds
    paraphraser_label = args.paraphraser
    decoder_labels = [d.strip() for d in args.decoders.split(",")]

    db = get_db(args.db)
    ensure_paraphrases_table(db)

    client = anthropic.Anthropic()
    paraphraser_cfg = MODELS[paraphraser_label]
    paraphraser_model_id = paraphraser_cfg["model_id"]

    # Get all encoder models that have data for this prompt type
    encoder_models = [r[0] for r in db.execute(
        """SELECT DISTINCT encoder_model FROM trials
           WHERE prompt_type = ? AND seed_set = ? AND meaning_preserved = 1""",
        (prompt_type, seed_set),
    ).fetchall()]
    if args.encoders:
        filter_set = {e.strip() for e in args.encoders.split(",")}
        encoder_models = [m for m in encoder_models if m in filter_set]

    print(f"Prompt: {prompt_type}, Seeds: {seed_set}")
    print(f"Paraphraser: {paraphraser_label}")
    print(f"Decoders: {decoder_labels}")
    print(f"Encoder models: {encoder_models}")

    # Phase 1: Paraphrase
    print("\n" + "=" * 60)
    print("Phase 1: Paraphrase")
    print("=" * 60)

    for enc_label in encoder_models:
        preserved = db.execute(
            """SELECT trial_idx, encoded_text FROM trials
               WHERE encoder_model = ? AND prompt_type = ? AND seed_set = ?
               AND meaning_preserved = 1 AND encoded_text IS NOT NULL""",
            (enc_label, prompt_type, seed_set),
        ).fetchall()

        existing = get_existing_paraphrases(db, enc_label, prompt_type, seed_set, paraphraser_label)
        needs = [(idx, text) for idx, text in preserved if idx not in existing]

        if not needs:
            print(f"  {enc_label}: CACHED ({len(preserved)} paraphrases)")
            continue

        print(f"  {enc_label}: paraphrasing {len(needs)} trials")

        if args.fast:
            def _para(item):
                trial_idx, encoded_text = item
                prompt = PARAPHRASE_PROMPT.format(sentence=encoded_text)
                content = anthropic_direct(client, paraphraser_model_id, prompt,
                                           thinking_budget=None, max_tokens=2000)
                return trial_idx, content

            n_ok = 0
            with ThreadPoolExecutor(max_workers=20) as pool:
                futures = {pool.submit(_para, item): item for item in needs}
                for future in as_completed(futures):
                    trial_idx, content = future.result()
                    text = extract_text(content)
                    para_text = strip_fences(text.strip().strip('"')) if text else None
                    raw = json.dumps(content) if content else None
                    db.execute(
                        """INSERT OR REPLACE INTO paraphrases
                           (trial_idx, encoder_model, prompt_type, seed_set,
                            paraphraser_model, paraphrased_text, raw_response)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (trial_idx, enc_label, prompt_type, seed_set,
                         paraphraser_label, para_text, raw),
                    )
                    db.commit()
                    if para_text:
                        n_ok += 1
                    if n_ok % 200 == 0 and n_ok > 0:
                        print(f"    {enc_label}: {n_ok}/{len(needs)} paraphrased")
            print(f"  {enc_label}: {n_ok}/{len(needs)} paraphrased")
        else:
            requests = []
            for trial_idx, encoded_text in needs:
                prompt = PARAPHRASE_PROMPT.format(sentence=encoded_text)
                params = {
                    "model": paraphraser_model_id,
                    "max_tokens": 2000,
                    "messages": [{"role": "user", "content": prompt}],
                }
                requests.append({
                    "custom_id": _sanitize_id(f"para:{enc_label}:{trial_idx}"),
                    "params": params,
                })

            results = run_batch(client, requests)

            n_ok = 0
            for trial_idx, encoded_text in needs:
                cid = _sanitize_id(f"para:{enc_label}:{trial_idx}")
                content = results.get(cid)
                text = extract_text(content)
                para_text = strip_fences(text.strip().strip('"')) if text else None
                raw = json.dumps(content) if content else None

                db.execute(
                    """INSERT OR REPLACE INTO paraphrases
                       (trial_idx, encoder_model, prompt_type, seed_set,
                        paraphraser_model, paraphrased_text, raw_response)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (trial_idx, enc_label, prompt_type, seed_set,
                     paraphraser_label, para_text, raw),
                )
                if para_text:
                    n_ok += 1
            db.commit()
            print(f"  {enc_label}: {n_ok}/{len(needs)} paraphrased")

    # Phase 2: Decode paraphrased text
    print("\n" + "=" * 60)
    print("Phase 2: Decode paraphrased text")
    print("=" * 60)

    domain = detect_domain(seed_set)
    decoder_prompt_template = DECODER_PROMPTS.get(prompt_type, DECODER_PROMPTS[domain])
    response_parser = RESPONSE_PARSERS.get(prompt_type)

    for enc_label in encoder_models:
        paraphrased = get_paraphrased_trials(db, enc_label, prompt_type, seed_set, paraphraser_label)
        if not paraphrased:
            print(f"  {enc_label}: no paraphrased trials")
            continue

        for dec_label in decoder_labels:
            dec_cfg = MODELS[dec_label]
            dec_via = f"{dec_label} via {paraphraser_label}-paraphrase"

            # Check existing
            existing = db.execute(
                """SELECT trial_idx FROM decodes
                   WHERE encoder_model = ? AND prompt_type = ? AND seed_set = ?
                   AND decoder_model = ? AND decoded_symbol IS NOT NULL""",
                (enc_label, prompt_type, seed_set, dec_via),
            ).fetchall()
            existing_set = {r[0] for r in existing}

            needs = [(idx, text) for idx, text in paraphrased if idx not in existing_set]
            if not needs:
                print(f"  {enc_label} → {dec_via}: CACHED")
                continue

            print(f"  {enc_label} → {dec_via}: decoding {len(needs)} trials")

            model_id = dec_cfg["model_id"]
            thinking_budget = dec_cfg["thinking_budget"]

            if args.fast:
                def _decode(item):
                    trial_idx, para_text = item
                    prompt = decoder_prompt_template.format(sentence=para_text)
                    content = anthropic_direct(client, model_id, prompt,
                                               thinking_budget=thinking_budget,
                                               max_tokens=(thinking_budget or 0) + 3000)
                    return trial_idx, para_text, content

                n_ok = 0
                with ThreadPoolExecutor(max_workers=20) as pool:
                    futures = {pool.submit(_decode, item): item for item in needs}
                    for future in as_completed(futures):
                        trial_idx, para_text, content = future.result()
                        text = extract_text(content)
                        decoded = response_parser(text)
                        raw = json.dumps(content) if content else None
                        dec_prompt = decoder_prompt_template.format(sentence=para_text)
                        save_decode(db, trial_idx, enc_label, prompt_type, seed_set,
                                    dec_via, decoded, raw, dec_prompt)
                        if decoded:
                            n_ok += 1
                        if n_ok % 200 == 0 and n_ok > 0:
                            print(f"    {enc_label} → {dec_via}: {n_ok}/{len(needs)} decoded")
                print(f"  {enc_label} → {dec_via}: {n_ok}/{len(needs)} decoded")
            else:
                requests = []
                for trial_idx, para_text in needs:
                    prompt = decoder_prompt_template.format(sentence=para_text)
                    params = {
                        "model": model_id,
                        "max_tokens": (thinking_budget or 0) + 3000,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                    if thinking_budget:
                        params["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
                    requests.append({
                        "custom_id": _sanitize_id(f"{enc_label}:{dec_via}:{trial_idx}"),
                        "params": params,
                    })

                results = run_batch(client, requests)

                n_ok = 0
                for trial_idx, para_text in needs:
                    cid = _sanitize_id(f"{enc_label}:{dec_via}:{trial_idx}")
                    content = results.get(cid)
                    text = extract_text(content)
                    decoded = response_parser(text)
                    raw = json.dumps(content) if content else None
                    dec_prompt = decoder_prompt_template.format(sentence=para_text)

                    save_decode(db, trial_idx, enc_label, prompt_type, seed_set,
                                dec_via, decoded, raw, dec_prompt)
                    if decoded:
                        n_ok += 1
                print(f"  {enc_label} → {dec_via}: {n_ok}/{len(needs)} decoded")

    # Phase 3: Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    for enc_label in encoder_models:
        print(f"\n  {enc_label}:")
        for dec_label in decoder_labels:
            dec_via = f"{dec_label} via {paraphraser_label}-paraphrase"
            row = db.execute(
                """SELECT
                    COUNT(*) as n,
                    SUM(CASE WHEN d.decoded_symbol = t.symbol THEN 1 ELSE 0 END) as correct
                FROM decodes d
                JOIN trials t ON d.trial_idx = t.trial_idx
                    AND d.encoder_model = t.encoder_model
                    AND d.prompt_type = t.prompt_type
                    AND d.seed_set = t.seed_set
                WHERE d.encoder_model = ? AND d.prompt_type = ? AND d.seed_set = ?
                AND d.decoder_model = ? AND d.decoded_symbol IS NOT NULL""",
                (enc_label, prompt_type, seed_set, dec_via),
            ).fetchone()
            n, correct = row
            if n > 0:
                print(f"    {dec_via}: {correct}/{n} ({100*correct/n:.1f}%)")

            # Also print original for comparison
            row2 = db.execute(
                """SELECT
                    COUNT(*) as n,
                    SUM(CASE WHEN d.decoded_symbol = t.symbol THEN 1 ELSE 0 END) as correct
                FROM decodes d
                JOIN trials t ON d.trial_idx = t.trial_idx
                    AND d.encoder_model = t.encoder_model
                    AND d.prompt_type = t.prompt_type
                    AND d.seed_set = t.seed_set
                WHERE d.encoder_model = ? AND d.prompt_type = ? AND d.seed_set = ?
                AND d.decoder_model = ? AND d.decoded_symbol IS NOT NULL""",
                (enc_label, prompt_type, seed_set, dec_label),
            ).fetchone()
            n2, correct2 = row2
            if n2 > 0:
                print(f"    {dec_label} (original): {correct2}/{n2} ({100*correct2/n2:.1f}%)")


if __name__ == "__main__":
    main()
