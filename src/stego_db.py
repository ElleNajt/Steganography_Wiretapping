"""SQLite storage for Schelling steganography experiments."""
import json
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "stego.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    trial_idx INTEGER NOT NULL,
    encoder_model TEXT NOT NULL,
    prompt_type TEXT NOT NULL,
    seed_set TEXT NOT NULL,
    git_hash TEXT,
    seed_text TEXT NOT NULL,
    symbol TEXT NOT NULL,
    encoded_text TEXT,
    raw_encoder_response TEXT,
    meaning_preserved INTEGER,
    raw_judge_response TEXT,
    encoder_prompt_text TEXT,
    judge_prompt_text TEXT,
    PRIMARY KEY (trial_idx, encoder_model, prompt_type, seed_set)
);

CREATE TABLE IF NOT EXISTS decodes (
    trial_idx INTEGER NOT NULL,
    encoder_model TEXT NOT NULL,
    prompt_type TEXT NOT NULL,
    seed_set TEXT NOT NULL,
    decoder_model TEXT NOT NULL,
    decoded_symbol TEXT,
    raw_response TEXT,
    decoder_prompt_text TEXT,
    PRIMARY KEY (trial_idx, encoder_model, prompt_type, seed_set, decoder_model),
    FOREIGN KEY (trial_idx, encoder_model, prompt_type, seed_set)
        REFERENCES trials (trial_idx, encoder_model, prompt_type, seed_set)
);
"""

# Migration: add columns to existing DBs
_MIGRATIONS = [
    "ALTER TABLE trials ADD COLUMN encoder_prompt_text TEXT",
    "ALTER TABLE trials ADD COLUMN judge_prompt_text TEXT",
    "ALTER TABLE decodes ADD COLUMN decoder_prompt_text TEXT",
]


def get_db(path=None):
    if path is None:
        path = DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    # Add columns to existing tables (idempotent - "duplicate column" is the only expected error)
    for migration in _MIGRATIONS:
        try:
            conn.execute(migration)
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e):
                raise
    return conn


def save_trial(conn, trial_idx, encoder_model, prompt_type, seed_set,
               git_hash, seed_text, symbol, encoded_text,
               raw_encoder_response, meaning_preserved, encoder_prompt_text=None):
    conn.execute(
        """INSERT OR REPLACE INTO trials
           (trial_idx, encoder_model, prompt_type, seed_set, git_hash,
            seed_text, symbol, encoded_text, raw_encoder_response, meaning_preserved,
            encoder_prompt_text)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (trial_idx, encoder_model, prompt_type, seed_set, git_hash,
         seed_text, symbol, encoded_text, raw_encoder_response, meaning_preserved,
         encoder_prompt_text),
    )
    conn.commit()


def save_trial(conn, row):
    """Insert single trial. row: 11-tuple."""
    conn.execute(
        """INSERT OR REPLACE INTO trials
           (trial_idx, encoder_model, prompt_type, seed_set, git_hash,
            seed_text, symbol, encoded_text, raw_encoder_response, meaning_preserved,
            encoder_prompt_text)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        row,
    )
    conn.commit()


def save_trials_batch(conn, rows):
    """Batch insert trials. rows: list of 11-tuples (trial_idx, encoder_model, prompt_type,
    seed_set, git_hash, seed_text, symbol, encoded_text, raw_encoder_response,
    meaning_preserved, encoder_prompt_text)."""
    conn.executemany(
        """INSERT OR REPLACE INTO trials
           (trial_idx, encoder_model, prompt_type, seed_set, git_hash,
            seed_text, symbol, encoded_text, raw_encoder_response, meaning_preserved,
            encoder_prompt_text)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()


def save_decode(conn, trial_idx, encoder_model, prompt_type, seed_set,
                decoder_model, decoded_symbol, raw_response, decoder_prompt_text=None):
    conn.execute(
        """INSERT OR REPLACE INTO decodes
           (trial_idx, encoder_model, prompt_type, seed_set,
            decoder_model, decoded_symbol, raw_response, decoder_prompt_text)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (trial_idx, encoder_model, prompt_type, seed_set,
         decoder_model, decoded_symbol, raw_response, decoder_prompt_text),
    )
    conn.commit()


def save_decodes_batch(conn, rows):
    """Batch insert decodes. rows: list of 8-tuples (trial_idx, encoder_model, prompt_type,
    seed_set, decoder_model, decoded_symbol, raw_response, decoder_prompt_text)."""
    conn.executemany(
        """INSERT OR REPLACE INTO decodes
           (trial_idx, encoder_model, prompt_type, seed_set,
            decoder_model, decoded_symbol, raw_response, decoder_prompt_text)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()


def update_meaning_preserved(conn, trial_idx, encoder_model, prompt_type,
                             seed_set, meaning_preserved, raw_judge_response=None,
                             judge_prompt_text=None):
    conn.execute(
        """UPDATE trials SET meaning_preserved = ?, raw_judge_response = ?, judge_prompt_text = ?
           WHERE trial_idx = ? AND encoder_model = ?
           AND prompt_type = ? AND seed_set = ?""",
        (meaning_preserved, raw_judge_response, judge_prompt_text,
         trial_idx, encoder_model, prompt_type, seed_set),
    )
    conn.commit()


def update_meaning_preserved_batch(conn, rows):
    """Batch update. rows: list of (meaning_preserved, raw_judge_response, judge_prompt_text,
    trial_idx, encoder_model, prompt_type, seed_set)."""
    conn.executemany(
        """UPDATE trials SET meaning_preserved = ?, raw_judge_response = ?, judge_prompt_text = ?
           WHERE trial_idx = ? AND encoder_model = ?
           AND prompt_type = ? AND seed_set = ?""",
        rows,
    )
    conn.commit()


def get_encoded_trial_idxs(conn, encoder_model, prompt_type, seed_set):
    """Return set of trial_idx that have encoded_text."""
    rows = conn.execute(
        """SELECT trial_idx FROM trials
           WHERE encoder_model = ? AND prompt_type = ? AND seed_set = ?
           AND encoded_text IS NOT NULL""",
        (encoder_model, prompt_type, seed_set),
    ).fetchall()
    return {r[0] for r in rows}


def get_judged_trial_idxs(conn, encoder_model, prompt_type, seed_set):
    """Return set of trial_idx that have meaning_preserved set."""
    rows = conn.execute(
        """SELECT trial_idx FROM trials
           WHERE encoder_model = ? AND prompt_type = ? AND seed_set = ?
           AND meaning_preserved IS NOT NULL""",
        (encoder_model, prompt_type, seed_set),
    ).fetchall()
    return {r[0] for r in rows}


def get_decoded_trial_idxs(conn, encoder_model, prompt_type, seed_set, decoder_model):
    """Return set of trial_idx already decoded by decoder_model."""
    rows = conn.execute(
        """SELECT trial_idx FROM decodes
           WHERE encoder_model = ? AND prompt_type = ? AND seed_set = ?
           AND decoder_model = ? AND decoded_symbol IS NOT NULL""",
        (encoder_model, prompt_type, seed_set, decoder_model),
    ).fetchall()
    return {r[0] for r in rows}


def get_preserved_trials(conn, encoder_model, prompt_type, seed_set):
    """Return list of (trial_idx, encoded_text) for meaning-preserved trials."""
    rows = conn.execute(
        """SELECT trial_idx, encoded_text FROM trials
           WHERE encoder_model = ? AND prompt_type = ? AND seed_set = ?
           AND meaning_preserved = 1 AND encoded_text IS NOT NULL""",
        (encoder_model, prompt_type, seed_set),
    ).fetchall()
    return rows


def get_unjudged_trials(conn, encoder_model, prompt_type, seed_set):
    """Return list of (trial_idx, seed_text, encoded_text) needing judging."""
    rows = conn.execute(
        """SELECT trial_idx, seed_text, encoded_text FROM trials
           WHERE encoder_model = ? AND prompt_type = ? AND seed_set = ?
           AND encoded_text IS NOT NULL AND meaning_preserved IS NULL""",
        (encoder_model, prompt_type, seed_set),
    ).fetchall()
    return rows
