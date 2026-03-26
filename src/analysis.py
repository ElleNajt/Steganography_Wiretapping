"""Shared analysis helpers for org-mode src blocks."""
import json, os, re
import numpy as np

ALPHA_DIR = "results/stego_b22c67f_2026-03-25_wikipedia_alphabet"

ENCODERS = {
    "opus_4_6": "Opus 4.6",
    "opus_4_5": "Opus 4.5",
    "sonnet_4": "Sonnet 4",
    "gemini_3_1_pro": "Gemini 3.1 Pro",
    "gemini_flash": "Gemini Flash",
    "gpt-5_4": "GPT-5.4",
}

EAVESDROPPERS = ["Opus 4.6", "Haiku", "Gemini Flash", "Gemini 3.1 Pro", "GPT-5.4"]

EAVES_COLORS = {
    'Opus 4.6': '#4878CF', 'Opus 4.5': '#7B68EE', 'Haiku': '#E24A33',
    'Gemini Flash': '#E8825E', 'Gemini 3.1 Pro': '#2CA02C',
    'GPT-5.4': '#8B5CF6', 'Sonnet 4': '#17BECF',
}

N_SYMBOLS = 26
CHANCE = 1 / N_SYMBOLS


def _encode_ints(arr):
    """Map arbitrary labels to contiguous ints. Returns (int_arr, n_classes)."""
    uniq, inv = np.unique(arr, return_inverse=True)
    return inv, len(uniq)


def mi(symbols, guesses):
    """Mutual information (plugin estimator)."""
    symbols, guesses = np.asarray(symbols), np.asarray(guesses)
    n = len(symbols)
    if n == 0:
        return 0.0
    si, ns = _encode_ints(symbols)
    gi, ng = _encode_ints(guesses)
    joint = np.zeros((ns, ng), dtype=np.int64)
    np.add.at(joint, (si, gi), 1)
    p_joint = joint / n
    p_s = p_joint.sum(axis=1, keepdims=True)
    p_g = p_joint.sum(axis=0, keepdims=True)
    mask = joint > 0
    return np.sum(p_joint[mask] * np.log2(p_joint[mask] / (p_s * p_g)[mask]))


def extract_decoder(data, decoder_label):
    """Extract (symbols, guesses) arrays for a decoder."""
    symbols, guesses = [], []
    for t in data["trials"]:
        if not t or not t.get("meaning_preserved"):
            continue
        dec_entries = t.get("decoders", {}).get(decoder_label, [])
        if not dec_entries:
            continue
        g = dec_entries[0].get("bit")
        if g is not None:
            symbols.append(t["bit"])
            guesses.append(g)
    return np.array(symbols), np.array(guesses)


def extract_paired(data, self_label, eaves_label):
    """Extract paired (symbols, self_guesses, eaves_guesses) for trials where both decoded."""
    symbols, gs, ge = [], [], []
    for t in data["trials"]:
        if not t or not t.get("meaning_preserved"):
            continue
        dec_s = t.get("decoders", {}).get(self_label, [])
        dec_e = t.get("decoders", {}).get(eaves_label, [])
        if not dec_s or not dec_e:
            continue
        g_s = dec_s[0].get("bit")
        g_e = dec_e[0].get("bit")
        if g_s is not None and g_e is not None:
            symbols.append(t["bit"])
            gs.append(g_s)
            ge.append(g_e)
    return np.array(symbols), np.array(gs), np.array(ge)


def _mi_from_counts(joint, n):
    """MI from a joint count matrix (single sample)."""
    p = joint / n
    p_s = p.sum(axis=1, keepdims=True)
    p_g = p.sum(axis=0, keepdims=True)
    mask = joint > 0
    return np.sum(p[mask] * np.log2(p[mask] / (p_s * p_g)[mask]))


def _batch_mi(sym_ints, guess_ints, n_sym, n_guess, indices, n):
    """Compute MI for each row in indices (n_boot x n) using vectorized bincount."""
    n_boot = indices.shape[0]
    # Flatten: encode (sym, guess) pairs as single int for bincount
    n_cells = n_sym * n_guess
    pair_codes = sym_ints * n_guess + guess_ints  # shape (n_data,)
    flat_pairs = pair_codes[indices]  # (n_boot, n)
    # Bincount each bootstrap sample
    joints = np.zeros((n_boot, n_cells), dtype=np.int64)
    for b in range(n_boot):
        joints[b] = np.bincount(flat_pairs[b], minlength=n_cells)
    joints = joints.reshape(n_boot, n_sym, n_guess)
    # Vectorized MI across bootstrap dimension
    p = joints / n
    p_s = p.sum(axis=2, keepdims=True)  # (n_boot, n_sym, 1)
    p_g = p.sum(axis=1, keepdims=True)  # (n_boot, 1, n_guess)
    mask = joints > 0
    log_term = np.zeros_like(p)
    denom = p_s * p_g
    valid = mask & (denom > 0)
    log_term[valid] = p[valid] * np.log2(p[valid] / denom[valid])
    return log_term.sum(axis=(1, 2))  # (n_boot,)


def bootstrap_delta(symbols, gs, ge, n_boot=10000):
    """Bootstrap CI for secrecy capacity delta = MI(self) - MI(eaves).

    Uses percentile bootstrap on the delta. The observed value is the
    point estimate; lo/hi are the 2.5th/97.5th percentiles of the
    bootstrap distribution of deltas.
    """
    n = len(symbols)
    observed = mi(symbols, gs) - mi(symbols, ge)
    all_vals = np.unique(np.concatenate([symbols, gs, ge]))
    val_map = {v: i for i, v in enumerate(all_vals)}
    si = np.array([val_map[v] for v in symbols])
    gsi = np.array([val_map[v] for v in gs])
    gei = np.array([val_map[v] for v in ge])
    n_vals = len(all_vals)
    rng = np.random.default_rng(42)
    indices = rng.choice(n, (n_boot, n), replace=True)
    mi_self = _batch_mi(si, gsi, n_vals, n_vals, indices, n)
    mi_eaves = _batch_mi(si, gei, n_vals, n_vals, indices, n)
    deltas = mi_self - mi_eaves
    return observed, np.percentile(deltas, 2.5), np.percentile(deltas, 97.5)


def paired_bootstrap_pvalue(symbols_s, guesses_s, symbols_e, guesses_e, n_boot=10000):
    """Bootstrap p-value for H0: MI(self) <= MI(eaves)."""
    n = min(len(symbols_s), len(symbols_e))
    symbols_s = np.asarray(symbols_s[:n])
    guesses_s = np.asarray(guesses_s[:n])
    symbols_e = np.asarray(symbols_e[:n])
    guesses_e = np.asarray(guesses_e[:n])
    observed = mi(symbols_s, guesses_s) - mi(symbols_e, guesses_e)
    all_vals = np.unique(np.concatenate([symbols_s, guesses_s, symbols_e, guesses_e]))
    val_map = {v: i for i, v in enumerate(all_vals)}
    ssi = np.array([val_map[v] for v in symbols_s])
    gsi = np.array([val_map[v] for v in guesses_s])
    sei = np.array([val_map[v] for v in symbols_e])
    gei = np.array([val_map[v] for v in guesses_e])
    n_vals = len(all_vals)
    rng = np.random.default_rng(42)
    indices = rng.choice(n, (n_boot, n), replace=True)
    mi_self = _batch_mi(ssi, gsi, n_vals, n_vals, indices, n)
    mi_eaves = _batch_mi(sei, gei, n_vals, n_vals, indices, n)
    deltas = mi_self - mi_eaves
    return observed, np.mean(deltas <= 0)


def load_all(data_dir=None):
    """Load all encoder result files. Returns {label: data}."""
    if data_dir is None:
        data_dir = ALPHA_DIR
    all_data = {}
    for fname, label in ENCODERS.items():
        path = os.path.join(data_dir, f"{fname}.json")
        if os.path.exists(path):
            with open(path) as f:
                all_data[label] = json.load(f)
    return all_data


def acc_and_n(data, decoder_label):
    """Return (accuracy_pct, n) for a decoder."""
    symbols, guesses = extract_decoder(data, decoder_label)
    if len(symbols) == 0:
        return None, 0
    correct = np.sum(symbols == guesses)
    return 100 * correct / len(symbols), len(symbols)


def ci95(acc_pct, n):
    """95% CI half-width for accuracy."""
    if n == 0:
        return 0
    p = acc_pct / 100
    return 1.96 * np.sqrt(p * (1 - p) / n) * 100
