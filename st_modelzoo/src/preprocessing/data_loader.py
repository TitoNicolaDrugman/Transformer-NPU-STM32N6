# src/preprocessing/data_loader.py
# ------------------------------------------------------------------------------
# Word-level data loader for the Tiny-BERT text-generator
# – Builds a capped word tokenizer
# – Generates (input, target) pairs by sliding a window over word-IDs
# – Keeps an extra “quantization” dataset (200 samples) for PTQ
# ------------------------------------------------------------------------------

import os
import json
from typing import Tuple, Dict

import tensorflow as tf


# ------------------------------------------------------------------ tokeniser
def _create_and_save_tokenizer(
    text: str,
    tokenizer_path: str,
    max_vocab_size: int,
    oov_token: str = "<OOV>",
) -> Tuple[Dict, tf.keras.preprocessing.text.Tokenizer]:
    """
    Build a word-level tokenizer limited to `max_vocab_size` most-common tokens,
    save it as a simple JSON dictionary, and return both the dict and the
    tf.keras Tokenizer object.
    """
    tok = tf.keras.preprocessing.text.Tokenizer(
        num_words=max_vocab_size,
        oov_token=oov_token,
        filters="",            # keep punctuation – it helps the model
        lower=False,           # preserve original case
    )
    tok.fit_on_texts([text])

    # Build plain python dicts (MCU-friendly)
    word2idx = {w: i for w, i in tok.word_index.items() if i < max_vocab_size}
    idx2word = {i: w for w, i in word2idx.items()}
    vocab_size = len(word2idx) + 1            # +1 because index 0 is “padding”

    tok_json = {
        "word2idx": word2idx,
        "idx2word": idx2word,
        "vocab_size": vocab_size,
    }

    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
    with open(tokenizer_path, "w", encoding="utf-8") as f:
        json.dump(tok_json, f)

    print(f"[INFO] Tokenizer saved to {tokenizer_path} "
          f"with {vocab_size:,} words (cap={max_vocab_size}).")
    return tok_json, tok


# ------------------------------------------------------------------ dataset
def _build_word_dataset(token_ids, seq_len):
    token_ds = tf.data.Dataset.from_tensor_slices(token_ids)

    # (seq_len+1) because we split into input … target later
    windows = (
        token_ds.window(seq_len + 1, shift=1, drop_remainder=True)
        .flat_map(lambda w: w.batch(seq_len + 1))
    )

    def split(chunk):
        return chunk[:-1], chunk[1:]

    return windows.map(split).cache()


# ------------------------------------------------------------------ public API
def load_text_dataset(
    *,
    corpus_path: str,
    sequence_length: int,
    batch_size: int,
    validation_split: float,
    test_split: float,
    tokenizer_path: str,
    subset_size: int,
    cfg,
) -> Tuple[
    tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset
]:
    """
    Return (train, val, quant, test) datasets – all word-level.
    """
    max_vocab_size = cfg.dataset.get("max_vocab_size", 5000)

    # ---- 1 · tokenise whole corpus ------------------------------------------------
    raw_text = open(corpus_path, "rb").read().decode("utf-8")
    tok_json, tok = _create_and_save_tokenizer(
        raw_text, tokenizer_path, max_vocab_size
    )

    # Expose vocab size to the Hydra config so the model builds correctly
    cfg.training.model.vocab_size = tok_json["vocab_size"]
    print(f"[INFO] training.model.vocab_size set to {tok_json['vocab_size']}.")

    token_ids = tf.constant(
        tok.texts_to_sequences([raw_text])[0], dtype=tf.int64
    )

    # ---- 2 · build dataset --------------------------------------------------------
    dataset = _build_word_dataset(token_ids, sequence_length)

    # Optional small subset for quick experiments
    if subset_size and subset_size > 0:
        dataset = dataset.take(subset_size)
        total = subset_size
        print(f"[INFO] Using a subset of {subset_size} samples.")
    else:
        total = sum(1 for _ in dataset)        # one-pass count
        dataset = _build_word_dataset(token_ids, sequence_length)

    # Shuffle once for split reproducibility
    dataset = dataset.shuffle(10_000, reshuffle_each_iteration=False)

    test_n = int(test_split * total)
    val_n = int(validation_split * total)
    train_n = total - val_n - test_n
    print(f"[INFO] Split → train={train_n}, val={val_n}, test={test_n}")

    test_ds = dataset.take(test_n)
    val_ds = dataset.skip(test_n).take(val_n)
    train_ds = dataset.skip(test_n + val_n)

    # Batch (only training keeps drop_remainder=True for multi-GPU stability)
    train_ds = (
        train_ds.batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        val_ds.batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        test_ds.batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    # 200 single-sentence samples for representative set
    quant_ds = train_ds.unbatch().take(200).batch(1)
    print("[INFO] Quantization dataset: 200 samples.")

    return train_ds, val_ds, quant_ds, test_ds


# -------- façade kept for preprocess.py -----------------------------------------
def load_dataset(dataset_name: str, cfg, **kwargs):
    if dataset_name == "tinyshakespeare":
        return load_text_dataset(
            corpus_path=cfg.dataset.corpus_path,
            sequence_length=cfg.dataset.sequence_length,
            batch_size=cfg.training.batch_size,
            validation_split=cfg.dataset.validation_split,
            test_split=cfg.dataset.test_split,
            tokenizer_path=cfg.preprocessing.tokenizer_path,
            subset_size=cfg.dataset.get("subset_size", 0),
            cfg=cfg,
        )
    raise ValueError(
        "This data loader is configured for 'tinyshakespeare' only. "
        f"Got '{dataset_name}'."
    )
