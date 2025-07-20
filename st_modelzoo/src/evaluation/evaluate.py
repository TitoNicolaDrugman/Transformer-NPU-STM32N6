# src/evaluation/evaluate.py
"""
Evaluation utilities for word-level Tiny-BERT generator.
Handles both float (H5) and fully-quantized (TFLite INT8) models.
If the TFLite graph has no Embedding layer, we embed tokens on-the-fly
using the saved embedding_matrix.npy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import tensorflow as tf
from omegaconf import DictConfig

###############################################################################
# Tokeniser helpers
###############################################################################

def _load_tokenizer(tokenizer_path: str | Path) -> Tuple[dict, dict, int]:
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tok = json.load(f)
    word2idx = tok["word2idx"]
    idx2word = {int(k): v for k, v in tok["idx2word"].items()}
    return word2idx, idx2word, tok["vocab_size"]

###############################################################################
# Common helpers
###############################################################################

def _pad_or_crop(seq: list[int], seq_len: int) -> list[int]:
    if len(seq) < seq_len:
        return [0] * (seq_len - len(seq)) + seq
    return seq[-seq_len:]

def _sample_from_logits(logits: np.ndarray, temperature: float) -> int:
    logits = logits.astype(np.float32) / temperature
    return int(tf.random.categorical([logits], 1)[0, 0].numpy())

###############################################################################
# Float-model generation
###############################################################################

def generate_text_h5(
    model: tf.keras.Model,
    tokenizer_path: str | Path,
    seed_text: str,
    gen_length: int,
    seq_len: int,
    temperature: float = 1.0,
) -> str:
    # ban <OOV> tokens
    #word2idx, idx2word, _ = _load_tokenizer(tokenizer_path)
    word2idx, idx2word, _ = _load_tokenizer(tokenizer_path)
    oov_id = word2idx.get("<OOV>", None)
    tokens = [word2idx.get(w, 0) for w in seed_text.split()]
    output = tokens.copy()

    for _ in range(gen_length):
        inp = tf.expand_dims(_pad_or_crop(output, seq_len), 0)
        # ban <OOV> tokens
        #logits = model.predict(inp, verbose=0)[0, -1] / temperature
        #output.append(_sample_from_logits(logits, 1.0))
        logits = model.predict(inp, verbose=0)[0, -1] / temperature
        if oov_id is not None: 
            logits[oov_id] = -1e9
        output.append(_sample_from_logits(logits, 1.0))


    return " ".join(idx2word.get(i, "<OOV>") for i in output)

###############################################################################
# TFLite generation (with optional external embedding)
###############################################################################

def _ids_to_embeddings(
    ids: list[int], emb_matrix: np.ndarray, seq_len: int
) -> np.ndarray:
    ids = _pad_or_crop(ids, seq_len)
    return emb_matrix[ids]                       # (seq_len, emb_dim)

def generate_text_tflite(
    tflite_path: str | Path,
    tokenizer_path: str | Path,
    seed_text: str,
    gen_length: int,
    seq_len: int,
    temperature: float = 1.0,
) -> str:
    # ban <OOV> tokens
    #word2idx, idx2word, _ = _load_tokenizer(tokenizer_path)
    word2idx, idx2word, _ = _load_tokenizer(tokenizer_path)
    oov_id = word2idx.get("<OOV>", None)
    tokens = [word2idx.get(w, 0) for w in seed_text.split()]
    output = tokens.copy()

    # -------------------------------------------------------------------- load model
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inp_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    rank = len(inp_detail["shape"])

    # -------------------------------------------------------------------- optional embedding
    emb_matrix = None
    if rank == 3:                                # model expects embeddings
        emb_path = Path(tflite_path).with_name("embedding_matrix.npy")
        if not emb_path.exists():
            raise FileNotFoundError(
                f"Embedding matrix not found at {emb_path}. "
                "It is required because the TFLite model has no Embedding layer."
            )
        emb_matrix = np.load(emb_path)

    # -------------------------------------------------------------------- loop
    for _ in range(gen_length):
        if rank == 3:            # feed (1, seq_len, emb_dim)
            x = _ids_to_embeddings(output, emb_matrix, seq_len).astype(
                inp_detail["dtype"]
            )
            x = x.reshape(1, seq_len, -1)
        else:                    # feed raw IDs (1, seq_len)
            x = np.asarray(_pad_or_crop(output, seq_len), dtype=inp_detail["dtype"])
            x = x.reshape(1, seq_len)

        interpreter.set_tensor(inp_detail["index"], x)
        interpreter.invoke()
        # ban <OOV> tokens
        #logits = interpreter.get_tensor(out_detail["index"])[0, -1]
        #output.append(_sample_from_logits(logits, temperature))
        logits = interpreter.get_tensor(out_detail["index"])[0, -1]
        if oov_id is not None:  
            logits[oov_id] = -1e9
        output.append(_sample_from_logits(logits, temperature))
    
    return " ".join(idx2word.get(i, "<OOV>") for i in output)

###############################################################################
# Public entrypoint
###############################################################################

def evaluate(
    cfg: DictConfig,
    eval_ds: tf.data.Dataset,          # placeholder for future perplexity
    name_ds: str = "validation",
    model_path_to_evaluate: Optional[str] = None,
) -> None:
    print(f"[INFO] Evaluating on {name_ds} set…")

    sample = cfg.evaluation.sample_text
    length = cfg.evaluation.gen_length
    temp = cfg.evaluation.temperature
    seq_len = cfg.dataset.sequence_length
    tok_path = cfg.preprocessing.tokenizer_path

    if model_path_to_evaluate and model_path_to_evaluate.endswith(".tflite"):
        print(f"[INFO] Running TFLite sample from {model_path_to_evaluate}")
        generated = generate_text_tflite(
            model_path_to_evaluate, tok_path, sample, length, seq_len, temp
        )
    else:
        h5_path = model_path_to_evaluate or cfg.training.best_model_path
        print(f"[INFO] Running H5 sample from {h5_path}")
        model = tf.keras.models.load_model(h5_path)
        generated = generate_text_h5(
            model, tok_path, sample, length, seq_len, temp
        )

    print("\n--- Generated sample ---")
    print(generated)
