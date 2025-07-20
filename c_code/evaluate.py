#!/usr/bin/env python3
# evaluate.py  –  Top‑1 accuracy & perplexity for the STM32‑N6 text‑gen model
# drop next to cosine.py   © 2025 Tito  (public‑domain / CC0)

import math, json, pathlib, sys, time
from typing import List, Tuple
import numpy as np
import tensorflow as tf

###############################################################################
# CONFIG – edit only if your paths are different
###############################################################################
TOKENIZER_J = pathlib.Path("C:/Users/drugm/stm32ai-modelzoo-services/dd2_text_gen_stm32n6v2/text_gen_stm32n6/tokenizer.json")
VAL_CORPUS  = pathlib.Path("valid.txt")          # validation text or .npy of token IDs
EMBED_BIN   = pathlib.Path("C:/Users/drugm/stm32ai-modelzoo-services/dd2_text_gen_stm32n6v2/text_gen_stm32n6/quantized_embeddings.raw.bin")
TFLITE_MODEL= pathlib.Path("quantized_model.tflite")
SEQ_LEN     = 30                                 # input sequence length
EMB_DIM     = 128                                # embedding dimension
VOCAB       = 20_000                             # vocab size
BATCH_SIZE  = 1                                  # process one window at a time

###############################################################################
# 1. Utils: tokenizer + raw embeddings loader
###############################################################################
def load_tokenizer(path: pathlib.Path) -> Tuple[dict, dict]:
    data = json.loads(path.read_text("utf-8"))
    if "word2idx" in data:
        w2id = {w: int(i) for w, i in data["word2idx"].items()}
        # idx2word keys are strings of IDs
        id2w = {int(k): v for k, v in data.get("idx2word", {}).items() if k.isdigit()}
    elif "char2idx" in data:
        w2id = {c: int(i) for c, i in data["char2idx"].items()}
        id2w = {int(k): v for k, v in data.get("idx2char", {}).items()}
    else:
        idx = data["config"].get("word_index") or data["config"].get("char_index")
        id2w = {int(i): w for w, i in idx.items()}
        w2id = {w: int(i) for i, w in id2w.items()}
    return w2id, id2w


def load_raw_embeddings(path: pathlib.Path) -> np.ndarray:
    """
    Load the int8 embedding table (VOCAB × EMB_DIM) from raw .bin
    """
    raw = np.fromfile(path, dtype=np.int8, count=VOCAB * EMB_DIM)
    return raw.reshape(VOCAB, EMB_DIM)

# load once
E_RAW = load_raw_embeddings(EMBED_BIN)

###############################################################################
# 2. Setup TFLite interpreter (quantized to int8)
###############################################################################
# Load the GPU delegate first, then fall back to CPU
try:
    gpu_delegate = tf.lite.experimental.load_delegate(
        'libtensorflowlite_gpu_delegate.dll'  # on Windows
    )
    _interpreter = tf.lite.Interpreter(
        model_path=str(TFLITE_MODEL),
        experimental_delegates=[gpu_delegate]
    )
    print("INFO: TFLite GPU delegate loaded")
except Exception as e:
    # Fallback to CPU XNNPACK if GPU delegate isn’t available
    print("WARN: GPU delegate failed to load, falling back to CPU:", e)
    _interpreter = tf.lite.Interpreter(
        model_path=str(TFLITE_MODEL),
        num_threads=4
    )


_interpreter.allocate_tensors()
_input_det  = _interpreter.get_input_details()[0]
_output_det = _interpreter.get_output_details()[0]

###############################################################################
# 3. Inference: from tokens → log‑probs tensor
###############################################################################
def model_forward(batch_tokens: np.ndarray) -> np.ndarray:
    """
    batch_tokens: shape (BATCH_SIZE, SEQ_LEN), dtype int32 (token IDs)
    returns: log‑probs shape (BATCH_SIZE, SEQ_LEN, VOCAB)
    """
    # gather quantized embeddings
    inp = E_RAW[batch_tokens]  # (B, SEQ_LEN, EMB_DIM), dtype int8
    # feed directly as int8
    _interpreter.set_tensor(_input_det['index'], inp)
    _interpreter.invoke()

    # fetch quantized logits
    out_q = _interpreter.get_tensor(_output_det['index'])  # int8, (B, SEQ_LEN, VOCAB)
    scale, zp = _output_det['quantization']                # e.g. (scale, zero_point)
    # dequantize to float32 logits
    logits = (out_q.astype(np.float32) - zp) * scale

    # compute log-softmax along last axis
    mx  = np.max(logits, axis=-1, keepdims=True)
    ex  = np.exp(logits - mx)
    sm  = ex / np.sum(ex, axis=-1, keepdims=True)
    return np.log(sm + 1e-9)

###############################################################################
# 4. Evaluation loop: sliding windows → accuracy & perplexity
###############################################################################
def encode(text: str, w2id: dict) -> List[int]:
    unk = w2id.get("<unk>", 1)
    return [w2id.get(tok, unk) for tok in text.split()]


def evaluate(val_tokens: List[int]) -> Tuple[float, float]:
    nll_sum, tok_count, correct = 0.0, 0, 0
    # sliding windows of length SEQ_LEN+1
    windows = np.lib.stride_tricks.sliding_window_view(
        np.array(val_tokens, dtype=np.int32), SEQ_LEN + 1
    )
    total = len(windows)
    for idx in range(total):
        ctx_tgt = windows[idx]          # shape (SEQ_LEN+1,)
        context = ctx_tgt[:SEQ_LEN][None, :]   # (1, SEQ_LEN)
        targets = ctx_tgt[1:][None, :]         # (1, SEQ_LEN)
        logp    = model_forward(context)       # (1, SEQ_LEN, VOCAB)
        # true log‑probs
        rows  = np.arange(SEQ_LEN)[None, :]
        ll    = logp[0, rows, targets[0]]      # (SEQ_LEN,)
        nll_sum   -= ll.sum(dtype=np.float64)
        pred_ids  = np.argmax(logp[0], axis=-1)
        correct  += (pred_ids == targets[0]).sum()
        tok_count += SEQ_LEN
        if (idx + 1) % 1000 == 0:
            print(f"{idx+1}/{total} done…", end='\r', file=sys.stderr)
    ppl = math.exp(nll_sum / tok_count)
    acc = correct / tok_count
    return acc, ppl

###############################################################################
# 5. Main
###############################################################################
if __name__ == "__main__":
    w2id, _ = load_tokenizer(TOKENIZER_J)
    if VAL_CORPUS.suffix == ".npy":
        val_ids = np.load(VAL_CORPUS).tolist()
    else:
        val_ids = encode(VAL_CORPUS.read_text("utf-8"), w2id)

    MAX_TOKENS = 20_000
    if len(val_ids) > MAX_TOKENS:
        val_ids = val_ids[:MAX_TOKENS]
        print(f"Truncated to {len(val_ids):,} tokens for faster eval.")

    print(f"Loaded {len(val_ids):,} validation tokens.")
    t0 = time.perf_counter()
    acc, ppl = evaluate(val_ids)
    dt = time.perf_counter() - t0
    print(f"\naccuracy = {acc:.4%}   perplexity = {ppl:.2f}   ({dt:.1f}s)")
