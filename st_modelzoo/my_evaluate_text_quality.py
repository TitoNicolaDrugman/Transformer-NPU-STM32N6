#!/usr/bin/env python
"""
my_evaluate_text_quality.py  —  WORD-level edition
--------------------------------------------------
Generate text (or compute word-level perplexity) from a *quantised* STM32-ready
`.tflite` model that takes **pre-embedded int8 vectors**.

✓ Works with tokenizer JSON that has  ➜  word2idx / idx2word / vocab_size
✓ Auto-infers  seq_len  and  emb_dim  from the TFLite graph & embedding.npy
✓ Handles TFLite input layouts (1,seq,emb)  or  (1,seq,1,emb)

USAGE (PowerShell example — backtick = line-continuation):

python my_evaluate_text_quality.py `
    --model      path\quantized_model.tflite `
    --embedding  path\embedding_matrix.npy `
    --tokenizer  path\tokenizer.json `
    --prompt     "ROMEO:" `
    --steps      40 `
    --temperature 1.0
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # CPU only

# --------------------------------------------------------------------- helpers
def load_tokenizer(json_path: str):
    """Return (word→id dict, id→word dict). Accepts both old & new formats."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "word2idx" in data:                         # new word-level
        w2id = {w: int(i) for w, i in data["word2idx"].items()}
        id2w = {int(i): w for i, w in data["idx2word"].items()}
    elif "char2idx" in data:                       # fallback (old char-level)
        w2id = {c: int(i) for c, i in data["char2idx"].items()}
        id2w = {int(i): c for i, c in data["idx2char"].items()}
    else:                                          # Keras Tokenizer backup
        idx = data.get("config", {}).get("word_index") or \
              data.get("config", {}).get("char_index")
        if not idx:
            raise ValueError("Unrecognised tokenizer JSON structure")
        id2w = {int(i): w for w, i in idx.items()}
        w2id = {w: int(i) for i, w in id2w.items()}

    return w2id, id2w


def quantize(x: np.ndarray, scale: float, zp: int) -> np.ndarray:
    return np.clip(np.round(x / scale + zp), -128, 127).astype(np.int8)


def softmax_temperature(logits: np.ndarray, temp: float = 1.0) -> np.ndarray:
    logits = logits.astype(np.float32) / temp
    logits -= logits.max()                         # for stability
    probs = np.exp(logits)
    return probs / probs.sum()

# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",      required=True, help=".tflite model path")
    ap.add_argument("--embedding",  required=True, help=".npy embedding path")
    ap.add_argument("--tokenizer",  required=True, help="tokenizer JSON path")
    ap.add_argument("--prompt",     default="ROMEO:")
    ap.add_argument("--steps",      type=int, default=40,
                    help="# new words to generate")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--test-file",  help="txt – compute word-level perplexity")
    args = ap.parse_args()

    # ---------- load artefacts ------------------------------------------------
    w2id, id2w = load_tokenizer(args.tokenizer)
    # ban <OOV> tokens
    oov_id = w2id.get("<OOV>", None)
    embedding  = np.load(args.embedding)           # (vocab, emb_dim)

    intr = tf.lite.Interpreter(model_path=args.model)
    intr.allocate_tensors()
    tin        = intr.get_input_details()[0]
    tout       = intr.get_output_details()[0]
    in_scale, in_zp   = tin["quantization"]
    out_scale, out_zp = tout["quantization"]

    print("\n--- C CODE QUANTIZATION PARAMETERS ---")
    print(f"const float in_scale = {in_scale:.8f}f;")
    print(f"const int32_t in_zp = {in_zp};")
    print(f"const float out_scale = {out_scale:.8f}f;")
    print(f"const int32_t out_zp = {out_zp};")
    print("--------------------------------------\n")

    # derive sequence length & embedding dim from tensors
    if len(tin["shape"]) == 4:                     # (1,seq,1,emb)
        SEQ_LEN, EMB_DIM = tin["shape"][1], tin["shape"][3]
        def reshape_in(q_emb): return q_emb[np.newaxis, :, np.newaxis, :]
        def slice_out(out):  return out[0, -1, 0]
    else:                                          # (1,seq,emb)
        SEQ_LEN, EMB_DIM = tin["shape"][1], tin["shape"][2]
        def reshape_in(q_emb): return q_emb[np.newaxis, :, :]
        def slice_out(out):  return out[0, -1]

    # sanity check
    if EMB_DIM != embedding.shape[1]:
        sys.exit(f"Embedding dim mismatch: emb.npy has {embedding.shape[1]}, "
                 f"but model expects {EMB_DIM}")

    # ---------- forward pass --------------------------------------------------
    def forward(token_ids: np.ndarray):
        """Run whole sequence → float32 logits for last position."""
        emb      = embedding[token_ids]            # (seq, emb_dim)
        q_emb    = quantize(emb, in_scale, in_zp)
        intr.set_tensor(tin["index"], reshape_in(q_emb))
        intr.invoke()
        logits_q = slice_out(intr.get_tensor(tout["index"]))
        return (logits_q.astype(np.float32) - out_zp) * out_scale

    # ---------- perplexity mode ----------------------------------------------
    if args.test_file:
        txt  = Path(args.test_file).read_text(encoding="utf-8")
        ids  = [w2id.get(w, 1) for w in txt.split()]  # OOV=1
        buf  = ids[:SEQ_LEN]
        nll, count = 0.0, 0
        for wid in ids[SEQ_LEN:]:
            logits = forward(np.array(buf, dtype=np.int64))
            probs  = softmax_temperature(logits, 1.0)
            nll   += -np.log(probs[wid])
            count += 1
            buf    = buf[1:] + [wid]
        ppl = np.exp(nll / count)
        print(f"Word-level perplexity on {args.test_file}: {ppl:.3f} (n={count})")
        return

    # ---------- generation mode ----------------------------------------------
    buf = [w2id.get(w, 1) for w in args.prompt.split()][-SEQ_LEN:]
    while len(buf) < SEQ_LEN:
        buf.insert(0, 0)                           # left-pad with <PAD>=0

    out_tokens = args.prompt.split()
    for _ in range(args.steps):
        logits  = forward(np.array(buf, dtype=np.int64))
        # ban <OOV> tokens
        #probs   = softmax_temperature(logits, args.temperature)
        #next_id = int(np.random.choice(len(probs), p=probs))
        probs = softmax_temperature(logits, args.temperature)
        if oov_id is not None:
            probs[oov_id] = 0.0          # mask out <OOV>
            probs /= probs.sum()         # renormalise
        next_id = int(np.random.choice(len(probs), p=probs))
        out_tokens.append(id2w.get(next_id, "<OOV>"))
        buf = buf[1:] + [next_id]

    print("------------ GENERATED ---------------")
    print(" ".join(out_tokens))
    print("--------------------------------------")

if __name__ == "__main__":
    main()
