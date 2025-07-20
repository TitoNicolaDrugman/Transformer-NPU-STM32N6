import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Union

PathLike = Union[str, Path]

###############################################################################
# CONFIG – edit these paths if you keep the files elsewhere
###############################################################################
EMBED_FILE   = Path("C:/Users/drugm/stm32ai-modelzoo-services/dd2_text_gen_stm32n6v2/"
                    "text_gen_stm32n6/quantized_embeddings.raw.bin")
TOKENIZER_J  = Path("C:/Users/drugm/stm32ai-modelzoo-services/dd2_text_gen_stm32n6v2/"
                    "text_gen_stm32n6/tokenizer.json")
DIM          = 128          # embedding dimension
VOCAB        = 20_000       # number of rows in the matrix

###############################################################################
# LOAD TOKEN MAPS
###############################################################################
def load_tokenizer(path: PathLike) -> Tuple[dict, dict]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if "word2idx" in data:             # new word‑level
        w2id = {w: int(i) for w, i in data["word2idx"].items()}
        id2w = {int(i): w for i, w in data["idx2word"].items()}
    elif "char2idx" in data:           # legacy char‑level
        w2id = {c: int(i) for c, i in data["char2idx"].items()}
        id2w = {int(i): c for i, c in data["idx2char"].items()}
    else:                              # Keras fallback
        idx = data["config"].get("word_index") or data["config"].get("char_index")
        id2w = {int(i): w for w, i in idx.items()}
        w2id = {w: int(i) for i, w in id2w.items()}
    return w2id, id2w


###############################################################################
# LOAD QUANTISED EMBEDDINGS
###############################################################################
def load_embeddings(path: PathLike, vocab: int, dim: int) -> np.ndarray:
    p = Path(path)
    raw = np.fromfile(p, dtype=np.int8, count=vocab * dim).astype(np.float32)
    return raw.reshape(vocab, dim)


###############################################################################
# COSINE SIMILARITY UTILITIES
###############################################################################
def cosine(v: np.ndarray, w: np.ndarray) -> float:
    return float(v @ w / (np.linalg.norm(v) * np.linalg.norm(w) + 1e-9))


def most_similar(
    query_vec: np.ndarray,
    all_vecs: np.ndarray,
    top_k: int = 10,
    exclude: List[int] = ()
) -> List[Tuple[int, float]]:
    dots = all_vecs @ query_vec
    norms = np.linalg.norm(all_vecs, axis=1) * np.linalg.norm(query_vec) + 1e-9
    sims = dots / norms
    sims[list(exclude)] = -np.inf
    best_ids = np.argpartition(-sims, range(top_k))[:top_k]
    return sorted([(int(i), float(sims[i])) for i in best_ids], key=lambda x: -x[1])


###############################################################################
# SIMPLE CLI
###############################################################################
if __name__ == "__main__":
    w2id, id2w = load_tokenizer(TOKENIZER_J)
    E          = load_embeddings(EMBED_FILE, VOCAB, DIM)
    print("→ loaded matrix", E.shape, "and", len(w2id), "tokens\n")

    while True:
        try:
            query = input("word or word1,word2 (quit=Q): ").strip()
            if query.lower() == "q":
                break

            if "," in query:
                w1, w2 = [s.strip() for s in query.split(",", 1)]
                for w in (w1, w2):
                    if w not in w2id:
                        raise KeyError(f"token '{w}' not in vocab")
                sim = cosine(E[w2id[w1]], E[w2id[w2]])
                print(f"cosine({w1}, {w2}) = {sim:.4f}\n")
            else:
                if query not in w2id:
                    raise KeyError(f"token '{query}' not in vocab")
                qid = w2id[query]
                neighbours = most_similar(E[qid], E, top_k=10, exclude=[qid])
                print("top-10 similar tokens:")
                for rank, (tid, score) in enumerate(neighbours, 1):
                    print(f"{rank:2}. {id2w[tid]:<15}  {score:.4f}")
                print()

        except Exception as e:
            print("error:", e, "\n")
