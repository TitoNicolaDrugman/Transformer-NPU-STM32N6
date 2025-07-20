# conversion_scripts/convert_to_c_array.py (Version 2 - Robust)
import argparse
import numpy as np
import json
from pathlib import Path

def npy_to_c_header(npy_path: Path, var_name: str, out_path: Path):
    """Converts a float32 .npy file (embedding matrix) to a C header file."""
    arr = np.load(npy_path)
    print(f"Converting {npy_path.name} with shape {arr.shape} and dtype {arr.dtype}...")

    if arr.dtype != np.float32 or len(arr.shape) != 2:
        raise TypeError(f"This function is for float32 embedding matrices. Got {arr.dtype} and shape {arr.shape}")

    with open(out_path, "w", encoding='utf-8') as f:
        f.write(f"#ifndef {var_name.upper()}_H\n")
        f.write(f"#define {var_name.upper()}_H\n\n")

        f.write(f"const float {var_name}[{arr.shape[0]}][{arr.shape[1]}] = {{\n")
        for row in arr:
            f.write("  {")
            f.write(", ".join(f"{x:.6f}f" for x in row))
            f.write("},\n")
        f.write("};\n")

        f.write(f"\n#endif // {var_name.upper()}_H\n")
    print(f"Successfully created {out_path}")

def tflite_to_c_header(tflite_path: Path, var_name: str, out_path: Path):
    """Reads a .tflite file as bytes and converts to a C header file."""
    model_bytes = tflite_path.read_bytes()
    print(f"Converting {tflite_path.name} with size {len(model_bytes)} bytes...")

    with open(out_path, "w", encoding='utf-8') as f:
        f.write(f"#ifndef {var_name.upper()}_H\n")
        f.write(f"#define {var_name.upper()}_H\n\n")

        f.write(f"const unsigned char {var_name}[] = {{\n")
        for i, byte in enumerate(model_bytes):
            f.write(f"0x{byte:02x},")
            if (i + 1) % 16 == 0:
                f.write("\n")
        f.write("\n};\n\n")
        f.write(f"const unsigned int {var_name}_len = {len(model_bytes)};\n")

        f.write(f"\n#endif // {var_name.upper()}_H\n")
    print(f"Successfully created {out_path}")

def tokenizer_to_c_header(json_path: Path, var_name: str, out_path: Path):
    """Converts tokenizer's idx2word to a C string array, with robust character cleaning."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    idx2word = {int(i): w for i, w in data["idx2word"].items()}
    vocab_size = data["vocab_size"]

    lookup_list = [idx2word.get(i, "<UNK>") for i in range(vocab_size)]

    print(f"Converting {json_path.name} with vocab size {vocab_size}...")

    with open(out_path, "w", encoding='utf-8') as f:
        f.write(f"#ifndef {var_name.upper()}_H\n")
        f.write(f"#define {var_name.upper()}_H\n\n")

        f.write(f"const char* {var_name}[{vocab_size}] = {{\n")
        for word in lookup_list:
            # --- START OF ROBUST CLEANING ---
            # 1. Replace common problematic characters with their standard ASCII equivalent
            replacements = {
                '’': "'",  # Right single quote -> Apostrophe
                '‘': "'",  # Left single quote -> Apostrophe
                '“': '"',  # Left double quote -> Standard quote
                '”': '"',  # Right double quote -> Standard quote
                '—': '-',  # Em-dash -> Hyphen
                '–': '-',  # En-dash -> Hyphen
                '…': '...',# Ellipsis -> Three dots
            }
            for old, new in replacements.items():
                word = word.replace(old, new)

            # 2. Escape any backslashes and quotes for C
            c_word = word.replace("\\", "\\\\").replace("\"", "\\\"")
            
            # 3. Final check: remove any remaining non-printable/non-ASCII characters
            # This ensures the string is clean for the C compiler.
            cleaned_word = "".join(c for c in c_word if 31 < ord(c) < 127)
            # --- END OF ROBUST CLEANING ---
            
            f.write(f"  \"{cleaned_word}\",\n")
        f.write("};\n\n")

        f.write(f"#endif // {var_name.upper()}_H\n")
    print(f"Successfully created {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert model files to C headers.")
    parser.add_argument("--tflite", help="Path to the .tflite model file.")
    parser.add_argument("--embedding", help="Path to the .npy embedding matrix.")
    parser.add_argument("--tokenizer", help="Path to the .json tokenizer file.")
    parser.add_argument("--outdir", help="Output directory for header files.", default="c_headers")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(exist_ok=True)

    if args.tflite:
        tflite_path = Path(args.tflite)
        tflite_to_c_header(tflite_path, "g_model_data", out_dir / "model.h")

    if args.embedding:
        embedding_path = Path(args.embedding)
        npy_to_c_header(embedding_path, "g_embedding_matrix", out_dir / "embedding.h")

    if args.tokenizer:
        tokenizer_path = Path(args.tokenizer)
        tokenizer_to_c_header(tokenizer_path, "g_id2word_lookup", out_dir / "tokenizer.h")