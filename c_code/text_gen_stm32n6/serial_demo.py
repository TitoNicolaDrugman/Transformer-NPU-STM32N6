import serial
import struct
import time
import json


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


w2id, id2w = load_tokenizer('tokenizer.json')

ser = serial.Serial('COM3', 115200, timeout=1)
time.sleep(2)  # Wait for the connection to initialize

number = 3178  # Romeo,
data = struct.pack('<I', number)  # Little-endian uint32_t

ser.write(data)
print("Romeo,",end=' ')
"""
for _ in range(128):
    raw = ser.read(4)
    if len(raw) < 4:
        break  # Not enough data received
    value = struct.unpack('<I', raw)[0]
    print( id2w.get(value, "<OOV>"), end=' ')
"""

# to print also the inference time and throghput
times = []
for _ in range(128):
    raw_token = ser.read(4)
    raw_time  = ser.read(4)
    if len(raw_token) < 4 or len(raw_time) < 4:
        break                       

    token_id    = struct.unpack('<I', raw_token)[0]
    elapsed_us  = struct.unpack('<I', raw_time )[0]
    times.append(elapsed_us)

    word = id2w.get(token_id, "<OOV>")
    print(f"{word}({elapsed_us} µs)", end=' ')

# print summary
total_us   = sum(times)
num_tokens = len(times)
avg_us     = total_us / len(times) if times else 0
throughput = num_tokens / (total_us / 1e6)
print(f"\n\nGenerated {len(times)} tokens in {total_us} µs "
       f"({total_us/1e6:.3f} s), avg {avg_us:.1f} µs/token")
print(f"Throughput: {throughput:.2f} tokens/s")




ser.close()