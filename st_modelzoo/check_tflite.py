#!/usr/bin/env python3
"""
Usage:
    python check_tflite.py path/to/quantized_model.tflite

Prints:
    RANDOM_UNIFORM present? True/False
"""
import sys
from pathlib import Path
import tflite

# Builtin code for RANDOM_UNIFORM
RANDOM_UNIFORM = 129

def has_random_uniform(tflite_path: str) -> bool:
    buf = Path(tflite_path).read_bytes()
    model = tflite.Model.GetRootAsModel(buf, 0)
    codes = {
        model.OperatorCodes(i).BuiltinCode()
        for i in range(model.OperatorCodesLength())
    }
    return RANDOM_UNIFORM in codes

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_tflite.py path/to/model.tflite")
        sys.exit(1)
    present = has_random_uniform(sys.argv[1])
    print(f"RANDOM_UNIFORM present? {present}")
    sys.exit(0 if not present else 2)
