# check_tflite_2.py
import tensorflow as tf
import argparse
import os

# Set up argument parser
parser = argparse.ArgumentParser(description="Check the operations in a TFLite model.")
parser.add_argument('model_path', type=str, help='The full path to the .tflite model file.')
args = parser.parse_args()

# Check if the file exists
if not os.path.exists(args.model_path):
    print(f"[ERROR] File not found at: {args.model_path}")
    exit()

print(f"[INFO] Analyzing model: {args.model_path}")
interp = tf.lite.Interpreter(model_path=args.model_path)
interp.allocate_tensors()

# _get_ops_details() is private, but works for introspection:
ops = interp._get_ops_details()
op_names = sorted({op["op_name"] for op in ops})
print("Ops in model:", op_names)