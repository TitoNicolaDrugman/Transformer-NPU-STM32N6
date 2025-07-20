import json
import numpy as np
import tensorflow as tf   # needs pip install tflite-runtime or tensorflow

# --- path configurables ------------------------------------------------------
MODEL_PATH   = r"C:\Users\drugm\stm32ai-modelzoo-services\text_generation_word\src\experiments_outputs\2025_06_28_21_19_37\quantized_models\quantized_model.tflite"
SAVE_AS      = r"C:\Users\drugm\stm32ai-code\val.json"
NB_SAMPLES   = 1                     import json
import numpy as np

# 1) point this at your .tflite
model_path = r"C:\Users\drugm\stm32ai-modelzoo-services\text_generation_word\src\experiments_outputs\2025_06_28_21_19_37\quantized_models\quantized_model.tflite"

# 2) where to write the validation JSON
output_path = r"C:\Users\drugm\stm32ai-code\val.json"

# Try imports
try:
    from tensorflow.lite.python.interpreter import Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter

# Build the interpreter, allocate, inspect inputs
interp = Interpreter(model_path=model_path)
interp.allocate_tensors()
input_details = interp.get_input_details()

# Create a single validation entry with zeros
entry = {"inputs": {}}
for d in input_details:
    name = d["name"]
    shape = d["shape"].tolist() if hasattr(d["shape"], "tolist") else list(d["shape"])
    dtype = d["dtype"]
    entry["inputs"][name] = np.zeros(shape, dtype=dtype).flatten().tolist()

# Write out
with open(output_path, "w") as f:
    json.dump([entry], f, indent=2)

print(f"Wrote validation JSON to: {output_path}")
                    # create N dummy samples
# -----------------------------------------------------------------------------

# 1. discover tensor names & shapes
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
in_spec  = interpreter.get_input_details()[0]
out_spec = interpreter.get_output_details()[0]

in_name,  in_shape  = in_spec["name"],  in_spec["shape"]  # e.g. ('serving_default_em.ed_tokens0', [ 1 30 128])
out_name, out_shape = out_spec["name"], out_spec["shape"] # e.g. ('eltwise_534',               [ 1 30 20000])

# 2. build dummy data (all zeros); replace with real test-vectors if you have them
samples = []
for _ in range(NB_SAMPLES):
    sample = {
        "inputs":  { in_name:  np.zeros(in_shape,  dtype=np.int8).flatten().tolist() },
        # "outputs": { out_name: np.zeros(out_shape, dtype=np.int8).flatten().tolist() }  # optional
    }
    samples.append(sample)

# 3. dump
with open(SAVE_AS, "w") as fp:
    json.dump(samples, fp, indent=2)

print(f"val.json written to {SAVE_AS}")
