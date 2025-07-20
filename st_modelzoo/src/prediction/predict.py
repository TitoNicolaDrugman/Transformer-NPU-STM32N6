# src/prediction/predict.py

# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  * MODIFIED FOR TEXT GENERATION
#  *--------------------------------------------------------------------------------------------*/

import os
import json
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig

def generate_text_tflite(interpreter, tokenizer_data, start_string, num_generate=500, temperature=1.0):
    """
    Generates text using a trained TFLite model.

    Args:
        interpreter: The loaded TFLite interpreter.
        tokenizer_data (dict): Dictionary containing 'char2idx' and 'idx2char'.
        start_string (str): The initial string to seed the generation.
        num_generate (int): The number of characters to generate.
        temperature (float): Controls the randomness of predictions.

    Returns:
        The generated text as a string.
    """
    char2idx = tokenizer_data['char2idx']
    idx2char = {int(k): v for k, v in tokenizer_data['idx2char'].items()}

    # Get input and output details from the TFLite model
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    sequence_length = input_details['shape'][-1]
    input_dtype = input_details['dtype']

    # Vectorize the start string
    input_eval = [char2idx.get(s, 0) for s in start_string]
    
    text_generated = []
    
    for _ in range(num_generate):
        # Pad the current sequence to the model's expected input length
        padded_input = tf.keras.preprocessing.sequence.pad_sequences(
            [input_eval], maxlen=sequence_length, padding='pre'
        )
        
        # Set the tensor and invoke the interpreter
        interpreter.set_tensor(input_details['index'], padded_input.astype(input_dtype))
        interpreter.invoke()
        
        # Get the predictions
        predictions = interpreter.get_tensor(output_details['index'])
        
        # We only care about the prediction for the last character
        predictions = predictions[0, -1, :]
        
        # Apply temperature and sample a token
        if temperature > 0:
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(tf.expand_dims(predictions, 0), num_samples=1)[-1,0].numpy()
        else: # Greedy decoding
            predicted_id = np.argmax(predictions)

        # Update the input sequence for the next iteration
        input_eval = input_eval[1:] + [predicted_id]
        
        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


def predict(cfg: DictConfig) -> None:
    """
    Performs prediction using a quantized TFLite model.
    This function is called when `operation_mode` is set to 'prediction'.
    """
    output_dir = cfg.output_dir
    
    # In prediction mode, we assume a TQE chain has been run and a quantized model exists.
    model_path_to_predict = os.path.join(output_dir, cfg.quantization.export_dir, "quantized_model.tflite")
    
    print(f"\n[INFO] : Running prediction with TFLite model: {model_path_to_predict}")

    if not os.path.exists(model_path_to_predict):
        print(f"[ERROR] : TFLite model not found at {model_path_to_predict}. Please run a training and quantization chain first.")
        return

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path_to_predict)
    interpreter.allocate_tensors()

    # Load the tokenizer
    tokenizer_path = cfg.preprocessing.tokenizer_path
    if not os.path.exists(tokenizer_path):
        print(f"[ERROR] : Tokenizer file not found at {tokenizer_path}. Cannot generate text.")
        return
    with open(tokenizer_path, 'r') as f:
        tokenizer_data = json.load(f)

    # Generate some sample text
    print("[INFO] : Generating sample text with TFLite model...")
    start_string = "JULIET:"
    generated_text = generate_text_tflite(interpreter, tokenizer_data, start_string=start_string)
    
    print("-" * 20)
    print(f"Generated text (seed='{start_string}'):")
    print(generated_text)
    print("-" * 20)
    
    # Save the generated text to a file in the output directory
    pred_output_path = os.path.join(output_dir, "prediction_output.txt")
    with open(pred_output_path, 'w') as f:
        f.write(f"Seed: {start_string}\n\n")
        f.write(generated_text)
    
    print(f"[INFO] : Prediction output saved to {pred_output_path}")