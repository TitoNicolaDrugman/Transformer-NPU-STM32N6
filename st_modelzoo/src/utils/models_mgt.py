# src/utils/models_mgt.py

# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  * MODIFIED FOR TEXT GENERATION
#  *--------------------------------------------------------------------------------------------*/

from omegaconf import DictConfig
import tensorflow as tf

# --- MODIFIED: Only import our new model ---
#from src.models import get_tiny_bert_generator
from src.models.tiny_bert_generator import get_tiny_bert_generator


def get_model(cfg: DictConfig = None,
              num_classes: int = None,
              dropout: float = None,
              section: str = None) -> tf.keras.Model:
    """
    Returns a Keras model instance based on the provided configuration.
    Simplified to only handle the text generation model.
    """
    
    # --- MODIFIED: Simplified model dictionary ---
    MODELS_ZOO = {
        "tiny_bert_generator": get_tiny_bert_generator,
    }

    model_name = cfg.name.lower()

    if model_name in MODELS_ZOO:
        # Pass the entire model config dictionary to the model creation function
        model = MODELS_ZOO[model_name](**cfg)
    else:
        raise ValueError(f"Unknown model name: {model_name}. "
                         f"Available models: {list(MODELS_ZOO.keys())}")
    return model


def get_loss(num_classes: int = None) -> tf.keras.losses.Loss:
    """
    Returns the appropriate loss function.
    For text generation, we use SparseCategoricalCrossentropy.
    The original logic for binary vs. categorical for images is kept for reference but simplified.
    """
    if num_classes > 1:
        # This is the standard for our multi-class (vocabulary size) character prediction
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    else:
        # This case is unlikely for text generation but kept for robustness.
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)