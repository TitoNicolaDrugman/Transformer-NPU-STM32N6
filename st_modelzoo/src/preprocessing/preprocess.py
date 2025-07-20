# src/preprocessing/preprocess.py

# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  * MODIFIED FOR TEXT GENERATION
#  *--------------------------------------------------------------------------------------------*/

from omegaconf import DictConfig
import tensorflow as tf
from typing import Tuple

# --- MODIFIED: We only need the new load_dataset function ---
from .data_loader import load_dataset


def preprocess(cfg: DictConfig = None) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Main preprocessing function. For text generation, this is now a simple wrapper
    around the load_dataset function. All image-specific logic has been removed.

    Args:
        cfg (DictConfig): The main Hydra configuration object.

    Returns:
        A tuple containing the training, validation, quantization, and test datasets.
    """
    
    print("[INFO] : Starting text data preprocessing...")

    # --- MODIFIED: Directly call our new data loader ---
    # The new load_dataset handles tokenization, sequence creation, and splitting.
    # We pass the entire config 'cfg' so the data loader can update it with the vocab_size.
    train_ds, valid_ds, quantization_ds, test_ds = load_dataset(
        dataset_name=cfg.dataset.name,
        cfg=cfg
    )

    print("[INFO] : Text data preprocessing complete.")

    return train_ds, valid_ds, quantization_ds, test_ds