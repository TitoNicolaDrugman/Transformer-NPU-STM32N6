# src/utils/parse_config.py

# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  * MODIFIED FOR TEXT GENERATION
#  *--------------------------------------------------------------------------------------------*/

from omegaconf import DictConfig, OmegaConf
import os
from munch import DefaultMunch # <-- Add this import

def get_config(cfg: DictConfig = None) -> DefaultMunch:
    """
    Parses and validates the configuration file.
    Also converts the OmegaConf object to a Munch object for compatibility
    with the rest of the framework's helper functions (e.g., get_optimizer).
    
    Args:
        cfg (DictConfig): The raw configuration object from Hydra.

    Returns:
        DefaultMunch: The processed configuration object, compatible with the framework.
    """

    print("[INFO] : Parsing configuration...")

    # Validate essential paths
    if not cfg.dataset.corpus_path:
        raise ValueError("Configuration error: `dataset.corpus_path` must be specified.")
    
    if not os.path.exists(cfg.dataset.corpus_path):
        raise FileNotFoundError(f"The specified corpus file was not found at: {cfg.dataset.corpus_path}")

    # --- ADDED CONVERSION LOGIC ---
    # Convert the OmegaConf DictConfig to a Munch object.
    # The original framework's utility functions (like get_optimizer) expect this type.
    plain_dict = OmegaConf.to_container(cfg, resolve=True)
    munch_cfg = DefaultMunch.fromDict(plain_dict)
    # --- END OF ADDED LOGIC ---

    print("[INFO] : Configuration parsing complete.")
    return munch_cfg # <-- Return the munch object instead of the original cfg