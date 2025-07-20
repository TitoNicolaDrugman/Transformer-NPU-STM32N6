# deployment/deploy.py

# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  * MODIFIED FOR TEXT GENERATION
#  *--------------------------------------------------------------------------------------------*/

from omegaconf import DictConfig

def deploy(cfg: DictConfig, model_path_to_deploy: str = None, credentials=None) -> None:
    """
    Placeholder for the deployment function.
    The original deployment script was for image classification and is not compatible.
    A new deployment script will be needed to handle text I/O (e.g., via UART).
    """
    print("\n[INFO] : --- SKIPPING DEPLOYMENT ---")
    print("[INFO] : The deployment script is a future step and needs to be custom-built for text generation.")
    print("[INFO] : For now, focus on the generated .tflite model in the experiments output directory.")
    pass

def deploy_mpu(cfg: DictConfig, model_path_to_deploy: str = None, credentials=None) -> None:
    """
    Placeholder for the MPU deployment function.
    """
    print("\n[INFO] : --- SKIPPING MPU DEPLOYMENT ---")
    print("[INFO] : The deployment script is a future step and needs to be custom-built for text generation.")
    pass