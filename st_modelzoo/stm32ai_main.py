# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
print("[INFO] : Programmatically set CUDA_VISIBLE_DEVICES='0,1'")


from hydra.core.hydra_config import HydraConfig
import hydra
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from omegaconf import DictConfig, OmegaConf
from omegaconf import DictConfig
import mlflow
import argparse
import logging
from typing import Optional
from clearml import Task
from clearml.backend_config.defs import get_active_config_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

def setup_gpus():
    """
    Checks for available GPUs and sets memory growth to avoid allocating all memory at once.
    This is the recommended best practice for multi-GPU setup.
    """
    print("[INFO] : Setting up GPU configuration...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth for each GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] : Successfully configured {len(gpus)} GPUs with memory growth.")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(f"[ERROR] : GPU setup failed: {e}")
    else:
        print("[WARNING] : No GPUs found by TensorFlow. Running on CPU.")


from common.utils import mlflow_ini, set_gpu_memory_limit, get_random_seed, display_figures, log_to_file
from common.benchmarking import benchmark, cloud_connect
from common.evaluation import gen_load_val
from common.prediction import gen_load_val_predict
from src.preprocessing import preprocess
from src.utils import get_config
from src.training import train
from src.evaluation import evaluate
from src.quantization import quantize
from src.prediction import predict
from deployment import deploy, deploy_mpu



def chain_qd(cfg: DictConfig = None, quantization_ds: tf.data.Dataset = None, hardware_type: str = "MCU") -> None:
    """
    Runs the chain_qd pipeline, including quantization, and deployment

    Args:
        cfg (DictConfig): Configuration dictionary. Defaults to None.
        quantization_ds:(tf.data.Dataset): quantization dataset. Defaults to None
        hardware_type (str): parameter to specify a target on which to deploy

    Returns:
        None
    """

    # Connect to STM32Cube.AI Developer Cloud
    credentials = None
    if cfg.tools.stm32ai.on_cloud:
        _, _, credentials = cloud_connect(stm32ai_version=cfg.tools.stm32ai.version)

    # whether data are coming from train set or quantization set, they end up in quantization_ds
    #source_image = cfg.dataset.quantization_path if cfg.dataset.quantization_path else cfg.dataset.training_path
    #source_image = source_image if source_image else "random generation"
    print('[INFO] : Quantization using representative data from the training corpus.')
    #print('[INFO] : Quantization using input images coming from {}'.format(source_image))
    quantized_model_path = quantize(cfg=cfg, quantization_ds=quantization_ds)
    print('[INFO] : Quantization complete.')

    if hardware_type == "MCU":
        deploy(cfg=cfg, model_path_to_deploy=quantized_model_path, credentials=credentials)
    else:
        print("MPU DEPLOYMENT")
        deploy_mpu(cfg=cfg, model_path_to_deploy=quantized_model_path, credentials=credentials)
    print('[INFO] : Deployment complete.')
    if cfg.deployment.hardware_setup.board == "STM32N6570-DK":
        print('[INFO] : Please on STM32N6570-DK toggle the boot switches to the left and power cycle the board.')


def chain_qb(cfg: DictConfig = None, quantization_ds: tf.data.Dataset = None) -> None:
    """
    Runs the chain_qb pipeline, including quantization and benchmarking.

    Args:
        cfg (DictConfig): Configuration dictionary. Defaults to None.
        quantization_ds:(tf.data.Dataset): quantization dataset. Defaults to None

    Returns:
        None
    """

    # Connect to STM32Cube.AI Developer Cloud
    credentials = None
    if cfg.tools.stm32ai.on_cloud:
        _, _, credentials = cloud_connect(stm32ai_version=cfg.tools.stm32ai.version)

    # whether data are coming from train set or quantization set, they end up in quantization_ds
    #source_image = cfg.dataset.quantization_path if cfg.dataset.quantization_path else cfg.dataset.training_path
    #source_image = source_image if source_image else "random generation"
    #print('[INFO] : Quantization using input images coming from {}'.format(source_image))
    print('[INFO] : Quantization using representative data from the training corpus.')
    quantized_model_path = quantize(cfg=cfg, quantization_ds=quantization_ds)
    print('[INFO] : Quantization complete.')
    benchmark(cfg=cfg, model_path_to_benchmark=quantized_model_path, credentials=credentials)
    print('[INFO] : Benchmarking complete.')


def chain_eqe(cfg: DictConfig = None, valid_ds: tf.data.Dataset = None, quantization_ds: tf.data.Dataset = None,
              test_ds: tf.data.Dataset = None) -> str:
    """
    Runs the chain_eqe pipeline, including evaluation of a float model, quantization and evaluation of
    the quantized model

    Args:
        cfg (DictConfig): Configuration dictionary. Defaults to None.
        valid_ds (tf.data.Dataset): Validation dataset. Defaults to None.
        test_ds (tf.data.Dataset): Test dataset. Defaults to None.
        quantization_ds:(tf.data.Dataset): quantization dataset. Defaults to None

    Returns:
        quantized_model_path (str): path to quantized model
    """

    if test_ds:
        evaluate(cfg=cfg, eval_ds=test_ds, name_ds="test_set")
    else:
        evaluate(cfg=cfg, eval_ds=valid_ds, name_ds="validation_set")
    print('[INFO] : Evaluation complete.')
    display_figures(cfg)

    # whether data are coming from train set or quantization set, they end up in quantization_ds
    #source_image = cfg.dataset.quantization_path if cfg.dataset.quantization_path else cfg.dataset.training_path
    #source_image = source_image if source_image else "random generation"
    #print('[INFO] : Quantization using input images coming from {}'.format(source_image))
    print('[INFO] : Quantization using representative data from the training corpus.')
    quantized_model_path = quantize(cfg=cfg, quantization_ds=quantization_ds)
    print('[INFO] : Quantization complete.')

    if test_ds:
        evaluate(cfg=cfg, eval_ds=test_ds, model_path_to_evaluate=quantized_model_path, name_ds="test_set")
    else:
        evaluate(cfg=cfg, eval_ds=valid_ds, model_path_to_evaluate=quantized_model_path, name_ds="validation_set")
    print('[INFO] : Evaluation complete.')
    display_figures(cfg)

    return quantized_model_path


def chain_eqeb(cfg: DictConfig = None, valid_ds: tf.data.Dataset = None, quantization_ds: tf.data.Dataset = None,
               test_ds: tf.data.Dataset = None) -> None:
    """
    Runs the chain_eqeb pipeline, including evaluation of the float model, quantization, evaluation of
    the quantized model and benchmarking

    Args:
        cfg (DictConfig): Configuration dictionary. Defaults to None.
        valid_ds (tf.data.Dataset): Validation dataset. Defaults to None.
        test_ds (tf.data.Dataset): Test dataset. Defaults to None.
        quantization_ds:(tf.data.Dataset): quantization dataset. Defaults to None

    Returns:
        None
    """

    # Connect to STM32Cube.AI Developer Cloud
    credentials = None
    if cfg.tools.stm32ai.on_cloud:
        _, _, credentials = cloud_connect(stm32ai_version=cfg.tools.stm32ai.version)

    quantized_model_path = chain_eqe(cfg=cfg, valid_ds=valid_ds, quantization_ds=quantization_ds, test_ds=test_ds)

    benchmark(cfg=cfg, model_path_to_benchmark=quantized_model_path, credentials=credentials)
    print('[INFO] : Benchmarking complete.')


def chain_tqe(cfg: DictConfig = None, train_ds: tf.data.Dataset = None, valid_ds: tf.data.Dataset = None,
              quantization_ds: tf.data.Dataset = None, test_ds: tf.data.Dataset = None) -> str:
    """
    Runs the chain_tqe pipeline, including training, quantization and evaluation.

    Args:
        cfg (DictConfig): Configuration dictionary. Defaults to None.
        train_ds (tf.data.Dataset): Training dataset. Defaults to None.
        valid_ds (tf.data.Dataset): Validation dataset. Defaults to None.
        test_ds (tf.data.Dataset): Test dataset. Defaults to None.
        quantization_ds:(tf.data.Dataset): quantization dataset. Defaults to None

    Returns:
        quantized_model_path (str): path to model quantized
    """
    if test_ds:
        trained_model_path = train(cfg=cfg, train_ds=train_ds, valid_ds=valid_ds, test_ds=test_ds)
    else:
        trained_model_path = train(cfg=cfg, train_ds=train_ds, valid_ds=valid_ds)
    print('[INFO] : Training complete.')

    # whether data are coming from train set or quantization set, they end up in quantization_ds
    #source_image = cfg.dataset.quantization_path if cfg.dataset.quantization_path else cfg.dataset.training_path
    #source_image = source_image if source_image else "random generation"
    #print('[INFO] : Quantization using input images coming from {}'.format(source_image))
    print('[INFO] : Quantization using representative data from the training corpus.')
    quantized_model_path = quantize(cfg=cfg, quantization_ds=quantization_ds, float_model_path=trained_model_path)
    print('[INFO] : Quantization complete.')

    if test_ds:
        evaluate(cfg=cfg, eval_ds=test_ds, model_path_to_evaluate=quantized_model_path, name_ds="test_set")
    else:
        evaluate(cfg=cfg, eval_ds=valid_ds, model_path_to_evaluate=quantized_model_path, name_ds="validation_set")
    print('[INFO] : Evaluation complete.')
    display_figures(cfg)

    return quantized_model_path


def chain_tqeb(cfg: DictConfig = None, train_ds: tf.data.Dataset = None, valid_ds: tf.data.Dataset = None,
               quantization_ds: tf.data.Dataset = None, test_ds: tf.data.Dataset = None) -> None:
    """
    Runs the chain_tqeb pipeline, including training, quantization, evaluation and benchmarking.

    Args:
        cfg (DictConfig): Configuration dictionary. Defaults to None.
        train_ds (tf.data.Dataset): Training dataset. Defaults to None.
        valid_ds (tf.data.Dataset): Validation dataset. Defaults to None.
        test_ds (tf.data.Dataset): Test dataset. Defaults to None.
        quantization_ds:(tf.data.Dataset): quantization dataset. Defaults to None

    Returns:
        None
    """

    # Connect to STM32Cube.AI Developer Cloud
    credentials = None
    if cfg.tools.stm32ai.on_cloud:
        _, _, credentials = cloud_connect(stm32ai_version=cfg.tools.stm32ai.version)

    quantized_model_path = chain_tqe(cfg=cfg, train_ds=train_ds, valid_ds=valid_ds, quantization_ds=quantization_ds,
                                     test_ds=test_ds)

    benchmark(cfg=cfg, model_path_to_benchmark=quantized_model_path, credentials=credentials)
    print('[INFO] : Benchmarking complete.')


def process_mode(mode: str = None,
                 configs: DictConfig = None,
                 train_ds: tf.data.Dataset = None,
                 valid_ds: tf.data.Dataset = None,
                 quantization_ds: tf.data.Dataset = None,
                 test_ds: tf.data.Dataset = None) -> None:
    """
    Process the selected mode of operation.
    """
    mlflow.log_param("model_path", configs.general.model_path)
    log_to_file(configs.output_dir, f'operation_mode: {mode}')
    
    # --- FIX: Check if we are running the text-gen project before attempting deployment ---
    is_text_gen_project = 'text_generation' in configs.general.project_name

    # Check the selected mode and perform the corresponding operation
    if mode == 'training':
        if test_ds:
            train(cfg=configs, train_ds=train_ds, valid_ds=valid_ds, test_ds=test_ds)
        else:
            train(cfg=configs, train_ds=train_ds, valid_ds=valid_ds)
        display_figures(configs)
        print('[INFO] : Training complete.')
    elif mode == 'evaluation':
        gen_load_val(cfg=configs)
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        if test_ds:
            evaluate(cfg=configs, eval_ds=test_ds, name_ds="test_set")
        else:
            evaluate(cfg=configs, eval_ds=valid_ds, name_ds="validation_set")
        display_figures(configs)
        print('[INFO] : Evaluation complete.')
    elif mode == 'deployment':
        # --- FIX: Add the check here ---
        if is_text_gen_project:
            print("[WARNING] : Deployment step is skipped for text generation project.")
            print("[INFO] : The C-code implementation for text I/O needs to be handled separately.")
        else:
            if configs.hardware_type == "MPU":
                print("MPU_DEPLOYMENT")
                deploy_mpu(cfg=configs)
            else:
                deploy(cfg=configs)
            print('[INFO] : Deployment complete.')
            if configs.deployment.hardware_setup.board == "STM32N6570-DK":
                print('[INFO] : Please on STM32N6570-DK toggle the boot switches to the left and power cycle the board.')
    elif mode == 'quantization':
        source_image = configs.dataset.quantization_path if configs.dataset.quantization_path \
            else configs.dataset.training_path
        source_image = source_image if source_image else "random generation"
        print('[INFO] : Quantization using input images coming from {}'.format(source_image))
        quantize(cfg=configs, quantization_ds=quantization_ds)
        print('[INFO] : Quantization complete.')
    elif mode == 'prediction':
        gen_load_val_predict(cfg=configs)
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        predict(cfg=configs)
        print('[INFO] : Prediction complete.')
    elif mode == 'benchmarking':
        benchmark(cfg=configs)
        print('[INFO] : Benchmark complete.')
    elif mode == 'chain_tqe':
        chain_tqe(cfg=configs, train_ds=train_ds, valid_ds=valid_ds, quantization_ds=quantization_ds, test_ds=test_ds)
        print('[INFO] : chain_tqe complete.')
    elif mode == 'chain_tqeb':
        chain_tqeb(cfg=configs, train_ds=train_ds, valid_ds=valid_ds, quantization_ds=quantization_ds, test_ds=test_ds)
        print('[INFO] : chain_tqeb complete.')
    elif mode == 'chain_eqe':
        chain_eqe(cfg=configs, valid_ds=valid_ds, quantization_ds=quantization_ds, test_ds=test_ds)
        print('[INFO] : chain_eqe complete.')
    elif mode == 'chain_eqeb':
        chain_eqeb(cfg=configs, valid_ds=valid_ds, quantization_ds=quantization_ds, test_ds=test_ds)
        print('[INFO] : chain_eqeb complete.')
    elif mode == 'chain_qb':
        chain_qb(cfg=configs, quantization_ds=quantization_ds)
        print('[INFO] : chain_qb complete.')
    elif mode == 'chain_qd':
        # --- FIX: Add the check here too ---
        if is_text_gen_project:
            print("[WARNING] : Deployment part of chain_qd is skipped for text generation project.")
            quantize(cfg=configs, quantization_ds=quantization_ds)
            print('[INFO] : Quantization complete.')
        else:
            chain_qd(cfg=configs, quantization_ds=quantization_ds, hardware_type=configs.hardware_type)
        print('[INFO] : chain_qd complete.')
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # ... (rest of the file remains the same) ...
    mlflow.log_artifact(configs.output_dir)
    if mode in ['benchmarking', 'chain_qb', 'chain_eqeb', 'chain_tqeb']:
        mlflow.log_param("stm32ai_version", configs.tools.stm32ai.version)
        mlflow.log_param("target", configs.benchmarking.board)
    log_to_file(configs.output_dir, f'operation finished: {mode}')
    if get_active_config_file() is not None: 
        print(f"[INFO] : ClearML task connection")
        task = Task.current_task()
        task.connect(configs)


@hydra.main(version_base=None, config_path="", config_name="user_config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point of the script.

    Args:
        cfg: Configuration dictionary.

    Returns:
        None
    """
    OmegaConf.set_struct(cfg, False)

    # --- CALL THE NEW GPU SETUP FUNCTION AT THE VERY BEGINNING ---
    setup_gpus()

    # Configure the GPU (the 'general' section may be missing)
    if "general" in cfg and cfg.general:
        # Set upper limit on usable GPU memory
        if "gpu_memory_limit" in cfg.general and cfg.general.gpu_memory_limit:
            set_gpu_memory_limit(cfg.general.gpu_memory_limit)
            print(f"[INFO] : Setting upper limit of usable GPU memory to {int(cfg.general.gpu_memory_limit)}GBytes.")
        else:
            print("[WARNING] The usable GPU memory is unlimited.\n"
                  "Please consider setting the 'gpu_memory_limit' attribute "
                  "in the 'general' section of your configuration file.")

    # Parse the configuration file
    cfg = get_config(cfg)
    cfg.output_dir = HydraConfig.get().run.dir
    mlflow_ini(cfg)

    # Checks if there's a valid ClearML configuration file
    print(f"[INFO] : ClearML config check")
    if get_active_config_file() is not None:
        print(f"[INFO] : ClearML initialization and configuration")
        # ClearML - Initializing ClearML's Task object.
        task = Task.init(project_name=cfg.general.project_name,
                         task_name='ic_modelzoo_task')
        # ClearML - Optional yaml logging 
        task.connect_configuration(name=cfg.operation_mode, 
                                   configuration=cfg)
          
    # Seed global seed for random generators
    seed = get_random_seed(cfg)
    print(f'[INFO] : The random seed for this simulation is {seed}')
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)

    # Extract the mode from the command-line arguments
    mode = cfg.operation_mode
    preprocess_output = preprocess(cfg=cfg)
    train_ds, valid_ds, quantization_ds, test_ds = preprocess_output
    # Process the selected mode
    process_mode(mode=mode, configs=cfg, train_ds=train_ds, valid_ds=valid_ds, quantization_ds=quantization_ds,
                 test_ds=test_ds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='', help='Path to folder containing configuration file')
    parser.add_argument('--config-name', type=str, default='user_config', help='name of the configuration file')
    # add arguments to the parser
    parser.add_argument('params', nargs='*',
                        help='List of parameters to over-ride in config.yaml')
    args = parser.parse_args()

    # Call the main function
    main()

    # log the config_path and config_name parameters
    mlflow.log_param('config_path', args.config_path)
    mlflow.log_param('config_name', args.config_name)
    mlflow.end_run()