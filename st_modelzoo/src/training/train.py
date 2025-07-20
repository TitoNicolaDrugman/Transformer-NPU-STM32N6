# src/training/train.py

# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  * MODIFIED FOR TEXT GENERATION
#  *--------------------------------------------------------------------------------------------*/

import os
from timeit import default_timer as timer
from datetime import timedelta
from typing import List, Optional

import mlflow
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import tensorflow as tf

from common.utils import log_to_file, log_last_epoch_history, LRTensorBoard, \
                         model_summary, collect_callback_args, vis_training_curves
from common.training import get_optimizer, lr_schedulers
from src.models.tiny_bert_generator import get_tiny_bert_generator
from src.evaluation import evaluate_h5_model

def _get_callbacks(callbacks_dict: DictConfig, output_dir: str = None, logs_dir: str = None,
                  saved_models_dir: str = None) -> List[tf.keras.callbacks.Callback]:
    """
    This function creates the list of Keras callbacks, monitoring 'val_loss'.
    """
    # ... (This function is already correct and does not need changes)
    message = "\nPlease check the 'training.callbacks' section of your configuration file."
    lr_scheduler_names = lr_schedulers.get_scheduler_names()
    num_lr_schedulers = 0
    callback_list = []
    if callbacks_dict is not None:
        for name, args in callbacks_dict.items():
            if name in ("ModelCheckpoint", "TensorBoard", "CSVLogger"):
                raise ValueError(f"The `{name}` callback is built-in and can't be redefined.{message}")
            text = f"lr_schedulers.{name}" if name in lr_scheduler_names else f"tf.keras.callbacks.{name}"
            text += collect_callback_args(name, args=args, message=message)
            try:
                callback = eval(text)
            except Exception as error:
                raise ValueError(f"Callback `{name}` is unknown or has invalid args.{message}") from error
            callback_list.append(callback)
            if name in lr_scheduler_names + ["ReduceLROnPlateau", "LearningRateScheduler"]:
                num_lr_schedulers += 1
    if num_lr_schedulers > 1:
        raise ValueError(f"\nFound more than one learning rate scheduler{message}")
    callback = tf.keras.callbacks.ModelCheckpoint(
                        filepath=os.path.join(output_dir, saved_models_dir, "best_model.h5"),
                        save_best_only=True, save_weights_only=False,
                        monitor="val_loss", mode="min")
    callback_list.append(callback)
    callback = tf.keras.callbacks.ModelCheckpoint(
                        filepath=os.path.join(output_dir, saved_models_dir, "last_model.h5"),
                        save_best_only=False)
    callback_list.append(callback)
    callback = LRTensorBoard(log_dir=os.path.join(output_dir, logs_dir))
    callback_list.append(callback)
    callback = tf.keras.callbacks.CSVLogger(os.path.join(output_dir, logs_dir, "metrics", "train_metrics.csv"))
    callback_list.append(callback)
    return callback_list

# In src/training/train.py

def train(cfg: DictConfig = None, train_ds: tf.data.Dataset = None,
          valid_ds: tf.data.Dataset = None, test_ds: Optional[tf.data.Dataset] = None) -> str:
    """
    Trains the model using the provided configuration and datasets.
    """
    output_dir = HydraConfig.get().runtime.output_dir
    saved_models_dir = cfg.general.saved_models_dir

    # ==============================================================================
    # --- FINAL, EXPLICIT MULTI-GPU STRATEGY ---
    # ==============================================================================
    # We will manually specify the devices for the strategy to use.
    # This avoids any auto-detection issues with mixed CPU/GPU environments.
    # It will use the two GPUs that were made visible by the os.environ call in stm32ai_main.py
    #strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:2"])
    #strategy = tf.distribute.MirroredStrategy()
    print("\n[DEBUG] : Initializing tf.distribute.MirroredStrategy in 'src/training/train.py'...\n")
    #strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
    print("\n[DEBUG] : Initializing tf.distribute.MirroredStrategy (Windows, no NCCL)…\n")
    strategy = tf.distribute.MirroredStrategy(
        devices=["/gpu:0", "/gpu:1"],
        cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

    # ==============================================================================
    
    print(f'[INFO] : Number of devices in strategy: {strategy.num_replicas_in_sync}')
    
    global_batch_size = cfg.training.batch_size * strategy.num_replicas_in_sync
    print(f'[INFO] : Per-replica batch size: {cfg.training.batch_size}')
    print(f'[INFO] : Global batch size: {global_batch_size}')

    # Open a strategy scope.
    with strategy.scope():
        # All model creation and compilation MUST be inside the scope
        if cfg.training.model.name == 'tiny_bert_generator':
            print(f"[INFO] : Creating new `{cfg.training.model.name}` model")
            model = get_tiny_bert_generator(**cfg.training.model)
        elif cfg.training.resume_training_from:
            print(f"[INFO] : Resuming training from model file {cfg.training.resume_training_from}")
            # When loading a model, it must be loaded inside the strategy scope
            model = tf.keras.models.load_model(cfg.training.resume_training_from)
        else:
            raise ValueError("Must specify a model to train or resume from a checkpoint.")
        
        # Compile the model
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(loss=loss_fn,
                      optimizer=get_optimizer(cfg=cfg.training.optimizer),
                      metrics=['accuracy'])
    # --- END OF STRATEGY SCOPE ---

    model_summary(model)

    callbacks = _get_callbacks(callbacks_dict=cfg.training.callbacks,
                               output_dir=output_dir,
                               saved_models_dir=saved_models_dir,
                               logs_dir=cfg.general.logs_dir
                               )

    print("[INFO] : Starting training...")
    start_time = timer()
    history = None
    try:
        history = model.fit(train_ds,
                            validation_data=valid_ds,
                            epochs=cfg.training.epochs,
                            callbacks=callbacks,
                            verbose=1)
    except Exception as e: 
        print(f'\n[ERROR] : Training interrupted: {e}')
        
    # ... (The rest of the file is correct) ...
    last_epoch = 0
    if history:
        try:
            last_epoch = log_last_epoch_history(cfg, history, output_dir)
            vis_training_curves(history=history, output_dir=output_dir)
        except Exception:
             print("[WARNING] Could not log or visualize training history.")
    end_time = timer()
    fit_run_time = int(end_time - start_time)
    if last_epoch > 0:
        average_time_per_epoch = round(fit_run_time / (int(last_epoch) + 1), 2)
        print(f"Training runtime: {timedelta(seconds=fit_run_time)}")
        log_to_file(cfg.output_dir, (f"Training runtime : {fit_run_time} s\n" +
                                     f"Average time per epoch : {average_time_per_epoch} s"))
    best_model_path = os.path.join(output_dir, saved_models_dir, "best_model.h5")
    if os.path.exists(best_model_path):
        print(f"[INFO] : Best model saved at: {best_model_path}")
        if cfg.training.trained_model_path:
            # When loading a saved model, it's best to do it inside the strategy scope
            with strategy.scope():
                loaded_model = tf.keras.models.load_model(best_model_path)
            loaded_model.save(cfg.training.trained_model_path)
            print(f"[INFO] : Saved trained model in file {cfg.training.trained_model_path}")
    else:
        print("[WARNING] : Best model was not saved. Returning path to last model.")
        best_model_path = os.path.join(output_dir, saved_models_dir, "last_model.h5")
    print("[INFO] : Evaluation of the float model will be done in the next step of the chain.")
    return best_model_path