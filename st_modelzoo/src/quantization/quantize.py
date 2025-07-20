# src/quantization/quantize.py
# ---------------------------------------------------------------------------------
#  MODIFIED 2025-06-24 – builds a second model *without* the Embedding so the
#  exported TFLite contains zero `Gather` ops.
# ---------------------------------------------------------------------------------
import os, numpy as np, tensorflow as tf
from omegaconf import DictConfig
from tensorflow.keras import backend as K

from src.models.tiny_bert_generator import (
    get_tiny_bert_generator,
    EncoderLayer,
    MultiHeadAttention,
)

# ---------------- utils ----------------------------------------------------------
def strip_dropout(model: tf.keras.Model) -> tf.keras.Model:
    """Return a copy with every Dropout replaced by a no-op Activation."""
    K.set_learning_phase(0)

    def _convert(layer):
        return (
            tf.keras.layers.Activation("linear", name=f"{layer.name}_noop")
            if isinstance(layer, tf.keras.layers.Dropout)
            else layer
        )

    clone = tf.keras.models.clone_model(model, clone_function=_convert)
    clone.set_weights(model.get_weights())
    return clone


def build_no_embedding_model(trained: tf.keras.Model, cfg: DictConfig):
    """Create a copy of the network that *skips* the token-Embedding layer."""
    m_cfg = dict(cfg.training.model)          # copy to avoid side-effects
    m_cfg["include_embedding"] = False        # <-- key trick
    inf_model = get_tiny_bert_generator(**m_cfg)

    # --- copy weights layer-by-layer (names match except the Embedding) ----------
    for layer in inf_model.layers:
        try:
            twin = trained.get_layer(layer.name)
            if twin.weights:
                layer.set_weights(twin.get_weights())
        except ValueError:
            # layer not present in trained model (e.g., new Input)
            continue
    return inf_model


# ---------------- main -----------------------------------------------------------
def quantize(
    cfg: DictConfig,
    quantization_ds: tf.data.Dataset,
    float_model_path: str = None,
) -> str:
    """
    Quantize *and remove the Embedding* so the generated TFLite has no Gather op.
    """
    out_dir = cfg.output_dir
    float_model_path = float_model_path or os.path.join(
        out_dir, cfg.general.saved_models_dir, "best_model.h5"
    )
    if not os.path.exists(float_model_path):
        float_model_path = float_model_path.replace("best_model", "last_model")
    if not os.path.exists(float_model_path):
        raise FileNotFoundError(float_model_path)

    # ---------------- 1. load the training-time model (with Embedding) ----------
    custom_objs = {"EncoderLayer": EncoderLayer, "MultiHeadAttention": MultiHeadAttention}
    trained_model = tf.keras.models.load_model(float_model_path, custom_objects=custom_objs)
    print("[INFO] loaded trained model")

    # ---------------- 2. build the inference model sans Embedding --------------
    inf_model = build_no_embedding_model(trained_model, cfg)
    print("[INFO] built inference model without Embedding (Gather-free)")

    # save a temp copy so TF-Lite converter has a clean graph
    tmp_path = os.path.join(out_dir, "tmp_emb-free_model.h5")
    inf_model.save(tmp_path)
    inf_model = tf.keras.models.load_model(tmp_path, custom_objects=custom_objs)

    # keep the embedding matrix – you’ll need it on the MCU side
    embedding_matrix = trained_model.get_layer("token_embedding").get_weights()[0]

    # ---------------- 3. representative dataset (already embedded) -------------
    def rep_data():
        for token_batch, _ in quantization_ds.take(200):
            # (B, T) int64  →  (B, T, E) float32
            emb = tf.nn.embedding_lookup(embedding_matrix, token_batch)
            yield [emb]

    # ---------------- 4. TF-Lite conversion ------------------------------------
    converter = tf.lite.TFLiteConverter.from_keras_model(inf_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    print("[INFO] TFLite conversion done – Gather op is GONE 🎉")

    export_dir = os.path.join(out_dir, cfg.quantization.export_dir)
    os.makedirs(export_dir, exist_ok=True)
    tfl_path = os.path.join(export_dir, "quantized_model.tflite")
    with open(tfl_path, "wb") as f:
        f.write(tflite_model)

    # optional: save the embedding matrix so you can embed tokens on the CPU/NPU
    np.save(os.path.join(export_dir, "embedding_matrix.npy"), embedding_matrix)

    # clean up
    os.remove(tmp_path)
    return tfl_path
