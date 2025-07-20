# src/models/tiny_bert_generator.py
# ---------------------------------------------------------------------------------
#  Copyright (c) 2022-2025 STMicroelectronics.
#  All rights reserved.
#  MODIFIED 2025-06-24 – adds `include_embedding` to build a Gather-free model
# ---------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

# ------------------------------------------------------------------ utils
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model,
    )
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


# ------------------------------------------------------------------ core blocks
class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.wq = layers.Dense(d_model, name="wq")
        self.wk = layers.Dense(d_model, name="wk")
        self.wv = layers.Dense(d_model, name="wv")
        self.dense = layers.Dense(d_model, name="output_dense")

    # ----- Keras plumbing
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "num_heads": self.num_heads})
        return cfg

    # ----- helpers
    def _split_heads(self, x, batch):
        x = tf.reshape(x, (batch, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    # ----- forward
    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        q = self._split_heads(self.wq(q), batch_size)
        k = self._split_heads(self.wk(k), batch_size)
        v = self._split_heads(self.wv(v), batch_size)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(dk)
        if mask is not None:
            scores += (mask * -1e9)

        weights = tf.nn.softmax(scores, axis=-1)
        out = tf.matmul(weights, v)                         # (B, H, T, depth)
        out = tf.transpose(out, perm=[0, 2, 1, 3])          # (B, T, H, depth)
        out = tf.reshape(out, (batch_size, -1, self.d_model))
        return self.dense(out)


def point_wise_ffn(d_model, dff):
    return tf.keras.Sequential(
        [layers.Dense(dff, activation="relu"), layers.Dense(d_model)], name="ffn"
    )


class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, **kwargs):
        super().__init__(**kwargs)
        self.mha = MultiHeadAttention(d_model, num_heads, name="mha")
        self.ffn = point_wise_ffn(d_model, dff)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="ln1")
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="ln2")

    def call(self, x, mask=None):
        attn_out = self.mha(x, x, x, mask)
        out1 = self.norm1(x + attn_out)
        ffn_out = self.ffn(out1)
        return self.norm2(out1 + ffn_out)

    # for save / reload
    def get_config(self):
        base = super().get_config()
        base.update(
            {
                "d_model": self.mha.d_model,
                "num_heads": self.mha.num_heads,
                "dff": self.ffn.layers[0].units,
            }
        )
        return base


# ------------------------------------------------------------------ factory
def get_tiny_bert_generator(
    *,
    input_shape,
    vocab_size,
    embedding_dim,
    num_layers,
    num_heads,
    dff,
    include_embedding: bool = True,
    name="tiny_bert_generator",
    **unused,
):
    """
    Build a *tiny* Transformer encoder for character-level generation.

    Parameters
    ----------
    include_embedding : bool, default **True**
        • True  – model input is a tensor of **token IDs** (int64)  
        • False – model input is **already embedded** (float32,
          shape = [seq_len, embedding_dim]).  
          → Use this for inference/quantization so the graph has **no
          `Embedding` / `Gather` op**.
    """
    seq_len = input_shape[0]

    if include_embedding:
        inputs = layers.Input(shape=(seq_len,), dtype="int64", name="token_ids")
        x = layers.Embedding(
            vocab_size, embedding_dim, name="token_embedding"
        )(inputs)
    else:
        inputs = layers.Input(
            shape=(seq_len, embedding_dim),
            dtype="float32",
            name="embedded_tokens",
        )
        x = inputs

    # scale + add positional encoding
    x *= tf.math.sqrt(tf.cast(embedding_dim, tf.float32))
    x = x + positional_encoding(seq_len, embedding_dim)

    # N encoder layers -------------------------------------------------
    for i in range(num_layers):
        x = EncoderLayer(embedding_dim, num_heads, dff, name=f"enc_{i}")(x)

    outputs = layers.Dense(vocab_size, name="output_dense")(x)
    return tf.keras.Model(inputs, outputs, name=name)
