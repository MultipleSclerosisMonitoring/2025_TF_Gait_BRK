# transformer_model.py

from __future__ import annotations
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout,
    LayerNormalization, MultiHeadAttention,
    Flatten
)
from tensorflow.keras.models import Model
import config

# ── Positional Encoding ─────────────────────────────────────
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, length: int, d_model: int):
        super().__init__()
        pos = tf.cast(tf.range(length)[:, None], tf.float32)
        i   = tf.cast(tf.range(d_model)[None, :], tf.float32)
        angle = pos / tf.pow(10000.0, 2 * (i // 2) / d_model)
        angle = tf.where(tf.math.floormod(i, 2) == 0,
                         tf.sin(angle), tf.cos(angle))
        self.pe = angle[None]  # shape=(1, length, d_model)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return x + self.pe[:, : tf.shape(x)[1], :]


# ── Encoder Block ───────────────────────────────────────────
class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int, dff: int, drop: float):
        super().__init__()
        depth = d_model // num_heads
        self.mha = MultiHeadAttention(num_heads, depth, dropout=drop)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation="relu"),
            Dense(d_model)
        ])
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.drop1 = Dropout(drop)
        self.drop2 = Dropout(drop)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        attn = self.mha(x, x, x, training=training)
        x = self.norm1(x + self.drop1(attn, training=training))
        f = self.ffn(x)
        return self.norm2(x + self.drop2(f, training=training))


# ── Decoder Block ───────────────────────────────────────────
class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int, dff: int, drop: float):
        super().__init__()
        depth = d_model // num_heads
        self.mha1 = MultiHeadAttention(num_heads, depth, dropout=drop)  # self-attn
        self.mha2 = MultiHeadAttention(num_heads, depth, dropout=drop)  # enc-dec
        self.ffn  = tf.keras.Sequential([
            Dense(dff, activation="relu"),
            Dense(d_model)
        ])
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.norm3 = LayerNormalization(epsilon=1e-6)
        self.drop1 = Dropout(drop)
        self.drop2 = Dropout(drop)
        self.drop3 = Dropout(drop)

    def call(
        self,
        x: tf.Tensor,
        enc_out: tf.Tensor,
        training: bool = False
    ) -> tf.Tensor:
        a1 = self.mha1(x, x, x, training=training)
        x  = self.norm1(x + self.drop1(a1, training=training))

        a2 = self.mha2(x, enc_out, enc_out, training=training)
        x  = self.norm2(x + self.drop2(a2, training=training))

        f = self.ffn(x)
        return self.norm3(x + self.drop3(f, training=training))


# ── Trainable Start Token ───────────────────────────────────
class StartToken(tf.keras.layers.Layer):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def build(self, input_shape):
        # both name and shape are keyword-only now
        self.token = self.add_weight(
            name="start_token",
            shape=(1, 1, self.d_model),
            initializer="random_normal",
            trainable=True
        )
        super().build(input_shape)  # optional but recommended

    def call(self, batch: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(batch)[0]
        return tf.tile(self.token, [batch_size, 1, 1])



# ── Model Factory ───────────────────────────────────────────
def build_transformer_classifier(
    seq_len: int,
    num_feats: int
) -> Model:
    """
    Builds an encoder–decoder Transformer that only outputs class
    probabilities (softmax). Drop this into your run script and compile
    with CategoricalCrossentropy.

    Args:
      seq_len: number of timesteps in each window
      num_feats: number of features per timestep (e.g. 4 after aggregation)
    """
    d_model = config.D_MODEL

    # 1) Encoder input + embedding + positional encoding
    enc_in = Input((seq_len, num_feats), name="encoder_input")
    x      = Dense(d_model)(enc_in)
    x      = PositionalEncoding(seq_len, d_model)(x)
    x      = Dropout(config.DROPOUT)(x)

    # 2) Stacked encoder blocks
    for _ in range(config.NUM_ENC_LAYERS):
        x = EncoderBlock(d_model, config.NUM_HEADS, config.DFF, config.DROPOUT)(x)

    # 3) Decoder: single start token queries the encoder
    y = StartToken(d_model)(enc_in)
    y = Dropout(config.DROPOUT)(y)
    for _ in range(config.NUM_DEC_LAYERS):
        y = DecoderBlock(d_model, config.NUM_HEADS, config.DFF, config.DROPOUT)(y, x)

    # 4) Classification head only
    flat = Flatten()(y)  # shape=(batch, d_model)
    cls  = Dense(config.NUM_CLASSES, activation="softmax", name="classification")(flat)

    return Model(inputs=enc_in, outputs=cls)
