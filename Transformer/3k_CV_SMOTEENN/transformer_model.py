"""
Full encoder–decoder Transformer classifier with 3 decoder blocks.

No reconstruction head—decoder output goes directly into softmax.
"""

from __future__ import annotations
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    Flatten
)
from tensorflow.keras.models import Model
import config


class PositionalEncoding(tf.keras.layers.Layer):
    """
    Add fixed sinusoidal positional encodings to embeddings.

    Args:
        length (int): Maximum sequence length (timesteps).
        d_model (int): Embedding dimension.
    """

    def __init__(self, length: int, d_model: int):
        super().__init__()
        pos = tf.cast(tf.range(length)[:, None], tf.float32)
        i = tf.cast(tf.range(d_model)[None, :], tf.float32)
        angle = pos / tf.pow(10000.0, 2 * (i // 2) / d_model)
        angle = tf.where(tf.math.floormod(i, 2) == 0,
                         tf.sin(angle), tf.cos(angle))
        self.pe = angle[None]

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Add positional encoding.

        Args:
            x (tf.Tensor): Shape (batch, seq_len, d_model).

        Returns:
            tf.Tensor: x + positional encodings.
        """
        return x + self.pe[:, : tf.shape(x)[1], :]


class EncoderBlock(tf.keras.layers.Layer):
    """
    Single Transformer encoder block.

    Args:
        d_model (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        dff (int): Inner dimension of feed-forward network.
        drop (float): Dropout rate.
    """

    def __init__(self, d_model: int, num_heads: int, dff: int, drop: float):
        super().__init__()
        depth = d_model // num_heads
        self.mha = MultiHeadAttention(num_heads, depth, dropout=drop)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation="relu"),
            Dense(d_model)
        ])
        self.ln1 = LayerNormalization(epsilon=1e-6)
        self.ln2 = LayerNormalization(epsilon=1e-6)
        self.do1 = Dropout(drop)
        self.do2 = Dropout(drop)

    def call(self, x: tf.Tensor, training: bool = None) -> tf.Tensor:
        """
        Forward pass.

        Args:
            x (tf.Tensor): (batch, seq_len, d_model).
            training (bool): Training flag.

        Returns:
            tf.Tensor: Same shape as input.
        """
        att = self.mha(x, x, x, training=training)
        x = self.ln1(x + self.do1(att, training=training))
        ffn_out = self.ffn(x)
        return self.ln2(x + self.do2(ffn_out, training=training))


class DecoderBlock(tf.keras.layers.Layer):
    """
    Single Transformer decoder block with cross-attention.

    Args:
        d_model (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        dff (int): Inner dimension of feed-forward network.
        drop (float): Dropout rate.
    """

    def __init__(self, d_model: int, num_heads: int, dff: int, drop: float):
        super().__init__()
        depth = d_model // num_heads
        self.mha1 = MultiHeadAttention(num_heads, depth, dropout=drop)
        self.mha2 = MultiHeadAttention(num_heads, depth, dropout=drop)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation="relu"),
            Dense(d_model)
        ])
        self.ln1 = LayerNormalization(epsilon=1e-6)
        self.ln2 = LayerNormalization(epsilon=1e-6)
        self.ln3 = LayerNormalization(epsilon=1e-6)
        self.do1 = Dropout(drop)
        self.do2 = Dropout(drop)
        self.do3 = Dropout(drop)

    def call(
        self,
        x: tf.Tensor,
        enc_out: tf.Tensor,
        training: bool = None
    ) -> tf.Tensor:
        """
        Forward pass.

        Args:
            x (tf.Tensor): Decoder input (batch, tgt_len, d_model).
            enc_out (tf.Tensor): Encoder output (batch, src_len, d_model).
            training (bool): Training flag.

        Returns:
            tf.Tensor: (batch, tgt_len, d_model).
        """
        a1 = self.mha1(x, x, x, training=training)
        x = self.ln1(x + self.do1(a1, training=training))
        a2 = self.mha2(x, enc_out, enc_out, training=training)
        x = self.ln2(x + self.do2(a2, training=training))
        ffn_out = self.ffn(x)
        return self.ln3(x + self.do3(ffn_out, training=training))


class StartToken(tf.keras.layers.Layer):
    """
    Learnable start-of-sequence token for the decoder.

    Args:
        d_model (int): Embedding dimension.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def build(self, input_shape) -> None:
        """
        Create the token weight.

        Args:
            input_shape: Unused.
        """
        self.token = self.add_weight(
            name="start_token",
            shape=(1, 1, self.d_model),
            initializer="random_normal",
            trainable=True
        )

    def call(self, batch: tf.Tensor) -> tf.Tensor:
        """
        Tile the start token for each batch.

        Args:
            batch (tf.Tensor): Encoder input (batch, seq_len, d_model).

        Returns:
            tf.Tensor: (batch, 1, d_model).
        """
        bs = tf.shape(batch)[0]
        return tf.tile(self.token, [bs, 1, 1])


class ClassAccuracy(tf.keras.metrics.Metric):
    """
    Custom metric: categorical accuracy on the classifier head.

    Args:
        num_classes (int): Number of target classes.
        name (str): Metric name.
        **kwargs: Additional args for base Metric.
    """

    def __init__(self, num_classes: int, name: str = "class_acc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.acc = tf.keras.metrics.CategoricalAccuracy()

    def update_state(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None
    ) -> None:
        """
        Update metric state with one batch.

        Args:
            y_true (tf.Tensor): One-hot true labels.
            y_pred (tf.Tensor): Predicted probabilities.
            sample_weight: Optional sample weights.
        """
        self.acc.update_state(y_true, y_pred, sample_weight)

    def result(self) -> tf.Tensor:
        """Return the current accuracy."""
        return self.acc.result()

    def reset_state(self) -> None:
        """Reset internal statistics."""
        self.acc.reset_state()


def build_encoder_decoder_classifier(
    seq_len: int,
    num_feats: int
) -> Model:
    """
    Build a full encoder–decoder Transformer for classification.

    Args:
        seq_len (int): Number of timesteps per input.
        num_feats (int): Number of features per timestep.

    Returns:
        Model: Keras Model with a softmax head of size NUM_CLASSES.
    """
    d_model = config.D_MODEL

    enc_in = Input((seq_len, num_feats), name="encoder_input")
    # Encoder stack
    x = Dense(d_model)(enc_in)
    x = PositionalEncoding(seq_len, d_model)(x)
    x = Dropout(config.DROPOUT)(x)
    for _ in range(config.NUM_ENC_LAYERS):
        x = EncoderBlock(d_model, config.NUM_HEADS, config.DFF, config.DROPOUT)(x)

    # Decoder stack (no recon head)
    y = StartToken(d_model)(enc_in)
    for _ in range(config.NUM_DEC_LAYERS):
        y = DecoderBlock(d_model, config.NUM_HEADS, config.DFF, config.DROPOUT)(y, x)

    # Classification head
    cls = Flatten()(y)
    out = Dense(config.NUM_CLASSES, activation="softmax", name="classifier")(cls)
    return Model(inputs=enc_in, outputs=out)
