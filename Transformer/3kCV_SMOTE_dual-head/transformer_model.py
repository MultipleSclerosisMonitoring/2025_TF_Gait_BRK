"""
Transformer Encoder–Decoder Model

Single concatenated output: [class-probabilities | reconstruction].

Loss = homoscedastic-weighted categorical crossentropy + Huber.
"""

from __future__ import annotations
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    Flatten, MultiHeadAttention, Concatenate
)
from tensorflow.keras.models import Model
import config


class PositionalEncoding(tf.keras.layers.Layer):
    """
    Implements fixed sinusoidal positional encodings.

    Adds a position-dependent bias to the input embeddings so the model
    can reason about token order.

    Args:
        length (int): Maximum sequence length.
        d_model (int): Dimension of the embedding space.
    """

    def __init__(self, length: int, d_model: int):
        super().__init__()
        pos = tf.cast(tf.range(length)[:, None], tf.float32)
        i = tf.cast(tf.range(d_model)[None, :], tf.float32)
        angle = pos / tf.pow(10000.0, 2 * (i // 2) / d_model)
        # alternate sin and cos across embedding dims
        angle = tf.where(tf.math.floormod(i, 2) == 0,
                         tf.sin(angle), tf.cos(angle))
        # shape = (1, length, d_model)
        self.pe = angle[None]

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Add positional encoding to the input tensor.

        Args:
            x (tf.Tensor): Input embeddings of shape (batch, seq_len, d_model).

        Returns:
            tf.Tensor: Positionally encoded tensor, same shape as input.
        """
        return x + self.pe[:, : tf.shape(x)[1], :]


class EncoderBlock(tf.keras.layers.Layer):
    """
    Single Transformer encoder block.

    Combines multi-head self-attention, residual connections,
    layer normalization, and a feed-forward network.

    Args:
        d_model (int): Dimension of the model.
        num_heads (int): Number of attention heads.
        dff (int): Dimensionality of the inner feed-forward layer.
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
        self.n1 = LayerNormalization(epsilon=1e-6)
        self.n2 = LayerNormalization(epsilon=1e-6)
        self.d1 = Dropout(drop)
        self.d2 = Dropout(drop)

    def call(self, x: tf.Tensor, training: bool = None) -> tf.Tensor:
        """
        Forward pass for the encoder block.

        Args:
            x (tf.Tensor): Input tensor (batch, seq_len, d_model).
            training (bool, optional): Whether in training mode.

        Returns:
            tf.Tensor: Output tensor of same shape as input.
        """
        attn_out = self.mha(x, x, x, training=training)
        x = self.n1(x + self.d1(attn_out, training=training))
        ffn_out = self.ffn(x)
        x = self.n2(x + self.d2(ffn_out, training=training))
        return x


class DecoderBlock(tf.keras.layers.Layer):
    """
    Single Transformer decoder block.

    Performs masked self-attention, encoder–decoder attention,
    residual connections, layer normalization, and feed-forward network.

    Args:
        d_model (int): Dimension of the model.
        num_heads (int): Number of attention heads.
        dff (int): Dimensionality of the inner feed-forward layer.
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
        self.n1 = LayerNormalization(epsilon=1e-6)
        self.n2 = LayerNormalization(epsilon=1e-6)
        self.n3 = LayerNormalization(epsilon=1e-6)
        self.d1 = Dropout(drop)
        self.d2 = Dropout(drop)
        self.d3 = Dropout(drop)

    def call(
        self,
        x: tf.Tensor,
        enc_out: tf.Tensor,
        training: bool = None
    ) -> tf.Tensor:
        """
        Forward pass for the decoder block.

        Args:
            x (tf.Tensor): Decoder input tensor (batch, tgt_len, d_model).
            enc_out (tf.Tensor): Encoder output (batch, src_len, d_model).
            training (bool, optional): Whether in training mode.

        Returns:
            tf.Tensor: Decoder output of shape (batch, tgt_len, d_model).
        """
        attn1 = self.mha1(x, x, x, training=training)
        x = self.n1(x + self.d1(attn1, training=training))

        attn2 = self.mha2(x, enc_out, enc_out, training=training)
        x = self.n2(x + self.d2(attn2, training=training))

        ffn_out = self.ffn(x)
        x = self.n3(x + self.d3(ffn_out, training=training))
        return x


class StartToken(tf.keras.layers.Layer):
    """
    Learns a trainable start-of-sequence token.

    Prepends a fixed learned vector to each input batch.

    Args:
        d_model (int): Dimension of the model embeddings.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def build(self, input_shape) -> None:
        """
        Create the trainable start token.

        Args:
            input_shape: Shape of the encoder input (unused).
        """
        self.token = self.add_weight(
            name="start_tok",
            shape=(1, 1, self.d_model),
            initializer="random_normal",
            trainable=True
        )

    def call(self, batch: tf.Tensor) -> tf.Tensor:
        """
        Replicate start token for each batch instance.

        Args:
            batch (tf.Tensor): Input tensor (batch, seq_len, d_model).

        Returns:
            tf.Tensor: Tensor of shape (batch, 1, d_model) prepended as token.
        """
        batch_size = tf.shape(batch)[0]
        return tf.tile(self.token, [batch_size, 1, 1])


class DualLossConcat(tf.keras.losses.Loss):
    """
    Combines classification and reconstruction losses with
    learnable homoscedastic uncertainty weights.

    Loss = exp(-s1)*CE + exp(-s2)*Huber + (s1 + s2).

    Args:
        num_classes (int): Number of target classes.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.k = num_classes
        # log-variance terms (trainable)
        self.log_s_cls = tf.Variable(0.0, trainable=True)
        self.log_s_rec = tf.Variable(0.0, trainable=True)
        self.ce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        self.huber = tf.keras.losses.Huber(delta=1.0)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute the weighted dual loss.

        Args:
            y_true (tf.Tensor): Ground truth concatenated tensor of shape (batch, k + num_feats).
            y_pred (tf.Tensor): Predicted concatenated tensor of same shape.

        Returns:
            tf.Tensor: Scalar loss value.
        """
        # split classification vs. reconstruction parts
        y_c_true, y_r_true = y_true[:, : self.k], y_true[:, self.k :]
        y_c_pred, y_r_pred = y_pred[:, : self.k], y_pred[:, self.k :]

        ce_loss = self.ce(y_c_true, y_c_pred)
        rec_loss = self.huber(y_r_true, y_r_pred)

        inv_s_cls = tf.exp(-self.log_s_cls)
        inv_s_rec = tf.exp(-self.log_s_rec)
        return (inv_s_cls * ce_loss +
                inv_s_rec * rec_loss +
                (self.log_s_cls + self.log_s_rec))


def build_transformer(seq_len: int, num_feats: int) -> Model:
    """
    Factory function to build the Transformer model.

    The model consists of:
      1. A Dense projection to d_model.
      2. Positional encoding + dropout.
      3. Stacked encoder blocks.
      4. Learned start token + stacked decoder blocks.
      5. Two heads: classification (softmax) and reconstruction (linear).
      6. Final concatenation of both heads.

    Args:
        seq_len (int): Length of the input sequence.
        num_feats (int): Number of features per timestep.

    Returns:
        Model: A compiled Keras Model ready for training.
    """
    d_model = config.D_MODEL

    # Encoder input
    x_in = Input((seq_len, num_feats), name="enc_in")
    x = Dense(d_model)(x_in)
    x = PositionalEncoding(seq_len, d_model)(x)
    x = Dropout(config.DROPOUT)(x)
    for _ in range(config.NUM_ENC_LAYERS):
        x = EncoderBlock(d_model, config.NUM_HEADS, config.DFF, config.DROPOUT)(x)

    # Decoder input (start token + original input length)
    y = StartToken(d_model)(x_in)
    for _ in range(config.NUM_DEC_LAYERS):
        y = DecoderBlock(d_model, config.NUM_HEADS, config.DFF, config.DROPOUT)(y, x)

    # Classification and reconstruction heads
    cls_vec = Flatten()(y)
    cls_out = Dense(config.NUM_CLASSES, activation="softmax")(cls_vec)
    recon_out = Dense(num_feats, activation="linear")(cls_vec)

    # Final concatenated output
    out = Concatenate(name="out")([cls_out, recon_out])

    return Model(inputs=x_in, outputs=out)
