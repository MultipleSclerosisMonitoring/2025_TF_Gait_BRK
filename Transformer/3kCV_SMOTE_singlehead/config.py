"""
Configuration file defining paths, windowing parameters, model hyperparameters,
training settings, and random seed for the data preprocessing and modeling pipeline.
"""

from pathlib import Path
import tensorflow as tf

# ── paths ─────────────────────────────────────────────────────
RAW_DATA_DIR = Path(r"\chunks_5s")        # Directory where raw .xlsx files live
ARTIFACT_ROOT = Path("./artifacts")       # Base directory for outputs (auto-created)
ARTIFACT_ROOT.mkdir(exist_ok=True, parents=True)
ARTIFACT_DIR = ARTIFACT_ROOT

# ── windowing ────────────────────────────────────────────────
TIMESTEPS = 220                           # Number of timesteps per window
STRIDE = 110                              # Sliding window shift
FEAT_COLS = ["A", "G", "M", "S0", "S1", "S2"]
FINAL_COLUMNS = FEAT_COLS                 # Alias for preprocessing pipeline

# ── model hyperparameters & training ────────────────────────
D_MODEL = 128                             # Transformer model dimension
NUM_HEADS = 4                             # Attention heads
DFF = 4 * D_MODEL                         # Feed-forward network size
NUM_ENC_LAYERS = 3                        # Encoder layers
NUM_DEC_LAYERS = 3                        # Decoder layers
DROPOUT = 0.1                             # Dropout rate


EPOCHS = 120                              # Training epochs
BATCH_SIZE = 64                           # Batch size
LR = 3e-4                                 # Initial learning rate
LR_SCHED = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=LR,
    first_decay_steps=4000,
    t_mul=2.0,
    m_mul=0.5
)
SEED = 42                                 # Random seed
NUM_CLASSES = 2                           # Number of target classes

NUM_FEATURES = len(FINAL_COLUMNS)

