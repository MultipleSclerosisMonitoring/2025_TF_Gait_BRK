"""
Configuration file defining paths, windowing parameters, model hyperparameters,
training settings, and random seed for the data preprocessing and modeling pipeline.
"""
from pathlib import Path
import tensorflow as tf

# ── paths ─────────────────────────────────────────────────────
RAW_DATA_DIR = Path(r"chunks_10s") # Folder of .xlsx chunks
ARTIFACT_ROOT  = Path("./artifacts")      # Root for outputs
ARTIFACT_ROOT.mkdir(exist_ok=True, parents=True)
ARTIFACT_DIR   = ARTIFACT_ROOT

# ── windowing ────────────────────────────────────────────────
TIMESTEPS = 256                           # Number of timesteps per window
STRIDE = 128                              # Sliding window shift
FEAT_COLS = ["A", "G", "M", "S0", "S1", "S2"]
FINAL_COLUMNS = FEAT_COLS                 # Alias for preprocessing pipeline

# ── model hyper-parameters ───────────────────────────────────
D_MODEL        = 128                       # Embedding  model dimension
NUM_HEADS      = 4                         # Attention heads
DFF            = 4 * D_MODEL               # FFN dim
NUM_ENC_LAYERS = 3                         # Encoder blocks
NUM_DEC_LAYERS = 3                         # Decoder blocks
DROPOUT        = 0.1                       # Dropout rate

# ── training ─────────────────────────────────────────────────
EPOCHS         = 120                       # Total epochs
BATCH_SIZE     = 64                        # Batch size
LR             = 3e-4                      # Initial LR
LR_SCHED       = tf.keras.optimizers.schedules.CosineDecayRestarts(
                    initial_learning_rate=LR,
                    first_decay_steps=4000,
                    t_mul=2.0,
                    m_mul=0.5
                 )
SEED           = 42                        # Random seed
NUM_CLASSES    = 2                         # Output classes
