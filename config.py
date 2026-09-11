#Centralized configuration of hyperparameters

# --- problem setup ---
DIGIT = 0 # digit class treated as "good"
ALL_DIGITS = tuple(range(10))
DIGITS_ANOMALY = tuple(d for d in ALL_DIGITS if d != DIGIT) # digits treated as anomaly
DIM = 784 # 28*28
SIGMA = 0.1 # regularization parameter

# --- model ---
BASE_CHANNELS = 32 # U-Net width; channels are 32 (BASE_CHANNELS) -> 64 (2*BASE_CHANNELS) -> 128 (4*BASE_CHANNELS)

# --- training & optimization ---
BATCH_SIZE = 64
NUM_EPOCHS = 60
LEARNING_RATE = 3e-4

# --- CFM score ---
NUM_SAMPLES = 1000 # Monte Carlo draws of (T, eps) per image to approx. CFM score

# --- likelihood score / ODE integration ---
NUM_STEPS = 100 # RK2 steps, i.e. step size h = 1/M_STEPS
NUM_PROBES = 5 # Monte Carlo draws for Hutchinson estimator of divergence, per integration step

# --- evaluation ---
N_PER_DIGIT = 100 # test images per digit in test_anomalies
N_NORMAL = 300 # normal-class images used for the ROC curve
N_ANOMALY_PER_DIGIT = 100   # anomaly images per digit used for the ROC curve

# --- checkpoints ---
CHECKPOINT_PATH = "checkpoints/unet_vector_field.pt" 

def checkpoint_for(sigma: float) -> str:
    """Checkpoint path for a given sigma, so sweeps don't overwrite each other."""
    return f"checkpoints/unet_sigma_{sigma}.pt"