import os

# We Create 4 Folder To Store Data, Plots, Models and Logs 
DATA_DIR = r"B:\satellite\data"
PLOTS_DIR = r"B:\satellite\plots"
MODELS_DIR = r"B:\satellite\models"
LOGS_DIR = r"B:\satellite\logs"

for folder in [DATA_DIR, PLOTS_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(folder, exist_ok=True)

# SATELLITE SETTINGS
N_SATELLITES = 70
MAX_DEBRIS = 500
SAFE_DISTANCE_KM = 15.0
DANGER_ZONE_KM = 50.0   
EARTH_RADIUS_KM = 6371.0

# SIMULATION SETTINGS
TIMESTEP_SECONDS = 10
EPISODE_LENGTH = 540

# TRAINING SETTINGS
LEARNING_RATE = 3e-4
GAMMA = 0.99
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
TOTAL_TIMESTEPS = 50000

# COLLISION SAFETY SETTINGS 
COST_LIMIT = 0.01
LAMBDA_INIT = 0.0
LAMBDA_LR = 0.01
LAMBDA_MAX = 10.0

# REWARD SETTINGS
REWARD_COVERAGE = 1.0
REWARD_DANGER = -1.0
REWARD_FUEL = -0.01

SEED = 42    # RANDOM SEED

print("config.py loaded successfully")
print(f"Number Of Satellites : {N_SATELLITES}")
print(f"Maximum Debris in Orbit : {MAX_DEBRIS}")
print(f"Episode : {EPISODE_LENGTH}  Total steps = {EPISODE_LENGTH * TIMESTEP_SECONDS / 60:.0f} minutes")

