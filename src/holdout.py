# -- Import Libraries --
import numpy as np
import pandas as pd

# -- Constants --

# Set random state for reproducibility.
RANDOM_STATE = 42

# Number of users to reserve for the holdout set.
HOLDOUT_SIZE = 3_000

# Path to the source training data. Update this to match your
# local data directory.
TRAIN_DATA_PATH = "data/train.csv"

# Paths to the split output files.
TRAIN_SPLIT_PATH = "data/train_split.csv"
HOLDOUT_SPLIT_PATH = "data/holdout_split.csv"

# -- Holdout Split --

data_train = pd.read_csv(TRAIN_DATA_PATH)

target_name = "is_cheating"
labeled_rows = data_train[data_train[target_name].notna()]

rng = np.random.RandomState(RANDOM_STATE)
holdout_hashes = rng.choice(
    labeled_rows["user_hash"], size=HOLDOUT_SIZE, replace=False
)

holdout_split = data_train[data_train["user_hash"].isin(holdout_hashes)]
train_split = data_train[~data_train["user_hash"].isin(holdout_hashes)]

# -- Save Split Files --

train_split.to_csv(TRAIN_SPLIT_PATH, index=False)
holdout_split.to_csv(HOLDOUT_SPLIT_PATH, index=False)
