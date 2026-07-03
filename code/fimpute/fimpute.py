# --- FEATURES --- #
# -----------------------------------------------------------------------------
# NOTE: The following statistics includes the test data.
# feature_001: 1, 2, 3, 4, 5   [11.9% missing]
# feature_002: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10   [11.7% missing]
# feature_003: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10   [11.7% missing (same rows as feature_002)]
# feature_004: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10   [0.9% missing]
# feature_005: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10   [2.1% missing]
# feature_006: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10   [0.9% missing]
# feature_007: 0, 1 (binary)   [2.6% missing]
# feature_008: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10   [2.7% missing]
# feature_009: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10   [2.7% missing] (same rows as feature_008)]
# feature_010: 0, 1, 2, ..., 39128914   [2.7% missing] (same rows as feature_008)]
# feature_011: 0, 1 (binary)   [2.6% missing]
# feature_012: 0, 1, 2, 3   [4.6% missing]
# feature_013: 0, 1 (binary)   [2.6% missing] (same rows as feature_011)
# feature_014: 0, 1 (binary)   [2.6% missing]
# feature_015: -7.0 - 898.0   [0.0% missing]
# feature_016: 0, 1, 2, ..., 581   [0.0% missing]
# feature_017: 0.0 - 24.0   [10.9% missing]
# feature_018: 0.0 - 1.0   [10.9% missing]
# -----------------------------------------------------------------------------

# --- REFERENCES --- #
# miceforest: https://github.com/AnotherSamWilson/miceforest

# --- IMPORTS --- #
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import miceforest as mf

# --- CONSTANTS --- #
MICE_NUM_DATASETS = 10
MICE_NUM_ITERATIONS = 5
TRAIN_OUTPUT_FILENAME = 'train-fi.csv'
TEST_OUTPUT_FILENAME = 'test-fi.csv'

# --- FILES --- #
train_filename = '../../data/mercor-cheating-detection/train.csv'
test_filename = '../../data/mercor-cheating-detection/test.csv'

# --- DATA --- #

# Load training and test data.
train_df = pd.read_csv(train_filename)
test_df = pd.read_csv(test_filename)

# Extract features.
train_feat_df = train_df.iloc[:, 1:-2]
test_feat_df = test_df.iloc[:, 1:]

# --- PREPROCESS DATA --- #
train_features = train_feat_df.values
test_features = test_feat_df.values

# Pre-process binary features.
# NOTE: This includes the last feature, which is real-valued in [0,1].
for index in (6, 10, 12, 13, 17):
    pass

# Pre-process ordinal features.
for index in (0, 1, 2, 3, 4, 5, 7, 8, 11):
    encoder = OrdinalEncoder()
    train_features[:, index:index+1] = encoder.fit_transform(train_features[:, index:index+1])
    test_features[:, index:index+1] = encoder.transform(test_features[:, index:index+1])

# Pre-process big-integer features.
for index in (9,):
    pass
    train_features[:, index:index+1] = np.log(1 + train_features[:, index:index+1])
    test_features[:, index:index+1] = np.log(1 + test_features[:, index:index+1])

# Pre-process real-valued features.
for index in (9, 14, 15, 16):
    scaler = StandardScaler()
    train_features[:, index:index+1] = scaler.fit_transform(train_features[:, index:index+1])
    test_features[:, index:index+1] = scaler.transform(test_features[:, index:index+1])

# --- IMPUTATION --- #
print("Imputing features...")
start_time = time.time()

# Create imputation kernel.
imp_kernel = mf.ImputationKernel(
    train_feat_df,
    num_datasets=MICE_NUM_DATASETS,
    random_state=123
)

# Run the MICE algorithm for 5 iterations on each of the datasets
imp_kernel.mice(MICE_NUM_ITERATIONS)

# Complete datasets.
train_feat_df_completed = imp_kernel.complete_data()
test_feat_df_imputed \
    = imp_kernel.impute_new_data(new_data=test_feat_df).complete_data()

# Store completed datasets.
train_df.update(pd.DataFrame(train_feat_df_completed,
                             columns=train_df.columns[1:-2]))
test_df.update(pd.DataFrame(test_feat_df_imputed,
                            columns=test_df.columns[1:]))
train_df.to_csv(TRAIN_OUTPUT_FILENAME, index=False)
test_df.to_csv(TEST_OUTPUT_FILENAME, index=False)

end_time = time.time()
print(f'Time elapsed: {end_time - start_time} seconds.')
