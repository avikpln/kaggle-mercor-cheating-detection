# --- STRATEGY --- #
# (1) Use MICE-forest feature imputation to learn missing features for all the train users. [✓]
# (2) Learn labeles for unlabeled users in social graph by using a GCN/GAT + labeled users in social graph. [✓]
# (3) Use GraphSAGE to predict labels for unlabeled examples outside the graph. [✓]
# (4) Build fully-connected NN to predict during testing.

# --- TODO --- #
# (1) Custom loss function.
# (2) FCNN + mini-batch mechanism to handle class bias (STEP 5).

# --- IMPORTS --- #
from collections import defaultdict
import json
import pandas as pd
import numpy as np
import h5py

import keras
#import keras.ops as ops
#import tensorflow.keras.backend as K
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

# --- CONSTANTS --- #
DEV_SPLIT = 0.1

# Random seed.
SEED = 0
np.random.seed(SEED)

# --- DATA --- #

# File names.
feature_metadata_filename = '../data/mercor-cheating-detection/feature_metadata.json'
train_filename = './train-fi.csv'
test_filename = './test-fi.csv'
labels_filename = './labels-sage.hdf5'
graph_filename = '../data/mercor-cheating-detection/social_graph.csv'

# --- FEATURE METADATA --- #
with open(feature_metadata_filename, 'r') as infile:
    feature_metadata = json.load(infile)
num_features = len(feature_metadata)
index_to_feature = sorted(feature_metadata.keys())  # feature list
feature_to_index = {f: i for i, f in enumerate(index_to_feature)}

# Load training data.
train_df = pd.read_csv(train_filename)
train_data = train_df.values

# Extract train user, features, and labels.
train_users = train_data[:, 0]
train_features = np.float64(train_data[:, 1:-2])
train_labels = h5py.File(labels_filename, 'r')['train_labels_ds'][:]

### Hard vs. soft labeling.
##naninds = np.isnan(train_labels)
##train_labels[~naninds] = np.float64(train_labels[~naninds] > 0.5)

# Map user hash to its index in the training data.
user_to_index = {u: i for i, u in enumerate(train_users)}

# Load graph data.
graph_raw = pd.read_csv(graph_filename).values
social_graph = defaultdict(set)
for u, v in graph_raw:
    if u in user_to_index and v in user_to_index:
        social_graph[u].add(v)
        social_graph[v].add(u)

# Load test data.
test_df = pd.read_csv(test_filename)
test_data = test_df.values

# Extract test features.
test_features = np.float64(test_data[:, 1:])

# --- PREPROCESSING --- #

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

# --- DATA SPLIT --- #

# Split train dataset into train/dev datasets, based on social graph.
orig_labels = np.float64(train_data[:, -1])
outside_indexes = [u not in social_graph for u in train_users]
labeled_outside_indexes = ~np.isnan(orig_labels) & outside_indexes
labeled_outside_features = train_features[labeled_outside_indexes, :]
labeled_outside_labels = train_labels[labeled_outside_indexes]
the_rest_features = train_features[~labeled_outside_indexes, :]
the_rest_labels = train_labels[~labeled_outside_indexes]
labeled_outside_size = len(labeled_outside_features)
perm_indexes = np.random.permutation(np.arange(labeled_outside_size))
dev_size = np.int64(labeled_outside_size * DEV_SPLIT)
dev_features = labeled_outside_features[perm_indexes[:dev_size], :]
dev_labels = labeled_outside_labels[perm_indexes[:dev_size]]
train_features = np.vstack((
    labeled_outside_features[perm_indexes[dev_size:], :],
    the_rest_features
))
train_labels = np.concatenate((
    labeled_outside_labels[perm_indexes[dev_size:]],
    the_rest_labels
))

# --- MODEL --- #

# The detection model architecture.
detection_model = keras.Sequential([
    #keras.layers.BatchNormalization(),

    keras.layers.Dense(units=256),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(units=256),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(units=256),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(units=256),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(units=256),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(units=1)
])

# Compile model.
detection_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                        loss = keras.losses.BinaryCrossentropy(from_logits=True),
                        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

# Learning rate scheduling.
def scheduler(epoch, lr):
    return lr * 0.99#np.exp(-0.05)
callback = keras.callbacks.LearningRateScheduler(scheduler)

# Fit the model.
class_weights = {0: 1, 1: 2}
detection_model.fit(x=train_features, y=train_labels, epochs=1000,
                    batch_size=512, verbose=2,
                    class_weight=class_weights,
                    #callbacks=[callback],
                    validation_data=(dev_features, dev_labels))

# --- PREDICTION --- #

# Final prediction on test set.
predictions = detection_model.predict(test_features, verbose=0)
predictions = keras.ops.sigmoid(predictions)

# Store test predictions in the designated format.
pd.DataFrame({
    'user_hash': test_data[:, 0],
    'prediction': predictions[:, 0]
}).to_csv('predictions.csv', columns=['user_hash', 'prediction'], index=False)
