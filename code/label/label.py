# --- STATISTICS --- #
# - Total number of exaples: 272819 (100%) >>> len(train_features)
# - Labeled examples: 112966 (41.4%) >>> len(train_labels[~np.isnan(train_labels)])
# - Positive examples: 34431 (12.62%) >>> len(train_labels[train_labels == 1.0])
# - Negative examples: 78535 (28.79%) >>> len(train_labels[train_labels == 0.0])
# - Unlabeled examples: 159853 (58.6%) >>> len(train_labels[np.isnan(train_labels)])
# - Examples in social graph: 197920 (72.55%) >>> len([u for u in set(train_users) if u in social_graph])
# - Labeled examples in social graph: 82153 (30.11%) >>> len([u for u in set(train_users[~np.isnan(train_labels)]) if u in social_graph])
# - Positive examples in social graph: 22049 (8.08%) >>> len([u for u in set(train_users[train_labels == 1.0]) if u in social_graph])
# - Negative examples in social graph: 60104 (22.03%) >>> len([u for u in set(train_users[train_labels == 0.0]) if u in social_graph])
# - Uabeled examples in social graph: 115767 (41.43%) >>> len([u for u in set(train_users[np.isnan(train_labels)]) if u in social_graph])
# - Examples outside social graph: 74899 (27.45%) >>> len([u for u in set(train_users) if u not in social_graph])
# - Labeled examples outside social graph: 30813 (11.29%) >>> len([u for u in set(train_users[~np.isnan(train_labels)]) if u not in social_graph])
# - Positive examples outside social graph: 12382 (4.54%) >>> len([u for u in set(train_users[train_labels == 1.0]) if u not in social_graph])
# - Negative examples outside social graph: 18431 (6.76%) >>> len([u for u in set(train_users[train_labels == 0.0]) if u not in social_graph])
# - Uabeled examples outside social graph: 44086 (16.16%) >>> len([u for u in set(train_users[np.isnan(train_labels)]) if u not in social_graph])

# --- IMPORTS --- #
import time
from collections import defaultdict
import pandas as pd
import numpy as np
import h5py
import pickle

import sklearn
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import FeaturePropagation
from torch_geometric.nn.models import GraphSAGE
from torch.nn import Linear, BatchNorm1d, MaxPool1d
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

import keras
import keras.ops

# --- CONSTANTS --- #
LABELS_PROP_FILENAME = 'labels-prop.hdf5'
LABELS_PRED_FILENAME = 'labels-pred.hdf5'

# --- FILES --- #
train_filename = './train-fi.csv'#'./train-ppfi.csv'
test_filename = './test-fi.csv'#'./test-ppfi.csv'
graph_filename = '../../data/mercor-cheating-detection/social_graph.csv'

# --- DATA --- #

# Load training data.
train_df = pd.read_csv(train_filename)
train_data = train_df.values

# Load test data.
test_df = pd.read_csv(test_filename)
test_data = test_df.values

# Extract training users, features, and labels.
train_users = train_data[:, 0]
train_features = np.float64(train_data[:, 1:-2])
train_labels = np.float64(train_data[:, -1])

# Extract test users and features.
test_users = test_data[:, 0]
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

# --- GRAPH --- #

# Load graph data.
graph_raw = pd.read_csv(graph_filename).values
social_graph_raw = defaultdict(set)
for u, v in graph_raw:
    social_graph_raw[u].add(v)
    social_graph_raw[v].add(u)
to_degree_raw = {u: len(social_graph_raw[u]) for u in social_graph_raw}

# Divide the training data into four disjoint groups:
# (1) star users - train users inside the social graph w/ degree > 1
# (2) moon_users - train users inside the social graph w/ degree = 1 (no isolated nodes in social graph!)
# (3) pool users - labeled train users outside the social graph
# (4) snowflakes - unlabeled train users outside the social graph
# STAR
star_inds = [i for i, u in enumerate(train_users) if u in social_graph_raw and to_degree_raw[u] > 0]
star_users = train_users[star_inds]
num_star_users = len(star_users)
star_features = train_features[star_inds, :]
star_labels = train_labels[star_inds]
# MOON
moon_inds = [i for i, u in enumerate(train_users) if u in social_graph_raw and to_degree_raw[u] == 1]
moon_users = train_users[moon_inds]
moon_features = train_features[moon_inds, :]
moon_labels = train_labels[moon_inds]

count = defaultdict(int)
moon_users_set = set(moon_users)
moon_to_index = {m: i for i, m in enumerate(moon_users)}
for i, u in enumerate(moon_users):
    label = moon_labels[i]
    if np.isnan(label): label = -1
    (v,) = social_graph_raw[u]
    if v in moon_users_set:
        label2 = moon_labels[moon_to_index[v]]
        if np.isnan(label2): label2 = -1
    else:
        if to_degree_raw[v] == 1
            label2 = 17
        else:
            label2 = 18
    count[(label, label2)] += 1
    

# POOL
pool_inds = [i for i, u in enumerate(train_users) \
             if u not in social_graph_raw and ~np.isnan(train_labels[i])]
pool_users = train_users[pool_inds]
pool_features = train_features[pool_inds, :]
pool_labels = train_labels[pool_inds]
# SNOWFLAKE
sf_inds = [i for i, u in enumerate(train_users) \
           if u not in social_graph_raw and np.isnan(train_labels[i])]
sf_users = train_users[sf_inds]
sf_features = train_features[sf_inds, :]
fs_labels = train_labels[sf_inds]

assert False

# Discard from the social graph ghost users with a single connection.
star_users_set = set(star_users)
ghost_users = np.array(
    [u for u in social_graph_raw
     if u not in star_users_set and to_degree_raw[u] > 1],
    dtype=train_users.dtype)
social_users = set(ghost_users).union(star_users_set)
social_graph = {u: set(v for v in social_graph_raw[u] if v in social_users) \
                for u in social_users}

# Assign dummy features to ghost users.
ghost_features = np.full([len(ghost_users), train_features.shape[1]], np.nan)

# Designate features to be used in feature propagation and embeddings.
x_features = np.vstack((star_features,ghost_features))

# Define the edge index.
user_to_index = {
    **{u: i for i, u in enumerate(star_users)},
    **{u: num_star_users + i for i, u in enumerate(ghost_users)}
}
edge_index = torch.tensor(
    np.array([[user_to_index[u], user_to_index[v]] \
              for u in social_graph \
              for v in social_graph[u]]).T,
    dtype=torch.long)

# --- FEATURE PROPAGATION --- #
# REFERENCE: https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.transforms.FeaturePropagation.html

x_fp = torch.tensor(x_features, dtype=torch.float)
data_fp = Data(x=x_fp, edge_index=edge_index)
transform_fp = FeaturePropagation(missing_mask=torch.isnan(data_fp.x))
data_fp = transform_fp(data_fp)
x_features = data_fp.x.detach().numpy()

# --- EMBEDDINGS --- #
# REFERENCE: https://pytorch-geometric.readthedocs.io/en/2.4.0/generated/torch_geometric.nn.models.GraphSAGE.html

# Generate embeddings using the GraphSAGE algorithm.
sage_model = GraphSAGE(in_channels=-1,
                       hidden_channels=128,
                       num_layers=5,
                       out_channels=128)
x = torch.tensor(x_features, dtype=torch.float)
embeddings = sage_model(x, edge_index)

# We don't need no ghost embeddings, so discard them.
star_embeddings = embeddings[:num_star_users, :].detach().numpy()

# --- LABEL PROPAGATION --- #
# REFERENCE: https://medium.com/we-talk-data/pytorch-geometric-tutorial-94af3ae2b8cb
print("Propagating labels...")
start_time = time.time()

# Define a GNN.
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.bn0 = BatchNorm1d(128)
        
        self.conv1 = GCNConv(128, 128)
        self.bn1 = BatchNorm1d(128)
        
        self.conv2 = GATConv(128, 128)
        self.bn2 = BatchNorm1d(128)
        
        self.pool = MaxPool1d(16, stride=16)
        self.linear = Linear(8, 1)

    def forward(self, x, edge_index):
        x = self.bn0(x)
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.bn1(x)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.pool(x)
        x = self.linear(x)
        return x[:, 0]

# Load precomputed propagated labels, if exist.
##try:
##    train_labels = h5py.File(LABELS_PROP_FILENAME, 'r')['train_labels_ds'][:]
##except FileNotFoundError:
if True:
    # Redefine the edge index after discarding ghost information.
    edge_index = torch.tensor(
        np.array([[user_to_index[u], user_to_index[v]] \
                  for u in social_graph if u in star_users_set \
                  for v in social_graph[u] if v in star_users_set]).T,
        dtype=torch.long)

    # Create a Data object.
    x = torch.tensor(star_embeddings, dtype=torch.float)
    y = torch.tensor(star_labels, dtype=torch.float)
    train_mask = ~np.isnan(star_labels)
    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)

    # Initialize the model.
    gnn_model = GNN()

    # Define optimizer and loss function.
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Define scheduler.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.99)

    # Training loop.
    for epoch in range(1000):        
        # Clear gradients from the previous step.
        optimizer.zero_grad()
        
        # Forward pass.
        out = gnn_model(data.x, data.edge_index)
        
        # Compute loss only for training nodes.
        loss = criterion(out[data.train_mask], data.y[data.train_mask])

##        if loss < 1E-2:
##            break
        
        # Backpropagation.
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Print progress every 100 epochs.
        if True: #epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    # Update (nan) train labels of social users to their new value.
    pred = gnn_model(data.x, data.edge_index)
    y[~data.train_mask] = pred[~data.train_mask]
    z = torch.sigmoid(y[:len(graph_user_indexes)])
    train_labels[star_inds] = z.detach().numpy()

    # Store new train labels.
    with h5py.File(LABELS_PROP_FILENAME, 'w') as hdf5_file:
        hdf5_file.create_dataset('train_labels_ds', data = train_labels)

##train_indexes = ~np.isnan(train_labels)
##x_train = embeddings[train_indexes, :].detach().numpy()
##y_train = train_labels[train_indexes]
##x_pred = embeddings[~train_indexes, :].detach().numpy()
##model = keras.Sequential([
##    #keras.layers.BatchNormalization(),
##
##    keras.layers.Dense(units=128),
##    keras.layers.ReLU(),
##    keras.layers.BatchNormalization(),
##
##    keras.layers.Dense(units=128),
##    keras.layers.ReLU(),
##    keras.layers.BatchNormalization(),
##
##    keras.layers.Dense(units=128),
##    keras.layers.ReLU(),
##    keras.layers.BatchNormalization(),
##
##    keras.layers.Dense(units=128),
##    keras.layers.ReLU(),
##    keras.layers.BatchNormalization(),
##    
##    keras.layers.Dense(units=1)
##])
##model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
##              loss = keras.losses.BinaryCrossentropy(from_logits=True),
##              metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
##
### Learning rate scheduling.
##def scheduler(epoch, lr):
##    return lr * 0.99#np.exp(-0.05)
##callback = keras.callbacks.LearningRateScheduler(scheduler)
##
##class_weights = {0: 1, 1: 2}
##model.fit(x=x_train, y=y_train, epochs=1000, batch_size=1024,
##          class_weight=class_weights,
##          #callbacks=[callback],
##          verbose=2)
##y_pred = model.predict(x_pred, verbose=0)
##y_pred = keras.ops.sigmoid(y_pred)
##train_labels[~train_indexes] = y_pred[:, 0]
##
### Store new train labels.
##with h5py.File('labels-sage.hdf5', 'w') as hdf5_file:
##    hdf5_file.create_dataset('train_labels_ds', data = train_labels) 

end_time = time.time()
print(f'Time elapsed: {end_time - start_time} seconds.')
