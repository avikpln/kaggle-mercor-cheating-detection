import pandas as pd

TRAIN_DATA_PATH = "data/train.csv"
TEST_DATA_PATH = "data/test.csv"
SOCIAL_GRAPH_PATH = "data/social_graph.csv"


def load_train_data():
    return pd.read_csv(TRAIN_DATA_PATH)


def load_test_data():
    return pd.read_csv(TEST_DATA_PATH)


def load_social_graph():
    return pd.read_csv(SOCIAL_GRAPH_PATH)
