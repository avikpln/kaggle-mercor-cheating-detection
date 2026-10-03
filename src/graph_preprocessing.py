# -- Imports --
import os
import pickle

import networkx as nx

from data import load_social_graph

# Path to the cached graph components object.
GRAPH_COMPONENTS_PATH = "src/graph_components.pkl"

# -- Graph Components Construction --

def build_graph_components():
    social_graph = load_social_graph()
    digraph = nx.from_pandas_edgelist(
        social_graph,
        source="user_a",
        target="user_b",
        create_using=nx.DiGraph,
    )

    connected_components = nx.weakly_connected_components(digraph)
    graph_components = [
        {uh: set(nx.neighbors(digraph, uh)) for uh in cc}
        for cc in connected_components
    ]

    return graph_components

def get_graph_components():
    if os.path.exists(GRAPH_COMPONENTS_PATH):
        with open(GRAPH_COMPONENTS_PATH, "rb") as f:
            return pickle.load(f)

    components = build_graph_components()

    with open(GRAPH_COMPONENTS_PATH, "wb") as f:
        pickle.dump(components, f)

    return components
