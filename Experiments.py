
# %%   
# Run example from their notebook
import time
import os
import torch
import torch_geometric.transforms as T
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.progress.progress_bar import ProgressBar

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_score,

    recall_score,
)

from src import *



# from cancernet.arch import PNet
# from cancernet.util import ProgressBar, InMemoryLogger, get_roc
# from cancernet import PnetDataSet, ReactomeNetwork
# from cancernet.dataset import get_layer_maps

# %%
## Load Reactome pathways
reactome_kws = dict(
    reactome_base_dir=os.path.join("lib", "cancer-net", "data", "reactome"),
    relations_file_name="ReactomePathwaysRelation.txt",
    pathway_names_file_name="ReactomePathways.txt",
    pathway_genes_file_name="ReactomePathways.gmt",
)
reactome = ReactomeNetwork(reactome_kws)

## Initalise dataset
prostate_root = os.path.join("data", "prostate")
# dataset = GraphDataSet(
#     root=prostate_root,
#     name="prostate_graph_humanbase",
#     edge_tol=0.5, ## Gene connectivity threshold to form an edge connection
#     pre_transform=T.Compose(
#         [T.GCNNorm(add_self_loops=False), T.ToSparseTensor(remove_edge_index=False)]
#     ),
# )


dataset = PnetDataSet(
    root=prostate_root
)


# loads the train/valid/test split from pnet
splits_root = os.path.join(prostate_root, "splits")
dataset.split_index_by_file(
    train_fp=os.path.join(splits_root, "training_set_0.csv"),
    valid_fp=os.path.join(splits_root, "validation_set.csv"),
    test_fp=os.path.join(splits_root, "test_set.csv"),
)

# %%
## Get Reactome masks
maps = get_layer_maps(
    genes=[g for g in dataset.genes],
    reactome=reactome,
    n_levels=6, ## Number of P-NET layers to include
    direction="root_to_leaf",
    add_unk_genes=False,
    verbose=False,
)

# %%
# Set random seed
pl.seed_everything(42, workers=True)

n_epochs = 100
batch_size = 10
lr = 0.001

num_workers = 0
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(dataset.train_idx),
    num_workers=num_workers,
)
valid_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(dataset.valid_idx),
    num_workers=num_workers,
)

# %%
model = PNet(
    layers=maps,
    num_genes=maps[0].shape[0], # 9054
    lr=lr
)

print("Number of params:",sum(p.numel() for p in model.parameters()))
logger = WandbLogger()
pbar = ProgressBar()

t0 = time.time()
trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=n_epochs,
    callbacks=pbar,
    # logger=logger,
    # deterministic=True,
)
trainer.fit(model, train_loader, valid_loader)
print(f"Training took {time.time() - t0:.1f} seconds.")
# %%


# %%
fpr_train, tpr_train, train_auc, _, _ = get_metrics(model, train_loader,exp=False)
fpr_valid, tpr_valid, valid_auc, ys, outs = get_metrics(model, valid_loader,exp=False)

print("validation")
print("accuracy", accuracy_score(ys, outs[:, 1] > 0.5))
print("auc", valid_auc)
print("aupr", average_precision_score(ys, outs[:, 1]))
print("f1", f1_score(ys, outs[:, 1] > 0.5))
print("precision", precision_score(ys, outs[:, 1] > 0.5))
print("recall", recall_score(ys, outs[:, 1] > 0.5))

test_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(dataset.test_idx),
    drop_last=True,
)
fpr_test, tpr_test, test_auc, ys, outs = get_metrics(model, test_loader,exp=False)

print("test")
print("accuracy", accuracy_score(ys, outs[:, 1] > 0.5))
print("auc", test_auc)
print("aupr", average_precision_score(ys, outs[:, 1]))
print("f1", f1_score(ys, outs[:, 1] > 0.5))
print("precision", precision_score(ys, outs[:, 1] > 0.5))
print("recall", recall_score(ys, outs[:, 1] > 0.5))

fig, ax = plt.subplots()
ax.plot(fpr_train, tpr_train, lw=2, label="train (area = %0.3f)" % train_auc)
ax.plot(fpr_valid, tpr_valid, lw=2, label="validation (area = %0.3f)" % valid_auc)
ax.plot(fpr_test, tpr_test, lw=2, label="test (area = %0.3f)" % test_auc)
ax.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Receiver operating characteristic")
ax.legend(loc="lower right", frameon=False)


# %%
# experiment with random graphs etc. 
import random

def create_random_graph(original_graph):
    num_nodes = len(original_graph.nodes)
    num_edges = len(original_graph.edges)

    # Create an empty graph
    random_graph = nx.DiGraph()

    # Add nodes
    random_graph.add_nodes_from(range(num_nodes))

    # Add random edges
    while len(random_graph.edges) < num_edges:
        source = random.randint(0, num_nodes - 1)
        target = random.randint(0, num_nodes - 1)
        if source != target and not random_graph.has_edge(source, target):
            random_graph.add_edge(source, target)

    return random_graph


def remove_edges_by_pathway(graph, pathway_name):
    edges_to_remove = [(u, v) for u, v, d in graph.edges(data=True) if pathway_name in d.get('pathway', '')]
    graph.remove_edges_from(edges_to_remove)

def visualize_graph(graph, title="Graph"):
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=50, font_size=8, node_color='skyblue', edge_color='gray')
    plt.title(title)
    plt.show()

def visualize_tree(tree, title="Tree"):
    pos = nx.spring_layout(tree)
    nx.draw(tree, pos, with_labels=True, node_size=50, font_size=8, node_color='lightgreen', edge_color='gray')
    plt.title(title)
    plt.show()


# %%
# Example usage
random_graph = create_random_graph(reactome.netx)
visualize_graph(random_graph, title="Random Graph")
visualize_graph(reactome.netx, title="Reactome Graph")

copy_graph = reactome.netx.copy()
pathway_name_to_remove = "R-HSA-69306"
remove_edges_by_pathway(copy_graph, pathway_name_to_remove)
visualize_graph(copy_graph, title="Reactome Graph")

# %%
visualize_tree(reactome.get_tree(), title="Reactome Tree")

# %%import networkx as nx
import random
import numpy as np

class Randomized_reactome(ReactomeNetwork):
    def __init__(self, reactome_kws):
        super().__init__(reactome_kws)
        self.netx = self.create_random_graph(self.netx)

    def create_random_graph(self, original_graph):
        num_nodes = len(original_graph.nodes)
        num_edges = len(original_graph.edges)
        random_graph = nx.DiGraph()

        # Map integer node labels to original string labels
        node_labels = list(original_graph.nodes)
        random_graph.add_nodes_from(node_labels)

        # Preserve the root node and its connections
        root_node = "root"
        root_connections = list(original_graph.successors(root_node))
        random_graph.add_node(root_node)
        for target in root_connections:
            random_graph.add_edge(root_node, target)

        # Add random edges while maintaining a similar degree distribution
        degree_sequence = [d for n, d in original_graph.degree()]
        while len(random_graph.edges) < num_edges:
            source = random.choice(node_labels)
            target = random.choice(node_labels)
            if source != target and not random_graph.has_edge(source, target):
                random_graph.add_edge(source, target)

        return random_graph


# Example usage
reactome_kws = dict(
    reactome_base_dir=os.path.join("lib", "cancer-net", "data", "reactome"),
    relations_file_name="ReactomePathwaysRelation.txt",
    pathway_names_file_name="ReactomePathways.txt",
    pathway_genes_file_name="ReactomePathways.gmt",
)

randomized_reactome = Randomized_reactome(reactome_kws)

# Build PNet from the randomized graph
randomized_maps = get_layer_maps(
    genes=[g for g in dataset.genes],
    reactome=randomized_reactome,
    n_levels=6,
    direction="root_to_leaf",
    add_unk_genes=False,
    verbose=False,
)

randomized_model = PNet(
    layers=randomized_maps,
    num_genes=randomized_maps[0].shape[0],
    lr=0.001
)

# %%
# Train the randomized model (similar to your existing code)
print("Train")
trainer.fit(randomized_model, train_loader, valid_loader)


def compare_graph_metrics(original_graph, randomized_graph):
    metrics = {
        "Degree Distribution": nx.degree_histogram,
        "Clustering Coefficient": nx.average_clustering,
        "Betweenness Centrality": nx.betweenness_centrality,
    }

    results = {}
    for metric_name, metric_func in metrics.items():
        original_metric = metric_func(original_graph)
        randomized_metric = metric_func(randomized_graph)
        results[metric_name] = (original_metric, randomized_metric)

    return results

def plot_degree_distribution(original_graph, randomized_graph):
    original_degrees = nx.degree_histogram(original_graph)
    randomized_degrees = nx.degree_histogram(randomized_graph)

    plt.figure(figsize=(10, 5))
    plt.plot(original_degrees, label="Original Graph")
    plt.plot(randomized_degrees, label="Randomized Graph")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title("Degree Distribution")
    plt.legend()
    plt.show()

def calculate_edge_overlap(original_graph, randomized_graph):
    original_edges = set(original_graph.edges)
    randomized_edges = set(randomized_graph.edges)
    overlap = original_edges.intersection(randomized_edges)
    overlap_ratio = len(overlap) / len(original_edges)
    return overlap_ratio

# Compare graph metrics
original_graph = reactome.netx

metrics_comparison = compare_graph_metrics(original_graph, randomized_reactome.netx)
plot_degree_distribution(original_graph, randomized_reactome.netx)

# Calculate edge overlap
overlap_ratio = calculate_edge_overlap(original_graph, randomized_reactome.netx)
print(f"Edge Overlap Ratio: {overlap_ratio:.2f}")

# Visualize graphs
visualize_graph(original_graph, title="Original Reactome Graph")
visualize_graph(randomized_reactome.netx, title="Randomized Reactome Graph")
