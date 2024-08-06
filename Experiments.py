
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
    pathway_genes_file_name="ReactomePathways_human.gmt",
)
reactome = ReactomeNetwork(reactome_kws)

## Initalise dataset
prostate_root = os.path.join("lib", "cancer-net", "data", "prostate")
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
original_model = PNet(
    layers=maps,
    num_genes=maps[0].shape[0], # 9054
    lr=lr
)

print("Number of params:",sum(p.numel() for p in original_model.parameters()))
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
trainer.fit(original_model, train_loader, valid_loader)
print(f"Training took {time.time() - t0:.1f} seconds.")
# %%


# %%
fpr_train, tpr_train, train_auc, _, _ = get_metrics(original_model, train_loader,exp=False)
fpr_valid, tpr_valid, valid_auc, ys, outs = get_metrics(original_model, valid_loader,exp=False)
# save for later
original_fpr, original_tpr, original_auc, original_ys, original_outs = get_metrics(original_model, valid_loader,exp=False)
original_accuracy = accuracy_score(ys, outs[:, 1] > 0.5)

print("validation")
print("accuracy", original_accuracy)
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
fpr_test, tpr_test, test_auc, ys, outs = get_metrics(original_model, test_loader,exp=False)

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

# %%import networkx as nx

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
        stubs = []
        for node, degree in original_graph.degree():
            stubs.extend([node] * degree)

        random.shuffle(stubs)
        while stubs:
            source = stubs.pop()
            target = stubs.pop()
            if source != target and not random_graph.has_edge(source, target):
                random_graph.add_edge(source, target)

        # Ensure the random graph has the same number of edges as the original graph
        while len(random_graph.edges) < num_edges:
            source = random.choice(node_labels)
            target = random.choice(node_labels)
            if source != target and not random_graph.has_edge(source, target):
                random_graph.add_edge(source, target)

        # Preserve the hierarchical structure
        layers = get_layers_from_net(original_graph, n_levels=6)
        for layer in layers:
            for pathway, genes in layer.items():
                for gene in genes:
                    if not random_graph.has_edge(pathway, gene):
                        random_graph.add_edge(pathway, gene)

        return random_graph

# Example usage
reactome_kws = dict(
    reactome_base_dir=os.path.join("lib", "cancer-net", "data", "reactome"),
    relations_file_name="ReactomePathwaysRelation.txt",
    pathway_names_file_name="ReactomePathways.txt",
    pathway_genes_file_name="ReactomePathways_human.gmt",
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


# Train the randomized model (similar to your existing code)
# print("Train")
# trainer.fit(randomized_model, train_loader, valid_loader)

# %%


# %%
# Loop to generate and evaluate multiple random graphs
device = torch.device("cuda")
num_random_graphs = 5
n_epochs = 10
results = []

for i in range(num_random_graphs):
    print(f"Generating random graph {i+1}/{num_random_graphs}")
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

    # Train the randomized model
    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=n_epochs,
        logger=WandbLogger(project="randomized_reactome"),
    )
    trainer.fit(randomized_model, train_loader, valid_loader)

    # Evaluate the model
    fpr, tpr, auc_value, ys, outs = get_metrics(randomized_model, valid_loader, seed=1, exp=True, takeLast=False)
    accuracy = accuracy_score(ys, outs[:, 1] > 0.5)
    results.append({
        "model": randomized_model,
        "accuracy": accuracy,
        "auc": auc_value,
        "fpr": fpr,
        "tpr": tpr
    })

    # Save the model
    model_path = f"models/randomized_model_{i+1}.pth"
    torch.save(randomized_model.state_dict(), model_path)
    print(f"Model {i+1} saved to {model_path}")

# Plot AUC curves
plt.figure()
# plot orig
plt.plot(original_fpr, original_tpr, label=f"Original Model (AUC = {original_auc:.2f})")
# plot all randoms 
for i, result in enumerate(results):
    plt.plot(result["fpr"], result["tpr"], label=f"Model {i+1} (AUC = {result['auc']:.2f})")
plt.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()

# Print results
for i, result in enumerate(results):
    print(f"Model {i+1}: Accuracy = {result['accuracy']:.2f}, AUC = {result['auc']:.2f}")
