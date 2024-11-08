
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

from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks.progress.progress_bar import ProgressBar
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)

from src import *

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

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
pl.seed_everything(0, workers=True)

n_epochs = 10
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
original_model = PNet_flatten(
    layers=maps,
    num_genes=maps[0].shape[0], # 9229
    lr=lr
)

print("Number of params:",sum(p.numel() for p in original_model.parameters()))
# logger = WandbLogger()
logger = TensorBoardLogger(save_dir="tensorboard_log/")
pbar = ProgressBar()

t0 = time.time()
trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=n_epochs,
    callbacks=pbar,
    logger=logger,
    # deterministic=True,
)
trainer.fit(original_model, train_loader, valid_loader)
print(f"Training took {time.time() - t0:.1f} seconds.")

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
        self.netx = self.randomize_edges_by_layer(self.netx)

    def randomize_edges_by_layer(self, original_graph: nx.DiGraph) -> nx.DiGraph:
        """Randomize the order of edges between nodes by layer while ensuring the graph remains connected."""
        # Get the layers of the original graph
        layers = self.get_layers(n_levels=6, direction="root_to_leaf")

        # Create a new graph with the same nodes
        random_graph = nx.DiGraph()
        random_graph.add_nodes_from(original_graph.nodes)

        # Randomize edges within each layer
        for layer in layers:
            pathway_nodes = []
            gene_nodes = []
            for pathway, genes in layer.items():
                for gene in genes:
                    if original_graph.has_edge(pathway, gene):
                        pathway_nodes.append(pathway)
                        gene_nodes.append(gene)
            random.shuffle(gene_nodes)
            for source, target in zip(pathway_nodes, gene_nodes):
                random_graph.add_edge(source, target)

        # Ensure the graph is connected
        random_graph = complete_network(random_graph, n_levels=6)

        return random_graph

    def get_layers(self, n_levels: int, direction: str = "root_to_leaf") -> List[Dict[str, List[str]]]:
        """Generate layers of nodes from root to leaves or vice versa.

        Depending on the direction specified, this function returns the layers of the network.
        """
        if direction == "root_to_leaf":
            net = self.get_completed_network(n_levels)
            layers = get_layers_from_net(net, n_levels)
        else:
            net = self.get_completed_network(5)
            layers = get_layers_from_net(net, 5)
            layers = layers[5 - n_levels : 5]

        # Get the last layer (genes level)
        terminal_nodes = [
            n for n, d in net.out_degree() if d == 0
        ]  # Set of terminal pathways
        # Find genes belonging to these pathways
        genes_df = self.reactome.pathway_genes

        dict = {}
        missing_pathways = []
        for p in terminal_nodes:
            pathway_name = re.sub("_copy.*", "", p)
            genes = genes_df[genes_df["group"] == pathway_name]["gene"].unique()
            if len(genes) == 0:
                missing_pathways.append(pathway_name)
            dict[pathway_name] = genes

        layers.append(dict)
        return layers
# Example usage
reactome_kws = dict(
    reactome_base_dir=os.path.join("lib", "cancer-net", "data", "reactome"),
    relations_file_name="ReactomePathwaysRelation.txt",
    pathway_names_file_name="ReactomePathways.txt",
    pathway_genes_file_name="ReactomePathways_human.gmt",
)

# randomized_reactome = Randomized_reactome(reactome_kws)

# # Build PNet from the randomized graph
# randomized_maps = get_layer_maps(
#     genes=[g for g in dataset.genes],
#     reactome=randomized_reactome,
#     n_levels=6,
#     direction="root_to_leaf",
#     add_unk_genes=False,
#     verbose=False,
# )

# randomized_model = PNet(
#     layers=randomized_maps,
#     num_genes=randomized_maps[0].shape[0],
#     lr=0.001
# )


# Train the randomized model (similar to your existing code)
# print("Train")
# trainer.fit(randomized_model, train_loader, valid_loader)

# %%
# Loop to generate and evaluate multiple random graphs
device = torch.device("cuda")
num_random_graphs = 1
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
        # logger=WandbLogger(project="randomized_reactome"),
        logger=logger
    )
    trainer.fit(randomized_model, train_loader, valid_loader)

    # Evaluate the model
    fpr, tpr, auc_value, ys, outs = get_metrics(randomized_model, valid_loader, seed=1, exp=False, takeLast=False)
    accuracy = accuracy_score(ys, outs[:, 1] > 0.5)
    results.append({
        "model": randomized_model,
        "accuracy": accuracy,
        "auc": auc_value,
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision_score(ys, outs[:, 1] > 0.5),
        "recall": recall_score(ys, outs[:, 1] > 0.5)
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

# %%
# Print results
for i, result in enumerate(results):
    print(f"Model {i+1}: Accuracy = {result['accuracy']:.2f}, AUC = {result['auc']:.2f}, precision = {result['precision']:.2f}, recall = {result['recall']:.2f}")


# TODO:
# - build "more random" network
# - Fully connected NN, regularize weights l1
# - Fully connected NN, regularize nodes? l1 ?
# - Fully connected NN, gets smaller to the end, regularize weights l1
# 
# %% actual random network 
# 6 layers all equally large

# Example usage
num_genes = maps[0].shape[0]  # Example number of genes
num_features = 3  # Example number of features per gene
hidden_size = 128  # Example hidden layer size
output_size = 1  # Example output size

# sparse_model = SparseNN(num_genes=num_genes, num_features=num_features, hidden_size=hidden_size, output_size=output_size, lr=0.001, l1_lambda=0.01)

# trainer = pl.Trainer(max_epochs=100)
# trainer.fit(sparse_model, train_loader, valid_loader)
# trainer.test(sparse_model, test_loader)

# %%
device = torch.device("cuda")
num_sparse_models = 1
n_epochs = 10
results = []

for i in range(num_sparse_models):
    print(f"Running full model {i+1}/{num_sparse_models}")

    full_model = FullyConnectedNet(num_genes=num_genes, num_features=num_features)

    # Train the randomized model
    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=n_epochs,
        # logger=WandbLogger(project="randomized_reactome"),
        logger=logger
    )
    trainer.fit(full_model, train_loader, valid_loader)

    # Evaluate the model
    fpr, tpr, auc_value, ys, outs = get_metrics(full_model, valid_loader, seed=1, exp=False, takeLast=False)
    accuracy = accuracy_score(ys, outs[:, 1] > 0.5)
    results.append({
        "model": full_model,
        "accuracy": accuracy,
        "auc": auc_value,
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision_score(ys, outs[:, 1] > 0.5),
        "recall": recall_score(ys, outs[:, 1] > 0.5)
    })

    # Save the model
    model_path = f"models/full_model_{i+1}.pth"
    torch.save(full_model.state_dict(), model_path)
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

# %%
# Print results
for i, result in enumerate(results):
    print(f"Sparse Model {i+1}: Accuracy = {result['accuracy']:.2f}, AUC = {result['auc']:.2f}, precision = {result['precision']:.2f}, recall = {result['recall']:.2f}")

# %%
# train baseline NN

# Initialize the model
input_size = dataset[0][0].numel()  # Flatten the input
hidden_size = 128
output_size = 1

flattendNN = FullyConnectedNet_flatten(input_size=input_size, hidden_size=hidden_size, output_size=output_size, lr=lr)

# Function to run the training and evaluation loop
def run_training_and_evaluation(model, train_loader, valid_loader, num_runs, n_epochs, model_save_dir, figure_save_path):
    results = []

    for i in range(num_runs):
        print(f"Running full model {i+1}/{num_runs}")

        # Train the model
        trainer = pl.Trainer(
            accelerator="auto",
            max_epochs=n_epochs,
            logger=logger
        )
        trainer.fit(model, train_loader, valid_loader)

        # Evaluate the model
        fpr, tpr, auc_value, ys, outs = get_metrics(full_model, valid_loader, seed=1, exp=False, takeLast=False)
        accuracy = accuracy_score(ys, outs[:, 1] > 0.5)
        results.append({
            "model": model,
            "accuracy": accuracy,
            "auc": auc_value,
            "fpr": fpr,
            "tpr": tpr,
            "precision": precision_score(ys, outs[:, 1] > 0.5),
            "recall": recall_score(ys, outs[:, 1] > 0.5)
        })

        # Save the model
        model_path = os.path.join(model_save_dir, f"full_model_{i+1}.pth")
        torch.save(model.state_dict(), model_path)
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
    plt.savefig(figure_save_path)
    plt.show()



# Define paths for saving models and figures
model_save_dir = "models"
figure_save_path = "figures/roc_curve.png"

# %%
# Run the training and evaluation loop
run_training_and_evaluation(flattendNN, train_loader, valid_loader, num_runs=2, n_epochs=2, model_save_dir=model_save_dir, figure_save_path=figure_save_path)

# %%
run_training_and_evaluation(full_model, train_loader, valid_loader, num_runs=2, n_epochs=2, model_save_dir=model_save_dir, figure_save_path=figure_save_path)

