# %%
# %%
# dfki color map
# def hex_to_rgb(hex):
#     hex = hex.lstrip('#')
#     return tuple(int(hex[i:i+2], 16) / 255.0 for i in (0, 2, 4))

# colors_dict = {
# #    'BRIGHT': '#FFFFFF',
#     'ABISKO_GREEN': '#6ABFA3',
#     'OSAKA_RED': '#EC619F',
#     'ERFOUD_ORANGE': '#F7A712',
#     'GUAM_BLUE': '#1D3A8F',
#     'CRIMSON': '#DC143C',
#     'VIBRANT_PURPLE': '#8A2BE2',
#     'SHADY_SKY_BLUE': '#4f9fa8',
#     'BRIGHT_YELLOW': '#FFD700',
#     # 'YELLOW': '#FFF381',
#     # 'MOON_GREY': '#D7DBDD',
#     # 'DARK': '#06171C',
#     # 'LIGHTER_GREEN': '#98CFBA',
# }
# rgb_colors = [hex_to_rgb(color) for color in colors_dict.values()]
# map = colors.LinearSegmentedColormap.from_list('dfki', rgb_colors, N=len(rgb_colors))
# colormaps.register(cmap=map)
# plt.set_cmap(map)
# %%
# Import necessary libraries
import time
import os
from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import colormaps

import torch
import torch_geometric.transforms as T
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.loader import DataLoader

import pytorch_lightning as pl
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

# %%
# Define functions
def run_training_and_evaluation(model,
    train_loader,
    valid_loader,
    num_runs,
    n_epochs,
    model_save_dir,
    save_fig=True,
    figure_save_path="{}/roc-curve-.format(outfolder)default.png"
    ):

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
        fpr, tpr, auc_value, ys, outs = get_metrics(model, valid_loader, exp=False)
        accuracy = accuracy_score(ys, outs[:, 1] > 0.5)
        precision = precision_score(ys, outs[:, 1] > 0.5)
        recall = recall_score(ys, outs[:, 1] > 0.5)
        f1 = f1_score(ys, outs[:, 1] > 0.5)
        aupr = average_precision_score(ys, outs[:, 1])

        results.append({
            "model": model,
            "accuracy": accuracy,
            "auc": auc_value,
            "fpr": fpr,
            "tpr": tpr,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "aupr": aupr
        })

        # Save the model
        model_path = os.path.join(model_save_dir, f"full_model_{i+1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model {i+1} saved to {model_path}")

    for i, result in enumerate(results):
        print(f"Model {i+1}: Accuracy = {result['accuracy']:.2f}, AUC = {result['auc']:.2f}, precision = {result['precision']:.2f}, recall = {result['recall']:.2f}")

    if save_fig:
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

    return results

# %%
# Declare variables, epochs and iteration numbers
test = True
if test: 
    n_epochs = 100
    batch_size = 10
    lr = 0.001
    num_workers = 0
    num_random_graphs = 1
    num_runs = 1
    outfolder = "test"
    pathobj = Path(outfolder)
    Path.mkdir(pathobj, exist_ok=True)
else:
    n_epochs = 100
    batch_size = 10
    lr = 0.001
    num_workers = 0
    num_random_graphs = 5
    num_runs = 5
    outfolder = "output"
    pathobj = Path(outfolder)
    Path.mkdir(pathobj, exist_ok=True)

# %%
# dataset
## Load Reactome pathways
reactome_kws = dict(
    reactome_base_dir=os.path.join("lib", "cancer-net", "data", "reactome"),
    relations_file_name="ReactomePathwaysRelation.txt",
    pathway_names_file_name="ReactomePathways.txt",
    pathway_genes_file_name="ReactomePathways_human.gmt",
)
reactome = ReactomeNetwork(reactome_kws)

## Initialize dataset
prostate_root = os.path.join("lib", "cancer-net", "data", "prostate")
dataset = PnetDataSet(root=prostate_root)

# Load the train/valid/test split from pnet
splits_root = os.path.join(prostate_root, "splits")
dataset.split_index_by_file(
    train_fp=os.path.join(splits_root, "training_set_0.csv"),
    valid_fp=os.path.join(splits_root, "validation_set.csv"),
    test_fp=os.path.join(splits_root, "test_set.csv"),
)

# Set random seed
pl.seed_everything(0, workers=True)

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
# original model

## Get Reactome masks
maps = get_layer_maps(
    genes=[g for g in dataset.genes],
    reactome=reactome,
    n_levels=6,  # Number of P-NET layers to include
    direction="root_to_leaf",
    add_unk_genes=False,
    verbose=False,
)

# init
original_model = PNet(
    layers=maps,
    num_genes=maps[0].shape[0],  # 9229
    lr=lr
)

original_flat_model = PNet_flatten(
    layers=maps,
    num_genes=maps[0].shape[0],  # 9229
    lr=lr
)

print("Number of params:", sum(p.numel() for p in original_model.parameters()))
logger = TensorBoardLogger(save_dir="tensorboard_log/")
pbar = ProgressBar()

t0 = time.time()
trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=n_epochs,
    callbacks=pbar,
    logger=logger,
)
trainer.fit(original_model, train_loader, valid_loader)
print(f"Training took {time.time() - t0:.1f} seconds.")

t0 = time.time()
trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=n_epochs,
    callbacks=pbar,
    logger=logger,
)
trainer.fit(original_flat_model, train_loader, valid_loader)
print(f"Training took {time.time() - t0:.1f} seconds.")

# %%
fpr_train, tpr_train, train_auc, _, _ = get_metrics(original_model, train_loader, exp=False)
fpr_valid, tpr_valid, valid_auc, ys, outs = get_metrics(original_model, valid_loader, exp=False)

fpr_train_flat, tpr_train_flat, train_auc_flat, _, _ = get_metrics(original_flat_model, train_loader, exp=False)
fpr_valid_flat, tpr_valid_flat, valid_auc_flat, ys, outs = get_metrics(original_flat_model, valid_loader, exp=False)

# save for later
original_fpr, original_tpr, original_auc, original_ys, original_outs = get_metrics(original_model, valid_loader, exp=False)
original_accuracy = accuracy_score(ys, outs[:, 1] > 0.5)
original_flat_fpr, original_flat_tpr, original_flat_auc, original_flat_ys, original_flat_outs = get_metrics(original_flat_model, valid_loader, exp=False)
original_flat_accuracy = accuracy_score(ys, outs[:, 1] > 0.5)

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
fpr_test, tpr_test, test_auc, ys, outs = get_metrics(original_model, test_loader, exp=False)
fpr_test_flat, tpr_test_flat, test_auc_flat, ys_flat, outs_flat = get_metrics(original_flat_model, test_loader, exp=False)

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
ax.plot(fpr_train_flat, tpr_train_flat, lw=2, label="flat: train (area = %0.3f)" % train_auc_flat)
ax.plot(fpr_valid_flat, tpr_valid_flat, lw=2, label="flat: validation (area = %0.3f)" % valid_auc_flat)
ax.plot(fpr_test_flat, tpr_test_flat, lw=2, label="flat: test (area = %0.3f)" % test_auc_flat)
ax.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Receiver operating characteristic")
ax.legend(loc="lower right", frameon=False)
fig.savefig("{}/roc curves_PNET_vs_PNet_flat.png".format(outfolder))

# %%
# Loop to generate and evaluate multiple random graphs
device = torch.device("cuda")
randomized_results = []

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

    # Use the function to train and evaluate the randomized model
    randomized_results.extend(
        run_training_and_evaluation(randomized_model,
                                train_loader,
                                valid_loader,
                                num_runs=1,
                                n_epochs=n_epochs,
                                model_save_dir="models",
                                save_fig=False)
    )

# Calculate mean metrics for randomized models
mean_randomized_metrics = {
    "model": "Randomized PNet",
    "accuracy": sum([r["accuracy"] for r in randomized_results]) / len(randomized_results),
    "auc": sum([r["auc"] for r in randomized_results]) / len(randomized_results),
    "precision": sum([r["precision"] for r in randomized_results]) / len(randomized_results),
    "recall": sum([r["recall"] for r in randomized_results]) / len(randomized_results),
    "f1": sum([r["f1"] for r in randomized_results]) / len(randomized_results),
    "aupr": sum([r["aupr"] for r in randomized_results]) / len(randomized_results)
}

# Plot AUC curves
plt.figure()
# plot orig
plt.plot(original_fpr, original_tpr, label=f"Original Model (AUC = {original_auc:.2f})")
# plot all randoms
for i, result in enumerate(randomized_results):
    plt.plot(result["fpr"], result["tpr"], label=f"Model {i+1} (AUC = {result['auc']:.2f})")

plt.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.savefig("{}/roc_curve_PNet_randomized_network.png".format(outfolder))
plt.show()

# %%
# Print results
for i, result in enumerate(randomized_results):
    print(f"Model {i+1}: Accuracy = {result['accuracy']:.2f}, AUC = {result['auc']:.2f}, precision = {result['precision']:.2f}, recall = {result['recall']:.2f}")

# %%
# Define paths for saving models and figures
model_save_dir = "models"

# %%
# Run the training and evaluation loop for the baseline NN
input_size = dataset[0][0].numel()  # Flatten the input
hidden_size = 128
output_size = 1

flattendNN = FullyConnectedNet_flatten(input_size=input_size, hidden_size=hidden_size, output_size=output_size, lr=lr)
figure_save_path = "{}/roc_curve_fullNN_flattened.png".format(outfolder)

# Run the training and evaluation loop
flattendNN_results = run_training_and_evaluation(flattendNN, train_loader, valid_loader,
                            num_runs=num_runs,
                            n_epochs=n_epochs,
                            model_save_dir=model_save_dir, figure_save_path=figure_save_path)

# Calculate mean metrics for flattened NN
mean_flattendNN_metrics = {
    "model": "Flattened NN",
    "accuracy": sum([r["accuracy"] for r in flattendNN_results]) / len(flattendNN_results),
    "auc": sum([r["auc"] for r in flattendNN_results]) / len(flattendNN_results),
    "precision": sum([r["precision"] for r in flattendNN_results]) / len(flattendNN_results),
    "recall": sum([r["recall"] for r in flattendNN_results]) / len(flattendNN_results),
    "f1": sum([r["f1"] for r in flattendNN_results]) / len(flattendNN_results),
    "aupr": sum([r["aupr"] for r in flattendNN_results]) / len(flattendNN_results)
}
# %%
flattendNN_reg = FullyConnectedNet_flatten(input_size=input_size, hidden_size=hidden_size, 
                                           output_size=output_size, lr=lr, l1_lambda=0.001)
figure_save_path = "{}/roc_curve_fullNN_flattened_regularized.png".format(outfolder)

# Run the training and evaluation loop
flattendNN_results = run_training_and_evaluation(flattendNN_reg, train_loader, valid_loader,
                            num_runs=num_runs,
                            n_epochs=n_epochs,
                            model_save_dir=model_save_dir, figure_save_path=figure_save_path)

# Calculate mean metrics for flattened NN
mean_flattendNN_reg_metrics = {
    "model": "Full NN flattened regularized",
    "accuracy": sum([r["accuracy"] for r in flattendNN_results]) / len(flattendNN_results),
    "auc": sum([r["auc"] for r in flattendNN_results]) / len(flattendNN_results),
    "precision": sum([r["precision"] for r in flattendNN_results]) / len(flattendNN_results),
    "recall": sum([r["recall"] for r in flattendNN_results]) / len(flattendNN_results),
    "f1": sum([r["f1"] for r in flattendNN_results]) / len(flattendNN_results),
    "aupr": sum([r["aupr"] for r in flattendNN_results]) / len(flattendNN_results)
}



# %%
# Run the training and evaluation loop for the full model
full_model = FullyConnectedNet(num_genes=maps[0].shape[0], num_features=3)
figure_save_path = "{}/roc_curve_fullNN_featureLayer.png".format(outfolder)
# Run the training and evaluation loop
full_model_results = run_training_and_evaluation(full_model, train_loader, valid_loader,
                            num_runs=num_runs,
                            n_epochs=n_epochs,
                            model_save_dir=model_save_dir, figure_save_path=figure_save_path)

# Calculate mean metrics for full model
mean_full_NN_metrics = {
    "model": "Full NN featureLayer",
    "accuracy": sum([r["accuracy"] for r in full_model_results]) / len(full_model_results),
    "auc": sum([r["auc"] for r in full_model_results]) / len(full_model_results),
    "precision": sum([r["precision"] for r in full_model_results]) / len(full_model_results),
    "recall": sum([r["recall"] for r in full_model_results]) / len(full_model_results),
    "f1": sum([r["f1"] for r in full_model_results]) / len(full_model_results),
    "aupr": sum([r["aupr"] for r in full_model_results]) / len(full_model_results)
}

# %%
# %%
# Run the training and evaluation loop for the full model
full_model_reg = FullyConnectedNet(num_genes=maps[0].shape[0], num_features=3, l1_lambda=0.00001)
figure_save_path = "{}/roc_curve_fullNN_featureLayer_regularized.png".format(outfolder)
# Run the training and evaluation loop
full_model_results = run_training_and_evaluation(full_model_reg, train_loader, valid_loader,
                            num_runs=num_runs,
                            n_epochs=n_epochs,
                            model_save_dir=model_save_dir, figure_save_path=figure_save_path)

# Calculate mean metrics for full model
mean_full_NN_reg_metrics = {
    "model": "Full NN featureLayer regularized",
    "accuracy": sum([r["accuracy"] for r in full_model_results]) / len(full_model_results),
    "auc": sum([r["auc"] for r in full_model_results]) / len(full_model_results),
    "precision": sum([r["precision"] for r in full_model_results]) / len(full_model_results),
    "recall": sum([r["recall"] for r in full_model_results]) / len(full_model_results),
    "f1": sum([r["f1"] for r in full_model_results]) / len(full_model_results),
    "aupr": sum([r["aupr"] for r in full_model_results]) / len(full_model_results)
}

# %%
# Collect all metrics
all_metrics = [
    {
        "model": "Original Model",
        "accuracy": original_accuracy,
        "auc": original_auc,
        "aupr": average_precision_score(original_ys, original_outs[:, 1]),
        "f1": f1_score(original_ys, original_outs[:, 1] > 0.5),
        "precision": precision_score(original_ys, original_outs[:, 1] > 0.5),
        "recall": recall_score(original_ys, original_outs[:, 1] > 0.5)
    },
    {
        "model": "Original Flat Model",
        "accuracy": original_flat_accuracy,
        "auc": original_flat_auc,
        "aupr": average_precision_score(original_flat_ys, original_flat_outs[:, 1]),
        "f1": f1_score(original_flat_ys, original_flat_outs[:, 1] > 0.5),
        "precision": precision_score(original_flat_ys, original_flat_outs[:, 1] > 0.5),
        "recall": recall_score(original_flat_ys, original_flat_outs[:, 1] > 0.5)
    },
    mean_randomized_metrics,
    mean_flattendNN_metrics,
    mean_flattendNN_reg_metrics,
    mean_full_NN_metrics, 
    mean_full_NN_reg_metrics
]

# %%
# Write metrics to CSV
df = pd.DataFrame(all_metrics).round(3)
df.to_csv("{}/metrics_report.csv".format(outfolder))

print("Metrics report saved to metrics_report.csv")
