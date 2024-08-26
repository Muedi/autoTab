# %%
# Import necessary libraries
import time
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch_geometric.transforms as T
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress.progress_bar import ProgressBar
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
from src import *

# %%
# Define functions
def count_nonzero_weights(model):
    nonzero_weights = sum(p.numel() for p in model.parameters() if p.abs().sum() > 0)
    return nonzero_weights

def run_training_and_evaluation(model,
    train_loader,
    valid_loader,
    num_runs,
    n_epochs,
    model_save_dir,
    save_fig=True,
    figure_save_path="{}/roc-curve-default.png"
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

        nonzero_weights = count_nonzero_weights(model)

        results.append({
            "model": model,
            "accuracy": accuracy,
            "auc": auc_value,
            "fpr": fpr,
            "tpr": tpr,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "aupr": aupr,
            "nonzero_weights": nonzero_weights
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
# Define paths for saving models and figures
model_save_dir = "models"
outfolder = "output_regu"
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
    batch_size=10,
    sampler=SubsetRandomSampler(dataset.train_idx),
    num_workers=0,
)
valid_loader = DataLoader(
    dataset,
    batch_size=10,
    sampler=SubsetRandomSampler(dataset.valid_idx),
    num_workers=0,
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
    lr=0.001
)

print("Number of params:", sum(p.numel() for p in original_model.parameters()))
logger = TensorBoardLogger(save_dir="tensorboard_log/")
pbar = ProgressBar()

t0 = time.time()
trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=100,
    callbacks=pbar,
    logger=logger,
)
trainer.fit(original_model, train_loader, valid_loader)
print(f"Training took {time.time() - t0:.1f} seconds.")

# Evaluate the original model
fpr_train, tpr_train, train_auc, _, _ = get_metrics(original_model, train_loader, exp=False)
fpr_valid, tpr_valid, valid_auc, ys, outs = get_metrics(original_model, valid_loader, exp=False)

# save for later
original_fpr, original_tpr, original_auc, original_ys, original_outs = get_metrics(original_model, valid_loader, exp=False)
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
    batch_size=10,
    sampler=SubsetRandomSampler(dataset.test_idx),
    drop_last=True,
)
fpr_test, tpr_test, test_auc, ys, outs = get_metrics(original_model, test_loader, exp=False)

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
fig.savefig("{}/roc curves_PNET.png".format(outfolder))

# %%
# Run the training and evaluation loop for the full models with different lambda values
# l1_lambda_values = sorted([0.0, 0.001, 0.0003, 0.0005, 0.00001, 0.00003, 0.00005, 0.000001, 0.000003, 0.000005])
l1_lambda_values = sorted([0.0, 0.00001, 0.00002, 0.00003, 0.00005, 0.000001, 0.000003, 0.000005, 0.000007])

#l1_lambda_values = [0.00003, 0.00005]
results_dict = {}

for l1_lambda in l1_lambda_values:
    print(f"Running full model with L1 lambda = {l1_lambda}")
    full_model = FullyConnectedNet(num_genes=maps[0].shape[0], num_features=3, l1_lambda=l1_lambda)
    figure_save_path = f"{outfolder}/roc_curve_fullNN_featureLayer_l1_{l1_lambda}.png"

    # Run the training and evaluation loop
    full_model_results = run_training_and_evaluation(full_model, train_loader, valid_loader,
                                num_runs=1,
                                n_epochs=100,
                                model_save_dir=model_save_dir, figure_save_path=figure_save_path)

    # Calculate mean metrics for full model
    mean_full_model_metrics = {
        "model": f"Full Model (L1={l1_lambda})",
        "l1_lambda": l1_lambda,
        "accuracy": sum([r["accuracy"] for r in full_model_results]) / len(full_model_results),
        "auc": sum([r["auc"] for r in full_model_results]) / len(full_model_results),
        "precision": sum([r["precision"] for r in full_model_results]) / len(full_model_results),
        "recall": sum([r["recall"] for r in full_model_results]) / len(full_model_results),
        "f1": sum([r["f1"] for r in full_model_results]) / len(full_model_results),
        "aupr": sum([r["aupr"] for r in full_model_results]) / len(full_model_results),
        "nonzero_weights": sum([r["nonzero_weights"] for r in full_model_results]) / len(full_model_results)
    }

    results_dict[l1_lambda] = {
        "metrics": mean_full_model_metrics,
        "models": full_model_results  # Store the actual models
    }

# %%
# Collect all metrics
all_metrics = [
    {
        "model": "Original Model",
        "l1_lambda": None,
        "accuracy": original_accuracy,
        "auc": original_auc,
        "aupr": average_precision_score(original_ys, original_outs[:, 1]),
        "f1": f1_score(original_ys, original_outs[:, 1] > 0.5),
        "precision": precision_score(original_ys, original_outs[:, 1] > 0.5),
        "recall": recall_score(original_ys, original_outs[:, 1] > 0.5),
        "nonzero_weights": count_nonzero_weights(original_model)
    },
    *[result["metrics"] for result in results_dict.values()]
]

# %%
# Write metrics to CSV
df = pd.DataFrame(all_metrics).round(6)
df.to_csv("{}/metrics_report.csv".format(outfolder))

print("Metrics report saved to metrics_report.csv")

# %%
# Plot the results
fig, ax = plt.subplots()

# Plot AUC curves
# ax.plot(fpr_train, tpr_train, lw=2, label="train (area = %0.3f)" % train_auc)
ax.plot(fpr_valid, tpr_valid, lw=2, label="validation PNet (area = %0.3f)" % valid_auc)
# ax.plot(fpr_test, tpr_test, lw=2, label="test (area = %0.3f)" % test_auc)

for l1_lambda, result in results_dict.items():
    for model_result in result["models"]:
        fpr, tpr, auc_value, _, _ = get_metrics(model_result["model"], valid_loader, exp=False)
        ax.plot(fpr, tpr, lw=2, label=f"L1={l1_lambda} (area = %0.3f)" % auc_value)

ax.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Receiver operating characteristic")
ax.legend(loc="lower right", frameon=False)
plt.savefig("{}/AUROCs_all.png".format(outfolder))
plt.show()

# %%
# Create facet panel plot
metrics = ["accuracy", "auc", "f1", "precision"] #, "recall", "nonzero_weights"]

g = sns.FacetGrid(df.melt(id_vars=["model", "l1_lambda"], value_vars=metrics, var_name="metric", value_name="value"),
                  col="metric", col_wrap=2, sharey=False, height=4)
g.map(sns.lineplot, "l1_lambda", "value", marker="o", color="#6ABFA3") # absiko green
g.set_titles("{col_name}")
g.set_axis_labels("L1 Regularization (lambda)", "Value")
g.add_legend()

plt.subplots_adjust(top=0.9)
g.fig.suptitle("Model Performance with Different L1 Regularization Values")

plt.savefig("{}/facet_plot_poster.png".format(outfolder))
plt.show()
