# %%
# Import necessary libraries
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

# %%
# Define functions
def run_training_and_evaluation(model,
    train_loader,
    valid_loader,
    num_runs,
    n_epochs,
    model_save_dir,
    save_fig=True,
    figure_save_path="figures/roc-curve-default.png"
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
n_epochs = 5
batch_size = 10
lr = 0.001
num_workers = 0
num_random_graphs = 1
num_runs = 1

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

# %%
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
    batch_size=batch_size,
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

# %%
# Loop to generate and evaluate multiple random graphs
device = torch.device("cuda")
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

    # Use the function to train and evaluate the randomized model
    results.append(
        run_training_and_evaluation(randomized_model, 
                                train_loader,
                                valid_loader,
                                num_runs=1,
                                n_epochs=n_epochs,
                                model_save_dir="models",
                                save_fig=False)[0]
    )
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
plt.savefig("figures/roc_curve_randomized.png")
plt.show()

# %%
# Print results
for i, result in enumerate(results):
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
figure_save_path = "figures/roc_curve_fullNN_featureLayer.png"

# Run the training and evaluation loop
run_training_and_evaluation(flattendNN, train_loader, valid_loader, 
                            num_runs=num_runs, 
                            n_epochs=n_epochs, 
                            model_save_dir=model_save_dir, figure_save_path=figure_save_path)

# %%
# Run the training and evaluation loop for the full model
full_model = FullyConnectedNet(num_genes=maps[0].shape[0], num_features=3)
figure_save_path = "figures/roc_curve_fullNN_featureLayer_flattened.png"
# Run the training and evaluation loop
run_training_and_evaluation(full_model, train_loader, valid_loader, 
                            num_runs=num_runs, 
                            n_epochs=n_epochs, 
                            model_save_dir=model_save_dir, figure_save_path=figure_save_path)
