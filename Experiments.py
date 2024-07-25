
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
    logger=logger,
    # deterministic=True,
)
trainer.fit(model, train_loader, valid_loader)
print(f"Training took {time.time() - t0:.1f} seconds.")
# %%
