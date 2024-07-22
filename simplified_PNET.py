# %%
# Utility to make a sparse weight list for a Network.
# https://github.com/zhanglab-aim/cancer-net/blob/main/cancernet/util/tensor.py

import torch

def scatter_nd(
    indices: torch.Tensor, weights: torch.Tensor, shape: tuple
) -> torch.Tensor:
    """For a list of indices and weights, return a sparse matrix of desired shape with
    weights only at the desired indices. Named after `tensorflow.scatter_nd`.
    """
    # Pytorch has scatter_add that does the same thing but only in one dimension
    # need to convert nd-indices to linear, then use Pytorch's function, then reshape

    ind1d = indices[:, 0]
    n = shape[0]
    for i in range(1, len(shape)):
        ind1d = ind1d * shape[i] + indices[:, i]
        n *= shape[i]

    # ensure all tensors are on the same device
    ind1d = ind1d.to(weights.device)

    # generate the flat output, then reshape
    res = weights.new_zeros(n)
    res = res.scatter_add_(0, ind1d, weights).reshape(*shape)
    return res


# %% 
# pythorch lighning stuff to train the model(s)
"""Scaffolding for building PyTorch Lightning modules."""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Tuple, List


class BaseNet(pl.LightningModule):
    """A basic scaffold for our modules, with default optimizer, scheduler, and loss
    function, and simple logging.
    """

    def __init__(self, lr: float = 0.01, scheduler: str="lambda"):
        super().__init__()

        self.lr = lr
        self.scheduler=scheduler

    def configure_optimizers(self) -> Tuple[List, List]:
        """Set up optimizers and schedulers.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if self.scheduler=="lambda":
            lr_lambda = lambda epoch: 1.0 if epoch < 30 else 0.5 if epoch < 60 else 0.1
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        elif self.scheduler=="pnet": ## Take scheduler from pnet
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.25)
        else:
            scheduler=None

        return [optimizer], [scheduler]

    def step(self, batch, kind: str) -> dict:
        """Generic step function that runs the network on a batch and outputs loss
        and accuracy information that will be aggregated at epoch end.

        This function is used to implement the training, validation, and test steps.
        """
        # run the model and calculate loss
        y_hat = self(batch)

        loss = F.nll_loss(y_hat, batch.y)

        # assess accuracy
        pred = y_hat.max(1)[1]
        correct = pred.eq(batch.y).sum().item()

        total = len(batch.y)

        batch_dict = {
            "loss": loss,
            # correct and total will be used at epoch end
            "correct": correct,
            "total": total,
        }
        return batch_dict

    def epoch_end(self, outputs, kind: str):
        """Generic function for summarizing and logging the loss and accuracy over an
        epoch.

        Creates log entries with name `f"{kind}_loss"` and `f"{kind}_accuracy"`.

        This function is used to implement the training, validation, and test epoch-end
        functions.
        """
        with torch.no_grad():
            # calculate average loss and average accuracy
            total_loss = sum(_["loss"] * _["total"] for _ in outputs)
            total = sum(_["total"] for _ in outputs)
            avg_loss = total_loss / total

            correct = sum(_["correct"] for _ in outputs)
            avg_acc = correct / total

        # log
        self.log(f"{kind}_loss", avg_loss)
        self.log(f"{kind}_accuracy", avg_acc)

    def training_step(self, batch, batch_idx) -> dict:
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx) -> dict:
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx) -> dict:
        return self.step(batch, "test")

    def training_epoch_end(self, outputs):
        self.epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self.epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        self.epoch_end(outputs, "test")

# %%
# the  PNET implementation

# import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, ReLU
import pandas as pd 

# from cancernet.arch.base_net import BaseNet
# from cancernet.util import scatter_nd

class FeatureLayer(torch.nn.Module):
    """
    Original notes:
    This layer will take our input data of size `(N_genes, N_features)`, and perform
    elementwise multiplication of the features of each gene. This is effectively
    collapsing the `N_features dimension`, outputting a single scalar latent variable
    for each gene.

    New notes: 
    This is just a full mapping of everything that is put in the network to the genes
    that are present in the pathway data that is used to generate the network. 
    Since its weights are learnable and fully connected, the model learns, which feature 'belongs' to 
    which gene.
    => This is already intesting, it has no regularization, 
    which would make the connections sparse after some learning and more interpretable?

    """

    def __init__(self, num_genes: int, num_features: int):
        super().__init__()
        self.num_genes = num_genes
        self.num_features = num_features
        weights = torch.Tensor(self.num_genes, self.num_features)
        self.weights = nn.Parameter(weights)
        self.bias = nn.Parameter(torch.Tensor(self.num_genes))
        torch.nn.init.xavier_uniform_(self.weights)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        x = x * self.weights
        x = torch.sum(x, dim=-1)
        x = x + self.bias
        return x


class SparseLayer(torch.nn.Module):
    """
    Original notes:
    Sparsely connected layer, with connections taken from pnet.
    
    New notes: 
    As is the network builds itself, with an established mapping. 
    We could build the AutoML as such, that it changes the mappings outside of the arch and not the weights,
    I guess this will be less efficient and likely not implemented by another party,
    but it would give us more control of the graph/pathways I guess.
    """

    def __init__(self, layer_map):
        super().__init__()
        if type(layer_map)==pd.core.frame.DataFrame:
            map_numpy = layer_map.to_numpy()
        else:
            map_numpy=layer_map
        self.register_buffer(
            "nonzero_indices", torch.LongTensor(np.array(np.nonzero(map_numpy)).T)
        )
        self.layer_map = layer_map
        self.shape = map_numpy.shape
        self.weights = nn.Parameter(torch.Tensor(self.nonzero_indices.shape[0], 1))
        self.bias = nn.Parameter(torch.Tensor(self.shape[1]))
        torch.nn.init.xavier_uniform_(self.weights)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        sparse_tensor = scatter_nd(
            self.nonzero_indices, self.weights.squeeze(), self.shape
        )
        x = torch.mm(x, sparse_tensor)
        # no bias yet
        x = x + self.bias
        return x


class PNet(BaseNet):
    """Implementation of the pnet sparse feedforward network in torch. Uses the same
    pytorch geometric dataset as the message passing networks.
    """

    def __init__(
        self,
        layers,
        num_genes: int,
        num_features: int = 3,
        lr: float = 0.001,
        intermediate_outputs: bool = True,
        class_weights: bool=True,
        scheduler: str="lambda"
    ):
        """Initialize.
        :param layers: list of pandas dataframes describing the pnet masks for each
            layer
        :param num_genes: number of genes in dataset
        :param num_features: number of features for each gene
        :param lr: learning rate
        """
        super().__init__(lr=lr,scheduler=scheduler)
        self.class_weights=class_weights
        self.layers = layers
        self.num_genes = num_genes
        self.num_features = num_features
        self.intermediate_outputs = intermediate_outputs
        self.network = nn.ModuleList()
        self.intermediate_outs = nn.ModuleList()
        self.network.append(
            nn.Sequential(FeatureLayer(self.num_genes, self.num_features), nn.Tanh())
        )
        ## Taken from pnet
        self.loss_weights = [2, 7, 20, 54, 148, 400]
        if len(self.layers) > 5:
            self.loss_weights = [2] * (len(self.layers) - 5) + self.loss_weights
        for i, layer_map in enumerate(layers):
            if i != (len(layers) - 1):
                if i == 0:
                    ## First layer has dropout of 0.5, the rest have 0.1
                    dropout = 0.5
                else:
                    dropout = 0.1
                    ## Build pnet layers
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout), SparseLayer(layer_map), nn.Tanh()
                    )
                )
                ## Build layers for intermediate output
                if self.intermediate_outputs:
                    self.intermediate_outs.append(
                        nn.Sequential(nn.Linear(layer_map.shape[0], 1), nn.Sigmoid())
                    )
            else:
                self.network.append(
                    nn.Sequential(nn.Linear(layer_map.shape[0], 1), nn.Sigmoid())
                )

    def forward(self, x):
        """ Forward pass, output a list containing predictions from each
            intermediate layer, which can be weighted differently during
            training & validation """

        y = []
        x = self.network[0](x)
        for aa in range(1, len(self.network) - 1):
            y.append(self.intermediate_outs[aa - 1](x))
            x = self.network[aa](x)
        y.append(self.network[-1](x))

        return y

    def step(self, batch, kind: str) -> dict:
        """Step function executed by lightning trainer module."""
        # run the model and calculate loss
        x,y_true=batch
        y_hat = self(x)

        loss = 0
        if self.class_weights:
            weights=y_true*0.75+0.75
        else:
            weights=None
            
        for aa, y in enumerate(y_hat):
            ## Here we take a weighted average of the preditive outputs. Intermediate layers first
            loss += self.loss_weights[aa] * F.binary_cross_entropy(y, y_true,weight=weights)
        loss /= np.sum(self.loss_weights[aa])

        correct = ((y_hat[-1] > 0.5).flatten() == y_true.flatten()).sum()
        # assess accuracy
        total = len(y_true)
        batch_dict = {
            "loss": loss,
            # correct and total will be used at epoch end
            "correct": correct,
            "total": total,
        }
        return batch_dict


# %%
# PNET dataset class
import os
import time
import copy
import gzip
import logging
import pickle
import json
import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset


# Change this ASAP, weird af....
cached_data = {}  # all data read will be referenced here


def load_data(filename, response=None, selected_genes=None):
    logging.info("loading data from %s," % filename)
    if filename in cached_data:
        logging.info("loading from memory cached_data")
        data = cached_data[filename]
    else:
        data = pd.read_csv(filename, index_col=0)
        cached_data[filename] = data
    logging.info(data.shape)

    if response is None:
        if "response" in cached_data:
            logging.info("loading from memory cached_data")
            labels = cached_data["response"]
        else:
            raise ValueError(
                "abort: must read response first, but can't find it in cached_data"
            )
    else:
        labels = copy.deepcopy(response)

    # join with the labels
    all = data.join(labels, how="inner")
    all = all[~all["response"].isnull()]

    response = all["response"]
    samples = all.index

    del all["response"]
    x = all
    genes = all.columns

    if not selected_genes is None:
        intersect = list(set.intersection(set(genes), selected_genes))
        if len(intersect) < len(selected_genes):
            # raise Exception('wrong gene')
            logging.warning("some genes don't exist in the original data set")
        x = x.loc[:, intersect]
        genes = intersect
    logging.info(
        "loaded data %d samples, %d variables, %d responses "
        % (x.shape[0], x.shape[1], response.shape[0])
    )
    logging.info(len(genes))
    return x, response, samples, genes


def processor(x, data_type):
    if data_type == "mut_important":
        x[x > 1.0] = 1.0
    elif data_type == "cnv_amp":
        x[x <= 0.0] = 0.0
        x[x == 1.0] = 0.0
        x[x == 2.0] = 1.0
    elif data_type == "cnv_del":
        x[x >= 0.0] = 0.0
        x[x == -1.0] = 0.0
        x[x == -2.0] = 1.0
    else:
        raise TypeError("unknown data type '%s' % data_type")
    return x

def get_response(response_filename):
    logging.info("loading response from %s" % response_filename)
    labels = pd.read_csv(response_filename)
    labels = labels.set_index("id")
    if "response" in cached_data:
        logging.warning(
            "response in cached_data is being overwritten by '%s'" % response_filename
        )
    else:
        logging.warning(
            "response in cached_data is being set by '%s'" % response_filename
        )

    cached_data["response"] = labels
    return labels




# complete_features: make sure all the data_types have the same set of features_processing (genes)
def combine(
    x_list,
    y_list,
    rows_list,
    cols_list,
    data_type_list,
    combine_type,
    use_coding_genes_only=None,
):
    cols_list_set = [set(list(c)) for c in cols_list]

    if combine_type == "intersection":
        cols = set.intersection(*cols_list_set)
    else:
        cols = set.union(*cols_list_set)
    logging.debug("step 1 union of gene features", len(cols))

    if use_coding_genes_only is not None:
        assert os.path.isfile(
            use_coding_genes_only
        ), "you specified a filepath to filter coding genes, but the file doesn't exist"
        f = os.path.join(use_coding_genes_only)
        coding_genes_df = pd.read_csv(f, sep="\t", header=None)
        coding_genes_df.columns = ["chr", "start", "end", "name"]
        coding_genes = set(coding_genes_df["name"].unique())
        cols = cols.intersection(coding_genes)
        logging.debug(
            "step 2 intersect w/ coding",
            len(coding_genes),
            "; coding AND in cols",
            len(cols),
        )

    # the unique (super) set of genes
    all_cols = list(cols)

    all_cols_df = pd.DataFrame(index=all_cols)

    df_list = []
    for x, y, r, c, d in zip(x_list, y_list, rows_list, cols_list, data_type_list):
        df = pd.DataFrame(x, columns=c, index=r)
        df = df.T.join(all_cols_df, how="right")
        df = df.T
        logging.info("step 3 fill NA-%s num NAs=" % d, df.isna().sum().sum())
        # IMPORTANT: using features in union will be filled zeros!!
        df = df.fillna(0)
        df_list.append(df)

    all_data = pd.concat(df_list, keys=data_type_list, join="inner", axis=1)

    # put genes on the first level and then the data type
    all_data = all_data.swaplevel(i=0, j=1, axis=1)

    # order the columns based on genes
    # NOTE: sort this for reproducibility; FZZ 2022.10.12
    order = sorted(all_data.columns.levels[0])
    all_data = all_data.reindex(columns=order, level=0)

    x = all_data.values
    # NOTE: only the last y is used; all else are discarded
    reordering_df = pd.DataFrame(index=all_data.index)
    y = reordering_df.join(y, how="left")

    y = y.values
    cols = all_data.columns
    rows = all_data.index
    logging.debug(
        "After combining, loaded data %d samples, %d variables, %d responses "
        % (x.shape[0], x.shape[1], y.shape[0])
    )

    return all_data, x, y, rows, cols


def graph_reader_and_processor(graph_file):
    # load gene graph (e.g., from HumanBase)
    # graph_file = os.path.join(self.root, self.graph_dir, self.gene_graph)
    graph_noext, _ = os.path.splitext(graph_file)
    graph_pickle = graph_noext + ".pkl"

    start_time = time.time()
    if os.path.exists(graph_pickle):
        # load pre-parsed version
        with open(graph_pickle, "rb") as f:
            edge_dict = pickle.load(f)
    else:
        # parse the tab-separated file
        edge_dict = defaultdict(dict)
        with gzip.open(graph_file, "rt") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            f.seek(0, os.SEEK_SET)

            pbar = tqdm.tqdm(
                total=file_size,
                unit_scale=True,
                unit_divisor=1024,
                mininterval=1.0,
                desc="gene graph",
            )
            for line in f:
                pbar.update(len(line))
                elems = line.strip().split("\t")
                if len(elems) == 0:
                    continue

                assert len(elems) == 3

                # symmetrize, since the initial graph contains edges in only one dir
                edge_dict[elems[0]][elems[1]] = float(elems[2])
                edge_dict[elems[1]][elems[0]] = float(elems[2])

            pbar.close()

        # save pickle for faster loading next time
        t0 = time.time()
        print("Caching the graph as a pickle...", end=None)
        with open(graph_pickle, "wb") as f:
            pickle.dump(edge_dict, f, pickle.HIGHEST_PROTOCOL)
        print(f" done (took {time.time() - t0:.2f} seconds).")

    print(f"loading gene graph took {time.time() - start_time:.2f} seconds.")
    return edge_dict


def data_reader(filename_dict,graph=True):
    # sanity checks for filename_dict
    assert "response" in filename_dict, "must parse a response file"
    fd = copy.deepcopy(filename_dict)
    # first get non-tumor genomic/config data types out
    if graph==True:
        ## Only check for graph files if we are loading graph data
        for f in filename_dict.values():
            if not os.path.isfile(f):
                raise FileNotFoundError(f)
        edge_dict = graph_reader_and_processor(graph_file=fd.pop("graph_file"))

    selected_genes = fd.pop("selected_genes")
    if selected_genes is not None:
        selected_genes = pd.read_csv(selected_genes)["genes"]
    use_coding_genes_only = fd.pop("use_coding_genes_only")
    # read the remaining tumor data
    labels = get_response(fd.pop("response"))
    x_list = []
    y_list = []
    rows_list = []
    cols_list = []
    data_type_list = []
    for data_type, filename in fd.items():
        x, y, info, genes = load_data(filename=filename, selected_genes=selected_genes)
        x = processor(x, data_type)
        x_list.append(x)
        y_list.append(y)
        rows_list.append(info)
        cols_list.append(genes)
        data_type_list.append(data_type)
    res = combine(
        x_list,
        y_list,
        rows_list,
        cols_list,
        data_type_list,
        combine_type="union",
        use_coding_genes_only=use_coding_genes_only,
    )
    all_data = res[0]
    response = labels.loc[all_data.index]
    if graph==True:
        return all_data, response, edge_dict
    else:
        return all_data, response


class PnetDataSet(Dataset):
    """ Prostate cancer dataset, used to reproduce https://www.nature.com/articles/s41586-021-03922-4 """
    def __init__(
            self,
            num_features=3,
            root: Optional[str] = "./data/prostate/",
            valid_ratio: float = 0.102,
            test_ratio: float = 0.102,
            valid_seed: int = 0,
            test_seed: int = 7357,
        ):
        """  
        We use 3 features for each gene, one-hot encodings of genetic mutation, copy
        number amplification, and copy number deletion.
        data vector, x, is in the shape [patient, gene, feature]
        """

        self.num_features=num_features
        self.root=root
        self._files={}
        all_data,response=data_reader(filename_dict=self.raw_file_names,graph=False)
        self.subject_id=list(response.index)
        self.x=torch.tensor(all_data.to_numpy(),dtype=torch.float32)
        self.x=self.x.view(len(self.x),-1,self.num_features)
        self.y=torch.tensor(response.to_numpy(),dtype=torch.float32)

        self.genes=[g[0] for g in list(all_data.head(0))[0::self.num_features]]
        
        self.num_samples = len(self.y)
        self.num_test_samples = int(test_ratio * self.num_samples)
        self.num_valid_samples = int(valid_ratio * self.num_samples)
        self.num_train_samples = (
            self.num_samples - self.num_test_samples - self.num_valid_samples
        )
        self.split_index_by_rng(test_seed=test_seed, valid_seed=valid_seed)
        
    def split_index_by_rng(self, test_seed, valid_seed):
        """ Generate random splits for train, valid, test """
        # train/valid/test random generators
        rng_test = np.random.default_rng(test_seed)
        rng_valid = np.random.default_rng(valid_seed)

        # splitting off the test indices
        test_split_perm = rng_test.permutation(self.num_samples)
        self.test_idx = list(test_split_perm[: self.num_test_samples])
        self.trainvalid_indices = test_split_perm[self.num_test_samples :]

        # splitting off the validation from the remainder
        valid_split_perm = rng_valid.permutation(len(self.trainvalid_indices))
        self.valid_idx = list(
            self.trainvalid_indices[valid_split_perm[: self.num_valid_samples]]
        )
        self.train_idx = list(
            self.trainvalid_indices[valid_split_perm[self.num_valid_samples :]]
        )

    def split_index_by_file(self, train_fp, valid_fp, test_fp):
        """ Load train, valid, test splits from file """
        train_set = pd.read_csv(train_fp, index_col=0)
        valid_set = pd.read_csv(valid_fp, index_col=0)
        test_set = pd.read_csv(test_fp, index_col=0)
        
        patients_train=list(train_set.loc[:,"id"])
        both = set(self.subject_id).intersection(patients_train)
        self.train_idx=[self.subject_id.index(x) for x in both]
        
        patients_valid=list(valid_set.loc[:,"id"])
        both = set(self.subject_id).intersection(patients_valid)
        self.valid_idx=[self.subject_id.index(x) for x in both]
        
        patients_test=list(test_set.loc[:,"id"])
        both = set(self.subject_id).intersection(patients_test)
        self.test_idx=[self.subject_id.index(x) for x in both]
        
        # check no redundency
        assert len(self.train_idx) == len(set(self.train_idx))
        assert len(self.valid_idx) == len(set(self.valid_idx))
        assert len(self.test_idx) == len(set(self.test_idx))
        # check no overlap
        assert len(set(self.train_idx).intersection(set(self.valid_idx))) == 0
        assert len(set(self.train_idx).intersection(set(self.test_idx))) == 0
        assert len(set(self.valid_idx).intersection(set(self.test_idx))) == 0
        
    def __repr__(self):
        return (
            f"PnetDataset("
            f"len={len(self)}, "
            f")"
        )

    @property
    def raw_file_names(self):
        return {
            "selected_genes": os.path.join(
                self.root,
                self._files.get(
                    "selected_genes",
                    "tcga_prostate_expressed_genes_and_cancer_genes.csv",
                ),
            ),
            "use_coding_genes_only": os.path.join(
                self.root,
                self._files.get(
                    "use_coding_genes_only",
                    "protein-coding_gene_with_coordinate_minimal.txt",
                ),
            ),
            # tumor data
            "response": os.path.join(
                self.root, self._files.get("response", "response_paper.csv")
            ),
            "mut_important": os.path.join(
                self.root,
                self._files.get(
                    "mut_important", "P1000_final_analysis_set_cross_important_only.csv"
                ),
            ),
            "cnv_amp": os.path.join(
                self.root, self._files.get("cnv_amp", "P1000_data_CNA_paper.csv")
            ),
            "cnv_del": os.path.join(
                self.root, self._files.get("cnv_del", "P1000_data_CNA_paper.csv")
            ),
        }

    @property
    def processed_file_names(self):
        return f"data-{self.name}-{self.edge_tol:.2f}.pt"

    @property
    def processed_dir(self) -> str:
        return self.root
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x[idx],self.y[idx]
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
dataset = PnetDataSet(
    root=prostate_root,
    name="prostate_graph_humanbase",
    edge_tol=0.5, ## Gene connectivity threshold to form an edge connection
    pre_transform=T.Compose(
        [T.GCNNorm(add_self_loops=False), T.ToSparseTensor(remove_edge_index=False)]
    ),
)

# loads the train/valid/test split from pnet
splits_root = os.path.join(prostate_root, "splits")
dataset.split_index_by_file(
    train_fp=os.path.join(splits_root, "training_set_0.csv"),
    valid_fp=os.path.join(splits_root, "validation_set.csv"),
    test_fp=os.path.join(splits_root, "test_set.csv"),
)