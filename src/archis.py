# %%
# Utility to make a sparse weight list for a Network.
# https://github.com/zhanglab-aim/cancer-net/blob/main/cancernet/util/tensor.py

import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Tuple, List

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
class BaseNet(pl.LightningModule):
    """A basic scaffold for our modules, with default optimizer, scheduler, and loss
    function, and simple logging.
    """

    def __init__(self, lr: float = 0.01, scheduler: str="lambda"):
        super().__init__()
        self.lr = lr
        self.scheduler = scheduler
        # init lists to store outputs
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def configure_optimizers(self) -> Tuple[List, List]:
        """Set up optimizers and schedulers.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if self.scheduler == "lambda":
            lr_lambda = lambda epoch: 1.0 if epoch < 30 else 0.5 if epoch < 60 else 0.1
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        elif self.scheduler == "pnet":  # Take scheduler from pnet
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.25)
        else:
            scheduler = None

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
        output = self.step(batch, "train")
        self.training_step_outputs.append(output)
        return output

    def validation_step(self, batch, batch_idx) -> dict:
        output = self.step(batch, "val")
        self.validation_step_outputs.append(output)
        return output

    def test_step(self, batch, batch_idx) -> dict:
        output = self.step(batch, "test")
        self.test_step_outputs.append(output)
        return output

    def on_train_epoch_end(self):
        self.epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()  # free memory

    def on_validation_epoch_end(self):
        self.epoch_end(self.validation_step_outputs, "val")
        self.validation_step_outputs.clear()  # free memory

    def on_test_epoch_end(self):
        self.epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()  # free memory


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
        # print(y_hat)
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



class PNet_flatten(BaseNet):
    """Implementation of the pnet without the feature layer, but the input is flattened.
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
        self.class_weights = class_weights
        self.layers = layers
        self.num_genes = num_genes
        self.num_features = num_features
        self.intermediate_outputs = intermediate_outputs
        self.network = nn.ModuleList()
        self.intermediate_outs = nn.ModuleList()

        # Flatten the input
        self.flatten = nn.Flatten()

        # First layer with flattened input
        self.network.append(
            nn.Sequential(
                nn.Linear(self.num_genes * self.num_features, self.num_genes),
                nn.Tanh()
            )
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
        x = self.flatten(x)
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
        # print(y_hat)
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


# def count_nonzero_weights(model):
#     nonzero_weights = 0
#     for param in model.parameters():
#         nonzero_weights += (param != 0).sum().item()
#     return nonzero_weights

class FullyConnectedNet(BaseNet):
    """A fully connected neural network with 6 layers, including the FeatureLayer."""

    def __init__(
        self,
        num_genes: int,
        num_features: int = 3,
        hidden_size: int = 128,
        lr: float = 0.001,
        l1_lambda: float = 0.0,
        scheduler: str="lambda"
    ):
        """Initialize.
        :param num_genes: number of genes in dataset
        :param num_features: number of features for each gene
        :param lr: learning rate
        :param l1_lambda: L1 regularization strength
        """
        super().__init__(lr=lr, scheduler=scheduler)
        self.num_genes = num_genes
        self.num_features = num_features
        self.l1_lambda = l1_lambda

        self.network = nn.Sequential(
            FeatureLayer(self.num_genes, self.num_features),
            nn.Tanh(),
            nn.Linear(self.num_genes, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """ Forward pass through the fully connected network. """
        return self.network(x)

    def step(self, batch, kind: str) -> dict:
        """Step function executed by lightning trainer module."""
        # run the model and calculate loss
        x, y_true = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y_true)

        # Add L1 regularization
        if self.l1_lambda > 0:
            l1_norm = sum(p.abs().sum() for p in self.parameters())
            loss += self.l1_lambda * l1_norm

        # assess accuracy
        correct = ((y_hat > 0.5).flatten() == y_true.flatten()).sum()
        total = len(y_true)
        batch_dict = {
            "loss": loss,
            # correct and total will be used at epoch end
            "correct": correct,
            "total": total,
        }
        return batch_dict
    
    def on_epoch_end(self):
        nonzero_weights = count_nonzero_weights(self)
        self.log('nonzero_weights', nonzero_weights)
        print(f'Epoch {self.current_epoch}: Non-zero weights = {nonzero_weights}')

class FullyConnectedNet_flatten(BaseNet):
    def __init__(self,
                input_size: int,
                hidden_size: int,
                output_size: int = 1,
                lr: int = 0.001,
                l1_lambda: int = 0.0,
                scheduler: str = "lambda"
    ):
        super().__init__(lr=lr, scheduler=scheduler)
        self.flatten = nn.Flatten()
        self.l1_lambda = l1_lambda
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

    def step(self, batch, kind: str) -> dict:
        """Step function executed by lightning trainer module."""
        # run the model and calculate loss
        x, y_true = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y_true)

        # Add L1 regularization
        if self.l1_lambda > 0:
            l1_norm = sum(p.abs().sum() for p in self.parameters())
            loss += self.l1_lambda * l1_norm

        # assess accuracy
        correct = ((y_hat > 0.5).flatten() == y_true.flatten()).sum()
        total = len(y_true)
        batch_dict = {
            "loss": loss,
            # correct and total will be used at epoch end
            "correct": correct,
            "total": total,
        }
        return batch_dict

    def on_epoch_end(self):
        nonzero_weights = count_nonzero_weights(self)
        self.log('nonzero_weights', nonzero_weights)
        print(f'Epoch {self.current_epoch}: Non-zero weights = {nonzero_weights}')
