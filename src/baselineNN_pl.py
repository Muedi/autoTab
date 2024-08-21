# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.transforms import Compose, GCNNorm, ToSparseTensor
from sklearn.metrics import accuracy_score, roc_auc_score

# Assuming the necessary imports and dataset initialization are done as provided

# Define the fully connected neural network
class FullyConnectedNet_flatten(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        super(FullyConnectedNet_flatten, self).__init__()
        self.lr = lr
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
        self.val_preds = []
        self.val_labels = []

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = flatten_data(x).float()
        y = y.float()
        outputs = self(x)
        loss = nn.BCELoss()(outputs, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = flatten_data(x).float()
        y = y.float()
        outputs = self(x)
        loss = nn.BCELoss()(outputs, y)
        self.log('val_loss', loss)
        self.val_preds.append(outputs.cpu().numpy())
        self.val_labels.append(y.cpu().numpy())
        return {'val_loss': loss, 'preds': outputs, 'labels': y}

    def on_validation_epoch_end(self):
        all_preds = np.concatenate(self.val_preds)
        all_labels = np.concatenate(self.val_labels)
        accuracy = accuracy_score(all_labels, all_preds > 0.5)
        auc = roc_auc_score(all_labels, all_preds)

        self.log('val_accuracy', accuracy)
        self.log('val_auc', auc)

        # Clear the lists for the next epoch
        self.val_preds.clear()
        self.val_labels.clear()

# Flatten the input data
def flatten_data(x):
    return x.view(x.size(0), -1)

# Initialize the dataset and data loaders
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

# Initialize the model
input_size = dataset[0][0].numel()  # Flatten the input
hidden_size = 128
output_size = 1

model = FullyConnectedNet_flatten(input_size=input_size, hidden_size=hidden_size, output_size=output_size, lr=lr)

# Initialize the trainer
trainer = pl.Trainer(max_epochs=n_epochs)

# Train the model
trainer.fit(model, train_loader, valid_loader)

# Evaluate the model
trainer.validate(model, valid_loader)
