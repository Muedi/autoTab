# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.transforms import Compose, GCNNorm, ToSparseTensor
from sklearn.metrics import accuracy_score, roc_auc_score

# Assuming the necessary imports and dataset initialization are done as provided

# Define the fully connected neural network
class FullyConnectedNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FullyConnectedNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# Flatten the input data
def flatten_data(x):
    return x.view(x.size(0), -1)

# Training loop
def train(model, train_loader, criterion, optimizer, n_epochs):
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            x, y = batch
            x = flatten_data(x).float()
            y = y.float()
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Evaluation loop
def evaluate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x = flatten_data(x).float()
            y = y.float()
            outputs = model(x)
            loss = criterion(outputs, y)
            val_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    accuracy = accuracy_score(all_labels, all_preds > 0.5)
    auc = roc_auc_score(all_labels, all_preds)

    print(f"Validation Loss: {val_loss/len(val_loader):.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation AUC: {auc:.4f}")

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

model = FullyConnectedNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train the model
train(model, train_loader, criterion, optimizer, n_epochs)

# Evaluate the model
evaluate(model, valid_loader, criterion)
