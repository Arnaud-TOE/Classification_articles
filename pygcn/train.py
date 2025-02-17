from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import sys
sys.path.append("c:/Users/PC MAROC/dev/pygcn-master")

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.') #200
parser.add_argument('--lr', type=float, default=0.02,
                    help='Initial learning rate.') #0.01
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).') #5e-4
parser.add_argument('--hidden', type=int, default=25,
                    help='Number of hidden units.') #16
parser.add_argument('--dropout', type=float, default=0.4,
                    help='Dropout rate (1 - keep probability).') #0.5

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Function to create a new model and optimizer for each iteration
def create_model_and_optimizer():
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    if args.cuda:
        model.cuda()
    return model, optimizer


# Function to train the model
def train_model(model, optimizer):
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
    


# Function to test the model
def test_model(model):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
    return loss_test.item(), acc_test.item()


# Run training and testing 100 times
losses = []
accuracies = []

model, optimizer = create_model_and_optimizer()

# Train the model
train_model(model, optimizer)

for i in range(100):
    print(f"Run {i + 1}/100")
   

    # Test the model
    loss, acc = test_model(model)

    # Collect results
    losses.append(loss)
    accuracies.append(acc)

# Calculate averages
avg_loss = np.mean(losses)
avg_acc = np.mean(accuracies)

print("\n===== Results =====")
print(f"Average Test Loss: {avg_loss:.4f}")
print(f"Average Test Accuracy: {avg_acc:.4f}")