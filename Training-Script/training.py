import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- 0. Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Define Human Skeleton Graph ---
num_nodes = 17
self_link = [(i, i) for i in range(num_nodes)]
neighbor_link = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 11), (6, 12)
]
edge = self_link + neighbor_link
A = np.zeros((num_nodes, num_nodes))
for i, j in edge:
    A[i, j] = 1
    A[j, i] = 1
A = torch.from_numpy(A).float().to(device)

# --- 2. Load Dataset ---
print("Loading processed 2-class dataset...")

base_path = "../Processed-Dataset-Numpy"
X_data = np.load(os.path.join(base_path, "X_data_2class.npy"))
y_data = np.load(os.path.join(base_path, "y_labels_2class.npy"))

# Reshape (N, T, V, C) -> (N, C, T, V)
X_data = np.transpose(X_data, (0, 3, 1, 2))

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(
    X_data, y_data, test_size=0.25, random_state=42, stratify=y_data
)

print("\n--- Dataset Shapes ---")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")

# --- 3. Define Model ---
class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, A):
        x = torch.einsum("nctv,vw->nctw", x, A)
        return self.conv(x)

class STGCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout):
        super().__init__()
        self.gcn = GraphConv(in_channels, out_channels)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (kernel_size, 1),
                      (stride, 1), ((kernel_size - 1) // 2, 0)),
            nn.BatchNorm2d(out_channels), nn.Dropout(dropout, inplace=True)
        )

    def forward(self, x, A):
        x = self.gcn(x, A)
        return self.tcn(x)

class SGT_Model(nn.Module):
    def __init__(self, num_classes, in_channels, num_nodes):
        super().__init__()
        self.stgcn1 = STGCN_Block(in_channels, 64, kernel_size=9, stride=1, dropout=0.3)
        self.stgcn2 = STGCN_Block(64, 128, kernel_size=9, stride=2, dropout=0.3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, A):
        x = self.stgcn1(x, A)
        x = self.stgcn2(x, A)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# --- 4. Train / Eval Functions ---
def train_epoch(model, loader, optimizer, criterion, A):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(data, A)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = output.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion, A):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            output = model(data, A)
            loss = criterion(output, labels)

            total_loss += loss.item()
            preds = output.argmax(dim=1)

            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), correct / total, np.array(all_preds), np.array(all_labels)

# --- 5. Train on different batch sizes ---
batch_sizes = [4, 8, 16, 32]
epochs = 50
criterion = nn.CrossEntropyLoss()

model_dir = "../models"
os.makedirs(model_dir, exist_ok=True)

for batch_size in batch_sizes:
    print(f"\n====================================")
    print(f"Training with Batch Size: {batch_size}")
    print(f"====================================")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(),
                      torch.from_numpy(y_train).long()),
        batch_size=batch_size, shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float(),
                      torch.from_numpy(y_val).long()),
        batch_size=batch_size
    )

    model = SGT_Model(num_classes=2, in_channels=2, num_nodes=17).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_acc = 0
    model_path = os.path.join(model_dir, f"best_sgt_model_2class_bs{batch_size}.pth")

    train_losses, val_losses = [], []

    print(f"Iterations per Epoch: {len(train_loader)}")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, A)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, A)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)

        print(f"Epoch [{epoch:02d}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    print(f"\nFinished Batch Size {batch_size}")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Saved model → {model_path}\n")

    model.load_state_dict(torch.load(model_path))
    _, _, preds, labels = evaluate(model, val_loader, criterion, A)

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Shoplifting"],
                yticklabels=["Normal", "Shoplifting"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - Batch Size {batch_size}")
    plt.show()

    # Classification Report
    print(f"\nClassification Report (Batch {batch_size})")
    print(classification_report(labels, preds, target_names=["Normal", "Shoplifting"], digits=4))

    # Loss Curve
    plt.figure(figsize=(7, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve - Batch Size {batch_size}")
    plt.legend()
    plt.grid(True)
    plt.show()
