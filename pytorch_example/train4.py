"""
LESSON 4 of 7: dropout layers.

What's new vs train3.py: two `nn.Dropout` layers added to the model itself,
right after each hidden layer's activation. Everything about *how* we train
(batch size, learning rate, epoch cap, patience) is unchanged from train3.py.

What dropout actually does: during training only, on every batch, it
randomly "turns off" a fraction of the neurons in that layer (here, 10% of
them, picked freshly each time) by forcing their output to zero. The rest of
the network has to keep working despite not being able to rely on any one
neuron always being there. This stops the network from building an
over-specific plan like "if neuron #37 fires a certain way, always predict
category 2" -- a plan like that would fall apart the instant #37 is dropped,
so the network is pushed toward more redundant, general-purpose patterns
instead. At evaluation and inference time, dropout does nothing (every
neuron is used); the model's `.eval()` call is what switches this off, and
`.train()` switches it back on -- which is exactly why every training
function calls `model.train()` and every evaluation function calls
`model.eval()`.

Why it's introduced right after train3.py: dropout is one of the standard
tools for fighting the exact overfitting pattern train3.py demonstrated
(train loss dropping while val loss creeps back up). It won't eliminate
that pattern here -- the dataset's labels have random noise baked in, so
some overfitting is unavoidable -- but it should narrow the gap between
train and val performance compared to train3.py.

Run it with: python train4.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

MODEL_PATH = "model4.pt"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print(f"Found {torch.cuda.device_count()} CUDA device(s), running torch on GPU.")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Found Apple MPS device, running torch on GPU.")
        return torch.device("mps")
    print("No GPU found, running torch on CPU.")
    return torch.device("cpu")


class SimpleClassifier(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),  # NEW IN THIS LESSON
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),  # NEW IN THIS LESSON
            nn.Linear(hidden_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_synthetic_dataset(num_samples: int, in_features: int, num_classes: int, seed: int = 0):
    truth_generator = torch.Generator().manual_seed(0)
    true_weights = torch.randn(in_features, num_classes, generator=truth_generator)

    sample_generator = torch.Generator().manual_seed(seed)
    X = torch.randn(num_samples, in_features, generator=sample_generator)
    logits = X @ true_weights + 0.5 * torch.randn(num_samples, num_classes, generator=sample_generator)
    y = logits.argmax(dim=1)
    return X, y


def train_one_epoch(model, device, train_loader, optimizer, criterion):
    model.train()  # enables dropout for this epoch's updates
    total_loss = 0.0
    correct = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        correct += (outputs.argmax(dim=1) == y_batch).sum().item()

    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = correct / len(train_loader.dataset)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, device, data_loader, criterion):
    model.eval()  # disables dropout, so scoring uses the whole network
    total_loss = 0.0
    correct = 0
    for X_batch, y_batch in data_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        total_loss += loss.item() * X_batch.size(0)
        correct += (outputs.argmax(dim=1) == y_batch).sum().item()

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)
    return avg_loss, accuracy


@torch.no_grad()
def demo_inference(model, device, in_features, num_classes):
    model.eval()
    X, y = make_synthetic_dataset(num_samples=20, in_features=in_features, num_classes=num_classes, seed=1)
    X = X.to(device)
    predictions = model(X).argmax(dim=1)

    print("\nInference on 20 unseen examples:")
    hits = 0
    for i, (pred, actual) in enumerate(zip(predictions.tolist(), y.tolist())):
        ok = pred == actual
        hits += ok
        print(f"  sample {i}: predicted={pred} actual={actual} [{'OK' if ok else 'MISS'}]")
    print(f"  {hits}/20 correct (accuracy: {hits / 20:.1%})")


def main():
    device = get_device()
    print(f"Using device: {device}")

    in_features = 20
    num_classes = 4
    hidden_features = 64
    dropout = 0.1  # NEW IN THIS LESSON: fraction of neurons randomly zeroed per batch during training

    batch_size = 64
    learning_rate = 1e-3
    num_epochs = 200
    patience = 15

    X, y = make_synthetic_dataset(num_samples=5000, in_features=in_features, num_classes=num_classes)
    split = int(0.8 * len(X))
    train_ds = TensorDataset(X[:split], y[:split])
    val_ds = TensorDataset(X[split:], y[split:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = SimpleClassifier(in_features, hidden_features, num_classes, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, device, val_loader, criterion)
        print(
            f"Epoch {epoch:3d} | train loss: {train_loss:.4f} acc: {train_acc:.2%} "
            f"| val loss: {val_loss:.4f} acc: {val_acc:.2%}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"No val loss improvement for {patience} epochs, stopping early "
                    f"(best was epoch {best_epoch}, val loss {best_val_loss:.4f})."
                )
                break

    torch.save(
        {
            "model_state": best_state,
            "in_features": in_features,
            "hidden_features": hidden_features,
            "num_classes": num_classes,
        },
        MODEL_PATH,
    )
    print(f"Saved model from epoch {best_epoch} (val loss {best_val_loss:.4f}) to {MODEL_PATH}")

    model.load_state_dict(best_state)
    demo_inference(model, device, in_features, num_classes)


if __name__ == "__main__":
    main()
