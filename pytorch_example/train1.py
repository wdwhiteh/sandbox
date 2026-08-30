"""
LESSON 1 of 6: the plain, "just get it working" baseline.

This is the simplest version of the training script: pick some settings,
train for a fixed number of passes over the data, and check the result once
at the end. It has no safety nets — nothing here watches for the model
"cheating" by memorizing the training data instead of actually learning.
The next five lessons (train2.py -> train6.py) each add ONE new idea on top
of this file to fix that, and pytorch_example/README.md explains what
changes and why after every step.

Hyperparameters used in this lesson (deliberately naive, chosen before
knowing how the model behaves):
    batch_size    = 64     (how many examples the model looks at per update)
    learning_rate = 1e-3   (how big a step it takes when correcting itself)
    num_epochs    = 10     (how many full passes it makes over the training data)

Run it with: python train1.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

MODEL_PATH = "model1.pt"


def get_device() -> torch.device:
    """Picks the fastest hardware available: an NVIDIA GPU, an Apple GPU, or plain CPU."""
    if torch.cuda.is_available():
        print(f"Found {torch.cuda.device_count()} CUDA device(s), running torch on GPU.")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Found Apple MPS device, running torch on GPU.")
        return torch.device("mps")
    print("No GPU found, running torch on CPU.")
    return torch.device("cpu")


class SimpleClassifier(nn.Module):
    """A small neural network: 20 numbers in, a category guess out.

    See pytorch_example/README.md for an illustrated explanation of what
    each layer does. There's no dropout layer in this lesson — that's
    introduced in train4.py once we can see why it helps.
    """

    def __init__(self, in_features: int, hidden_features: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_synthetic_dataset(num_samples: int, in_features: int, num_classes: int, seed: int = 0):
    """Generates a made-up classification dataset so this example needs no downloads.

    The underlying "true" decision boundary (true_weights) always comes from
    a fixed seed so every lesson's model is judged on the same ground truth;
    only the sampled inputs vary with `seed`, so we can draw a fresh batch of
    "unseen" examples later for the inference demo at the bottom of this file.
    """
    truth_generator = torch.Generator().manual_seed(0)
    true_weights = torch.randn(in_features, num_classes, generator=truth_generator)

    sample_generator = torch.Generator().manual_seed(seed)
    X = torch.randn(num_samples, in_features, generator=sample_generator)
    logits = X @ true_weights + 0.5 * torch.randn(num_samples, num_classes, generator=sample_generator)
    y = logits.argmax(dim=1)
    return X, y


def train_one_epoch(model, device, train_loader, optimizer, criterion):
    model.train()
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
    model.eval()
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
    """Tries the trained model on 20 brand-new examples it never trained on."""
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

    # --- The three "knobs" this lesson uses, picked without any tuning yet ---
    batch_size = 64
    learning_rate = 1e-3
    num_epochs = 10

    X, y = make_synthetic_dataset(num_samples=5000, in_features=in_features, num_classes=num_classes)
    split = int(0.8 * len(X))
    train_ds = TensorDataset(X[:split], y[:split])
    val_ds = TensorDataset(X[split:], y[split:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = SimpleClassifier(in_features, hidden_features, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # NOTE: this lesson only checks progress using the training data itself,
    # every epoch. We're not yet looking at the held-out val_ds at all during
    # training -- that's the entire subject of the next lesson, train2.py.
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)
        print(f"Epoch {epoch:2d} | train loss: {train_loss:.4f} acc: {train_acc:.2%}")

    # We only look at the held-out data once, right at the very end, purely
    # out of curiosity -- it plays no role in training or in which weights
    # get saved. Whatever the last epoch produced is what gets saved below.
    final_val_loss, final_val_acc = evaluate(model, device, val_loader, criterion)
    print(f"\nFinal check on held-out data | loss: {final_val_loss:.4f} acc: {final_val_acc:.2%}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "in_features": in_features,
            "hidden_features": hidden_features,
            "num_classes": num_classes,
        },
        MODEL_PATH,
    )
    print(f"Saved model (from the final epoch) to {MODEL_PATH}")

    demo_inference(model, device, in_features, num_classes)


if __name__ == "__main__":
    main()
