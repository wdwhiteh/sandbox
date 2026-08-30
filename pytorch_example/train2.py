"""
LESSON 2 of 7: checking a "practice test" every epoch (validation loss/accuracy).

What's new vs train1.py: train1.py only checked the held-out data once, at
the very end, after training was already finished. That's too late to be
useful -- if something had gone wrong partway through, we'd have no way to
notice. This lesson checks the held-out data after EVERY epoch instead, so
we can watch training and "generalizing to new data" side by side as they
happen. Everything else (batch size, learning rate, epoch count, no
dropout, no early stopping) is unchanged from train1.py.

Why this matters: a model's score on the data it was trained on
("train loss"/"train acc") only tells you how well it memorized what it saw.
Its score on data it was NOT trained on ("val loss"/"val acc", short for
"validation") is what actually tells you whether it learned something
general and useful, or just memorized. Watching both together, epoch by
epoch, is how you catch it if the two start disagreeing.

Run it with: python train2.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

MODEL_PATH = "model2.pt"


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
    # model.eval() turns off training-only behavior (relevant again once
    # train4.py adds dropout); for now it doesn't change anything here.
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

    # Unchanged from train1.py -- this lesson is only about *watching*
    # training more closely, not about changing how it trains.
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

    # ------------------------------------------------------------------
    # NEW IN THIS LESSON: run evaluate() on the held-out val_loader after
    # every single epoch (not just once at the end), and print train vs.
    # val numbers on the same line so they're easy to compare as we go.
    # ------------------------------------------------------------------
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, device, val_loader, criterion)
        print(
            f"Epoch {epoch:2d} | train loss: {train_loss:.4f} acc: {train_acc:.2%} "
            f"| val loss: {val_loss:.4f} acc: {val_acc:.2%}"
        )

    # We're still just saving whatever the final epoch produced -- we're not
    # yet doing anything with the val numbers besides printing them. Using
    # them to decide what to save is the subject of train3.py.
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
