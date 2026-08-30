"""
LESSON 6 of 7 (bonus): input noise injection, for robustness to noisy inputs.

This lesson isn't from the original list of four concepts -- it's an extra
one, because it addresses a different question than the others. Lessons
2-5 were all about getting the best possible score on THIS dataset. This
one is about a different, practical concern: what happens when the model is
later used on real-world inputs that are a little noisy or imprecise
(sensor jitter, rounding, measurement error)? A model that has only ever
seen "clean" training data can be more sensitive to that than you'd like.

What's new vs train5.py: inside the training loop only, a small amount of
random Gaussian noise is added to each batch of inputs before it's shown to
the model. Evaluation and inference are untouched -- they still use the
data exactly as generated, with no added noise. Everything else (dropout,
tuned batch size and learning rate, patience) is unchanged from train5.py.

Why adding noise during training helps: if the model is repeatedly shown
slightly-different versions of what is "supposed to be" the same input, it
can no longer succeed by keying off exact input values -- it's forced to
rely on the broader pattern that stays true even when the numbers wiggle a
little. That's the same underlying idea as dropout (train4.py) -- prevent
over-reliance on any one precise detail -- applied to the input data instead
of to the network's internal neurons.

Run it with: python train6.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

MODEL_PATH = "model6.pt"


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
            nn.Dropout(dropout),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
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


def train_one_epoch(model, device, train_loader, optimizer, criterion, noise_std: float = 0.0):
    model.train()
    total_loss = 0.0
    correct = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        # NEW IN THIS LESSON: jitter the inputs with a little random noise,
        # only here in the training loop -- never during evaluate() or
        # demo_inference() below.
        if noise_std > 0:
            X_batch = X_batch + noise_std * torch.randn_like(X_batch)

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
    dropout = 0.1

    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 300
    patience = 15
    noise_std = 0.1  # NEW IN THIS LESSON: standard deviation of the noise added to training inputs

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
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion, noise_std=noise_std)
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
