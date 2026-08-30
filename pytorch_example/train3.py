"""
LESSON 3 of 7: training for longer, and teaching it to stop itself (patience / early stopping).

What's new vs train2.py: two changes, because they're two sides of the same
problem.

1. We raise num_epochs way up (10 -> 200), to see what happens if we just
   let training run longer, hoping "more training = a better model."
2. We add "patience": after every epoch, if val loss is the best we've seen
   so far, we save a copy of the model's weights. If 15 epochs go by with no
   new best, we stop training early -- and the file we save at the end is
   the BEST-seen epoch's weights, not whatever the final epoch happened to
   produce.

Why: if you look closely at train2.py's own output, val loss actually
bottoms out around epoch 6 and then starts drifting back up slightly, even
though train loss keeps falling every single epoch. That gap is called
OVERFITTING: past a certain point, the model isn't learning the real pattern
anymore -- it's starting to memorize quirks specific to the training
examples (including the random noise baked into their labels), which makes
it worse, not better, at everything else. Change #1 alone (more epochs, no
early stopping) would make that problem much worse over 200 epochs instead
of 10. Change #2 (patience) is the fix: it lets training run as long as it's
actually improving, then stops and keeps the best version -- so you get the
benefit of "train for a while" without the downside of "then keep going long
after it stopped helping."

Run it with: python train3.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

MODEL_PATH = "model3.pt"


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

    batch_size = 64
    learning_rate = 1e-3
    # ------------------------------------------------------------------
    # NEW IN THIS LESSON, part 1: a much higher epoch ceiling. This is not
    # a promise we'll actually train this long -- it's an upper bound, in
    # case the model is still improving. Patience (below) decides when to
    # actually stop.
    # ------------------------------------------------------------------
    num_epochs = 200
    # NEW IN THIS LESSON, part 2: stop once val loss hasn't set a new best
    # for this many epochs in a row.
    patience = 15

    X, y = make_synthetic_dataset(num_samples=5000, in_features=in_features, num_classes=num_classes)
    split = int(0.8 * len(X))
    train_ds = TensorDataset(X[:split], y[:split])
    val_ds = TensorDataset(X[split:], y[split:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = SimpleClassifier(in_features, hidden_features, num_classes).to(device)
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

        # ------------------------------------------------------------------
        # NEW IN THIS LESSON: remember the best-so-far weights, and count how
        # many epochs it's been since we last improved. Once that count hits
        # `patience`, stop -- further training is more likely to hurt than help.
        # ------------------------------------------------------------------
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

    # Save the BEST epoch's weights, not the final epoch's -- that's the
    # whole point of tracking best_state above.
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
