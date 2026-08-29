"""
Simple PyTorch example: builds and trains a small neural network on
synthetic data, automatically using GPU acceleration (CUDA or Apple MPS)
when available and falling back to CPU otherwise. The trained weights are
saved to model.pt so infer.py can load them and run inference.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

MODEL_PATH = "model.pt"


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
    """Generates a linearly separable-ish classification dataset.

    The underlying "true" decision boundary (true_weights) always comes from
    a fixed seed so it stays consistent between train.py and infer.py; only
    the sampled inputs vary with `seed`, so infer.py can draw fresh, unseen
    examples that are still labeled by the same ground-truth function the
    model was trained on.
    """
    truth_generator = torch.Generator().manual_seed(0)
    true_weights = torch.randn(in_features, num_classes, generator=truth_generator)

    sample_generator = torch.Generator().manual_seed(seed)
    X = torch.randn(num_samples, in_features, generator=sample_generator)
    logits = X @ true_weights + 0.5 * torch.randn(num_samples, num_classes, generator=sample_generator)
    y = logits.argmax(dim=1)
    return X, y


def train(model, device, train_loader, optimizer, criterion, epoch):
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
    print(f"Epoch {epoch:2d} | loss: {avg_loss:.4f} | accuracy: {accuracy:.2%}")


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
    print(f"Eval        | loss: {avg_loss:.4f} | accuracy: {accuracy:.2%}")


def main():
    device = get_device()
    print(f"Using device: {device}")

    in_features = 20
    num_classes = 4
    hidden_features = 64
    num_epochs = 10
    batch_size = 64
    learning_rate = 1e-2

    X, y = make_synthetic_dataset(num_samples=5000, in_features=in_features, num_classes=num_classes)
    split = int(0.8 * len(X))
    train_ds = TensorDataset(X[:split], y[:split])
    test_ds = TensorDataset(X[split:], y[split:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = SimpleClassifier(in_features, hidden_features, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)

    evaluate(model, device, test_loader, criterion)

    torch.save(
        {
            "model_state": model.state_dict(),
            "in_features": in_features,
            "hidden_features": hidden_features,
            "num_classes": num_classes,
        },
        MODEL_PATH,
    )
    print(f"Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
