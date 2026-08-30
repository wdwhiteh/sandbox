"""
LESSON 7 of 7 (bonus): CPU vs. GPU -- does the GPU actually help here?

Every earlier lesson used get_device() to automatically pick the best
available hardware and just... trusted that a GPU is faster. This lesson
checks that assumption instead of taking it on faith, in keeping with
lesson 6's theme: measure, don't assume. It trains the exact same model,
on the exact same data, with the exact same starting weights, once on the
CPU and once on the GPU (if one is available) -- and times both, epoch by
epoch, so you can see the real difference rather than guess at it.

It does this TWICE, with only one thing changed between the two rounds:
the batch size (how many examples are shown to the model per weight
update).

  - Round 1 uses a SMALL batch size (32, same as every earlier lesson).
    Expect the CPU to win here. Sending a batch to the GPU and launching
    work on it has a small, fixed cost every single time, no matter how
    much or how little work is in that batch. With only 32 tiny examples
    per batch, that fixed cost is bigger than the actual computation --
    the GPU spends most of its time waiting on overhead, not calculating.

  - Round 2 uses a LARGE batch size (2048). Now each batch carries 64x more
    work, but the GPU still only pays that same fixed per-batch cost. The
    GPU's ability to crunch through thousands of examples at once (instead
    of one at a time, the way a CPU effectively does) finally has enough
    work to show off, and should come out ahead.

This is the actual mechanism behind "GPUs are faster for AI" -- it was
never automatic, it was always about there being enough parallel work per
batch to be worth the overhead of using the GPU at all. Small batch, tiny
model: CPU can win. Large batch: the GPU's advantage shows up.

One catch, straight out of lesson 5: batch size and learning rate are
coupled. A 64x bigger batch means 64x fewer weight-update steps per epoch
(2 vs. 125, at a fixed number of epochs), and each step's gradient is
averaged over far more examples, so it's a much steadier, lower-noise
estimate of "which way to move." Leaving the learning rate exactly as it
was for the small batch would badly undertrain round 2 -- it would take
far too few, far-too-cautious steps to get anywhere. This is a well-known
rule of thumb (the "linear scaling rule"): when you scale the batch size
up by some factor, scale the learning rate up by roughly that same factor
to compensate. So round 2 also raises `learning_rate`, from 1e-4 to 8e-3 --
found by trying a few values and checking validation accuracy, the same
"measure, don't guess" approach as every other tuning decision in this
series.

To make each round's timing comparison fair, this lesson intentionally does
NOT use early stopping (patience) like lessons 3-6 did -- CPU and GPU always
train for the exact same fixed number of epochs, with the same learning
rate, within a round, so "how long did training take" is a clean,
like-for-like comparison instead of being muddied by one run stopping
earlier than the other. Everything else (model size, dropout, input noise)
matches lesson 6's tuned settings.

Run it with: python train7.py
"""

import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

MODEL_PATH_TEMPLATE = "model7_{device}_batch{batch_size}.pt"


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
def inference_accuracy(model, device, in_features, num_classes):
    """Same 20-example check as the other lessons, but just returns the accuracy number."""
    model.eval()
    X, y = make_synthetic_dataset(num_samples=20, in_features=in_features, num_classes=num_classes, seed=1)
    X, y = X.to(device), y.to(device)
    predictions = model(X).argmax(dim=1)
    return (predictions == y).float().mean().item()


def run_training(device_name: str, in_features, num_classes, hidden_features, dropout,
                  batch_size, learning_rate, num_epochs, noise_std, train_ds, val_ds):
    """Trains one full run on the given device, timing every epoch. Returns a results dict."""
    device = torch.device(device_name)

    # ------------------------------------------------------------------
    # NEW IN THIS LESSON: give both runs the exact same starting point.
    # Re-seeding right before creating the model means CPU and GPU start
    # from identical weights; re-seeding the DataLoader's shuffle order
    # means they see batches in the same sequence too. That way, any
    # difference we measure in the *results* (not the *timing*) is coming
    # from tiny floating-point differences between CPU and GPU math, not
    # from the two runs training on different data.
    # ------------------------------------------------------------------
    torch.manual_seed(0)
    model = SimpleClassifier(in_features, hidden_features, num_classes, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    shuffle_generator = torch.Generator().manual_seed(0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=shuffle_generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # ------------------------------------------------------------------
    # NEW IN THIS LESSON: actually time it. torch.cuda.synchronize() is
    # required before reading the clock on a CUDA run -- GPU work is
    # queued asynchronously, so without it we'd measure "how long it took
    # to *launch* the work" instead of "how long the work actually took."
    # ------------------------------------------------------------------
    epoch_times = []
    history = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    run_start = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion, noise_std=noise_std)
        val_loss, val_acc = evaluate(model, device, val_loader, criterion)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - epoch_start)
        history.append((epoch, train_loss, train_acc, val_loss, val_acc))
        print(
            f"  [{device_name}] Epoch {epoch:3d} | train loss: {train_loss:.4f} acc: {train_acc:.2%} "
            f"| val loss: {val_loss:.4f} acc: {val_acc:.2%} | {epoch_times[-1] * 1000:.1f} ms"
        )

    total_time = time.perf_counter() - run_start
    infer_acc = inference_accuracy(model, device, in_features, num_classes)

    torch.save(
        {
            "model_state": model.state_dict(),
            "in_features": in_features,
            "hidden_features": hidden_features,
            "num_classes": num_classes,
        },
        MODEL_PATH_TEMPLATE.format(device=device_name, batch_size=batch_size),
    )

    final_epoch = history[-1]
    return {
        "device": device_name,
        "batch_size": batch_size,
        "total_time": total_time,
        # Skip epoch 1 for the average -- the first epoch on either device
        # pays a one-time "warm up" cost (CUDA context / kernel compilation
        # on GPU, memory allocator warm-up on CPU) that isn't representative
        # of steady-state speed.
        "avg_epoch_ms": sum(epoch_times[1:]) / len(epoch_times[1:]) * 1000 if len(epoch_times) > 1 else epoch_times[0] * 1000,
        "first_epoch_ms": epoch_times[0] * 1000,
        "final_train_loss": final_epoch[1],
        "final_train_acc": final_epoch[2],
        "final_val_loss": final_epoch[3],
        "final_val_acc": final_epoch[4],
        "inference_acc": infer_acc,
    }


def print_comparison(results, round_label):
    print("\n" + "=" * 70)
    print(f"CPU vs GPU comparison -- {round_label} (batch_size={results[0]['batch_size']})")
    print("=" * 70)
    header = f"{'metric':<28}" + "".join(f"{r['device']:>18}" for r in results)
    print(header)
    print("-" * len(header))

    def row(label, key, fmt):
        cells = "".join(f"{fmt(r[key]):>18}" for r in results)
        print(f"{label:<28}{cells}")

    row("total time (s)", "total_time", lambda v: f"{v:.2f}")
    row("avg ms / epoch (warm)", "avg_epoch_ms", lambda v: f"{v:.1f}")
    row("first epoch (ms)", "first_epoch_ms", lambda v: f"{v:.1f}")
    row("final train loss", "final_train_loss", lambda v: f"{v:.4f}")
    row("final train acc", "final_train_acc", lambda v: f"{v:.2%}")
    row("final val loss", "final_val_loss", lambda v: f"{v:.4f}")
    row("final val acc", "final_val_acc", lambda v: f"{v:.2%}")
    row("inference acc (20)", "inference_acc", lambda v: f"{v:.2%}")

    if len(results) == 2:
        cpu, gpu = results
        if gpu["avg_epoch_ms"] < cpu["avg_epoch_ms"]:
            speedup = cpu["avg_epoch_ms"] / gpu["avg_epoch_ms"]
            print(f"\nGPU was {speedup:.2f}x faster per epoch than CPU (after warm-up).")
        else:
            slowdown = gpu["avg_epoch_ms"] / cpu["avg_epoch_ms"]
            print(
                f"\nCPU was {slowdown:.2f}x faster per epoch than GPU here -- with a batch size "
                f"this small, the overhead of launching GPU work and moving data to and from the "
                f"GPU costs more than the GPU's raw compute speed saves."
            )


def run_round(round_label, batch_size, in_features, num_classes, hidden_features, dropout,
              learning_rate, num_epochs, noise_std, train_ds, val_ds):
    """Runs the CPU-vs-GPU comparison once, at the given batch size, and prints its table."""
    print(f"\n\n########## {round_label}: batch_size={batch_size} ##########")

    # A bigger batch size means fewer batches per epoch, which means fewer
    # weight-update steps for the same number of epochs -- round 2 gets 64x
    # fewer steps than round 1 despite training for the same 30 epochs.
    # That's why round 2 also uses a larger learning_rate (see the module
    # docstring): without compensating for it, round 2 would be badly
    # undertrained and its loss/accuracy numbers would look broken, which
    # would muddy the actual point of this lesson (CPU vs. GPU speed).
    batches_per_epoch = -(-len(train_ds) // batch_size)  # ceiling division
    print(f"({batches_per_epoch} batch(es)/epoch x {num_epochs} epochs = "
          f"{batches_per_epoch * num_epochs} weight updates total this round, "
          f"learning_rate={learning_rate})")

    results = []

    print("\nTraining on CPU...")
    results.append(
        run_training("cpu", in_features, num_classes, hidden_features, dropout,
                      batch_size, learning_rate, num_epochs, noise_std, train_ds, val_ds)
    )

    gpu_device = None
    if torch.cuda.is_available():
        gpu_device = "cuda"
        print(f"\nTraining on GPU ({torch.cuda.get_device_name(0)})...")
    elif torch.backends.mps.is_available():
        gpu_device = "mps"
        print("\nTraining on GPU (Apple MPS)...")

    if gpu_device:
        results.append(
            run_training(gpu_device, in_features, num_classes, hidden_features, dropout,
                         batch_size, learning_rate, num_epochs, noise_std, train_ds, val_ds)
        )
    else:
        print("\nNo GPU found on this machine, so there's nothing to compare CPU against here.")
        print("(Everything above already ran on CPU -- that's what every other lesson falls back to as well.)")

    print_comparison(results, round_label)
    return results


def main():
    in_features = 20
    num_classes = 4
    hidden_features = 64
    dropout = 0.1
    noise_std = 0.1
    # Fixed, not early-stopped -- see the module docstring for why.
    num_epochs = 30

    # NEW IN THIS LESSON: two batch sizes, run one after the other, so you
    # can see the crossover happen instead of just being told about it. The
    # learning rate is scaled up for the large batch too (the "linear
    # scaling rule" -- see the module docstring) so round 2 trains a model
    # that's actually worth comparing, not just a fast but undertrained one.
    small_batch_size = 32
    small_learning_rate = 1e-4
    large_batch_size = 2048
    large_learning_rate = 8e-3

    X, y = make_synthetic_dataset(num_samples=5000, in_features=in_features, num_classes=num_classes)
    split = int(0.8 * len(X))
    train_ds = TensorDataset(X[:split], y[:split])
    val_ds = TensorDataset(X[split:], y[split:])

    small_results = run_round(
        "Round 1 -- small batch size", small_batch_size, in_features, num_classes, hidden_features,
        dropout, small_learning_rate, num_epochs, noise_std, train_ds, val_ds,
    )
    large_results = run_round(
        "Round 2 -- large batch size", large_batch_size, in_features, num_classes, hidden_features,
        dropout, large_learning_rate, num_epochs, noise_std, train_ds, val_ds,
    )

    # ------------------------------------------------------------------
    # NEW IN THIS LESSON: put both rounds side by side so the crossover
    # -- CPU ahead with small batches, GPU ahead with large ones -- is
    # visible in one place instead of two separate tables you have to
    # scroll back to compare yourself.
    # ------------------------------------------------------------------
    if len(small_results) == 2 and len(large_results) == 2:
        print("\n\n" + "=" * 70)
        print("Summary: which device won, at each batch size?")
        print("=" * 70)
        for label, results in [("small batch", small_results), ("large batch", large_results)]:
            cpu, gpu = results
            if gpu["avg_epoch_ms"] < cpu["avg_epoch_ms"]:
                winner = f"GPU ({cpu['avg_epoch_ms'] / gpu['avg_epoch_ms']:.2f}x faster than CPU)"
            else:
                winner = f"CPU ({gpu['avg_epoch_ms'] / cpu['avg_epoch_ms']:.2f}x faster than GPU)"
            print(f"  {label} (batch_size={cpu['batch_size']:>5}): {winner}")
        print(
            "\nSame model, same data, same code -- what changed between rounds was the batch "
            "size (32 -> 2048) and, to compensate for it, the learning rate (1e-4 -> 8e-3). "
            "That's the whole story: a GPU's advantage isn't automatic, it has to be earned "
            "with enough parallel work per batch -- and getting a large batch size to actually "
            "train well takes a matching learning rate, exactly as lesson 5 showed."
        )


if __name__ == "__main__":
    main()
