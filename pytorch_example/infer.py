"""
Loads the model trained by train.py and runs inference on a handful of new,
unseen samples. Run `python train.py` first to produce model.pt.
"""

import torch

from train import MODEL_PATH, SimpleClassifier, get_device, make_synthetic_dataset


def main():
    device = get_device()
    print(f"Using device: {device}")

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = SimpleClassifier(
        checkpoint["in_features"],
        checkpoint["hidden_features"],
        checkpoint["num_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Stand in for "new" real-world inputs the model has never seen (seed=1
    # draws different samples than train.py's seed=0 training/eval split,
    # while still using the same ground-truth labeling function).
    X, y = make_synthetic_dataset(
        num_samples=100,
        in_features=checkpoint["in_features"],
        num_classes=checkpoint["num_classes"],
        seed=1,
    )
    X = X.to(device)

    with torch.no_grad():
        logits = model(X)
        predictions = logits.argmax(dim=1)

    hits = 0
    for i, (pred, actual) in enumerate(zip(predictions.tolist(), y.tolist())):
        marker = "OK" if pred == actual else "MISS"
        ok = pred == actual
        hits += ok
        print(f"sample {i}: predicted={pred} actual={actual} [{marker}]")
        
    print(f"  {hits}/100 correct (accuracy: {hits / 100:.1%})")


if __name__ == "__main__":
    main()
