# PyTorch Example: Teaching a Computer to Sort Things Into Categories

This is a small, complete example of an AI model: it builds one, trains it,
and then uses it to make predictions. You don't need to know how to code or
know anything about AI to run it and understand what's happening — this guide
explains every step in plain language.

## What this app actually does

The task is **classification**: given a description of something, decide
which of a few categories it belongs to. A real-world example would be
looking at a photo and deciding "cat," "dog," or "bird." To keep this example
self-contained (no downloads, no real-world dataset needed), it makes up its
own data instead: each "example" is just 20 random numbers, and each one
secretly belongs to one of 4 made-up categories (`0`, `1`, `2`, or `3`).

The model's job is to look at those 20 numbers and guess the category. It
starts out guessing randomly (about 25% correct, since there are 4 options),
and gets better through **training**, described below.

## The model, illustrated

The model is a small **neural network** — loosely inspired by how brain cells
pass signals to each other. It's organized into layers: numbers go in one
side, get transformed step by step, and a set of scores comes out the other
side.

```
   INPUT                  HIDDEN LAYER 1            HIDDEN LAYER 2            OUTPUT
20 numbers in           64 neurons look           64 neurons combine      4 scores, one
 (one example)           for simple patterns       those into more          per category
                                                     complex patterns

    o                        o                          o                      o  -> Category 0
    o                        o                          o                      o  -> Category 1
    o   --- weights --->     o   --- weights --->       o   --- weights --->   o  -> Category 2
    o                        o                          o                      o  -> Category 3
    :                        :                          :
    o (20th)                 o (64th)                   o (64th)          highest score
                        [ReLU: drop negatives]     [ReLU: drop negatives]     wins
```

- Each **"neuron"** just computes a weighted sum of the numbers coming into it
  (multiply each input by an adjustable number called a "weight," add them
  up). This is what the arrows labeled `weights` represent.
- **ReLU** is a tiny rule applied after each hidden layer: if a number comes
  out negative, replace it with zero; otherwise leave it alone. Without a
  rule like this, stacking layers would be mathematically the same as having
  just one layer — ReLU is what lets the network learn more complex,
  non-straight-line patterns.
- The **output layer** produces one score per category. The model's
  prediction is simply whichever category got the highest score.

This exact structure is defined in [train.py](train.py) as the
`SimpleClassifier` class — `in_features=20`, two hidden layers of 64 neurons
each, `num_classes=4`.

## What "training" means

When the model is first created, every weight (every number on every arrow
above) is random, so its guesses are random too. Training is the process of
nudging those weights to make better guesses:

1. Show the model a batch of examples it hasn't adjusted to yet.
2. Compare its guesses to the correct answers, and measure how wrong it was.
   This "wrongness" number is called the **loss** — lower is better.
3. Automatically figure out which direction to nudge each weight to make the
   loss a little smaller (PyTorch does this step, called
   "backpropagation," for you).
4. Repeat, thousands of times, over many passes through the data. Each full
   pass through all the training examples is called an **epoch**.

Over enough epochs, the weights settle into values that make genuinely good
predictions — that's "the model learned."

## What "inference" means

Once training is done, the weights are frozen — no more adjusting. Using the
trained model to make a prediction on a new example is called **inference**.
This is the "real" use of the model: everything before this point was just
preparation.

## Step-by-step: run it yourself

These steps assume you're working inside this project's dev container (see
[.devcontainer/README.md](../.devcontainer/README.md) for how to open it in
VS Code) — Python and PyTorch are already installed there.

### 1. Open a terminal

In VS Code: **Terminal > New Terminal** (or press `` Ctrl+` ``). A terminal is
just a place to type text commands instead of clicking buttons.

### 2. Install dependencies (only needed outside the dev container)

If you're using the dev container, PyTorch is already installed and you can
skip this step. Otherwise:

```bash
pip install -r requirements.txt
```

If you have an NVIDIA GPU, install the CUDA-enabled build of PyTorch instead
(see https://pytorch.org/get-started/locally/ for the right command for your
CUDA version).

### 3. Train the model

In the terminal, type this and press Enter:

```bash
python train.py
```

#### What you'll see

```
Using device: cuda
Epoch  1 | loss: 0.4968 | accuracy: 79.25%
Epoch  2 | loss: 0.2112 | accuracy: 91.17%
...
Epoch 10 | loss: 0.1117 | accuracy: 95.78%
Eval        | loss: 0.2681 | accuracy: 90.70%
Saved trained model to model.pt
```

- **`Using device: cuda`** — tells you the model is training on the GPU
  (`cuda`). If no GPU is available it would say `cpu` instead (or `mps` on an
  Apple Silicon Mac) — same result, just slower.
- **`Epoch` lines** — one per pass through the training data. `loss` should
  generally go down and `accuracy` should generally go up as training
  progresses — that's the model improving.
- **`Eval` line** — accuracy on examples the model never trained on, which is
  the fairest measure of how good it actually is (it's normal for this to be
  a bit lower than the last training epoch's accuracy).
- **`Saved trained model to model.pt`** — the learned weights are written to
  a new file, `model.pt`, in this folder. That file is what makes the next
  step possible.

### 4. Run inference with the trained model

```bash
python infer.py
```

#### What you'll see

```
Using device: cuda
sample 0: predicted=3 actual=3 [OK]
sample 1: predicted=2 actual=3 [MISS]
sample 2: predicted=2 actual=2 [OK]
...
```

This loads `model.pt` and shows the model's prediction (`predicted`) next to
the true category (`actual`) for 8 brand-new examples it has never seen
before — `[OK]` means it guessed correctly, `[MISS]` means it didn't. A few
misses are expected: random noise was mixed into the categories when the data
was generated, so even a well-trained model won't hit 100%. This is the same
"load a saved model and ask it about new data" pattern used for real models
on real data.
