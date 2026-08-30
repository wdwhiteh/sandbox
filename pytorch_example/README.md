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

## Files in this folder

- [train.py](train.py) / [infer.py](infer.py) — the finished, fully-tuned
  version of this example. Start here if you just want to run something and
  see it work (steps below).
- [train1.py](train1.py) through [train7.py](train7.py) — a 7-lesson,
  self-contained tutorial series that starts from a naive, untuned version
  and adds exactly one new idea per file, ending up at roughly the same
  place as `train.py`. Start here if you want to understand *why* `train.py`
  is built the way it is, one change at a time. See
  [Learn by experimenting](#learn-by-experimenting-a-7-lesson-training-walkthrough)
  below.

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

Alongside training, the script also checks the model's **validation loss**
after every epoch: its loss on a set of examples that are held back and never
used to adjust weights. This tells you how well the model is actually
generalizing, as opposed to just memorizing the training examples. A healthy
pattern is train loss and val loss both dropping together at first; a common
warning sign — easy to trigger with this dataset, since the labels have
random noise mixed in — is val loss flattening out and then creeping back up
while train loss keeps falling. That's the model starting to memorize
training-specific noise instead of learning the underlying pattern, and it's
why driving val loss all the way to 0 is not a goal here: with random noise
mixed into the labels, a model can only reach 0 val loss by memorizing that
noise, not by generalizing better.

Rather than let that happen, the script uses **early stopping**: after every
epoch, if val loss is the best seen so far, it saves a copy of the model's
weights in memory; if 10 epochs go by with no new best, training stops and
the weights from that best epoch — not the final epoch — are what get
written to `model.pt`. This is why `num_epochs` is set higher than training
actually needs (500): it's just an upper bound in case improvement is still
happening, and training will typically stop well before reaching it.

This example also uses two common tricks to keep the model from just
memorizing the training examples, so it generalizes better to new, noisy
data:

- **Input noise** — a small amount of random jitter is added to the numbers
  during training (but never during evaluation or inference), so the model
  learns the overall pattern rather than exact input values.
- **Dropout** — during training, a random slice of neurons is temporarily
  ignored on each pass, which stops the network from over-relying on any
  single one.

Both are controlled at the top of [train.py](train.py) (`noise_std` and the
`dropout` argument to `SimpleClassifier`) if you want to experiment with
them.

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
Found 2 CUDA device(s), running torch on GPU.
Using device: cuda
Epoch   1 | train loss: 1.3501 acc: 39.12% | val loss: 1.3169 acc: 51.20%
Epoch   2 | train loss: 1.2863 acc: 52.60% | val loss: 1.2432 acc: 60.10%
...
Epoch 329 | train loss: 0.2172 acc: 90.90% | val loss: 0.1770 acc: 92.90%
...
Epoch 339 | train loss: 0.2323 acc: 90.08% | val loss: 0.1771 acc: 92.90%
No val loss improvement for 10 epochs, stopping early (best was epoch 329, val loss 0.1770).
Saved model from epoch 329 (val loss 0.1770) to model.pt
```

- **First line** — tells you what device it found and is training on. If no
  GPU is available it would say `No GPU found, running torch on CPU.`
  instead (or report an Apple MPS device on an Apple Silicon Mac) — same
  result, just slower.
- **`Epoch` lines** — one per pass through the training data, showing
  performance on the training examples (`train loss`/`acc`) side by side
  with performance on the held-out validation examples (`val loss`/`acc`).
  Watch `val loss`: it should trend down for a long stretch and then level
  off — that leveling-off point is roughly where the model has learned as
  much of the real pattern as it's going to, and further training would
  mostly just fit noise. `val acc` is the more trustworthy number for judging
  the model, since it's measured on examples the model never adjusted its
  weights on.
- **`No val loss improvement...` line** — training stopped itself once val
  loss hadn't improved for 10 epochs in a row, rather than running the full
  epoch budget regardless. This is **early stopping**.
- **`Saved model from epoch...` line** — the weights written to `model.pt`
  are from whichever epoch had the *best* val loss (here, epoch 329) — not
  from the final epoch training happened to stop on. Those aren't always the
  same epoch, since val loss can wobble slightly from one epoch to the next.

Exactly which epoch training stops on, and how many total epochs it takes,
will vary a bit each run (model weights start out randomly), so don't expect
to see these exact numbers.

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

## Learn by experimenting: a 7-lesson training walkthrough

`train.py` is the finished product. This section walks through
**[train1.py](train1.py) → [train7.py](train7.py)**, seven small,
self-contained scripts that rebuild it from scratch, one idea at a time.
Every lesson file is fully runnable on its own (`python train3.py`, etc.)
and saves to its own checkpoint (`model3.pt`, etc.) so running one doesn't
overwrite another's results, and every one ends by testing itself on the
same 20 unseen examples, reporting an accuracy percentage, so you can
compare across lessons. Read the comments inside each file for the full
explanation — this section summarizes what changed and what actually
happened when each version was run.

A note on the numbers below: the model's starting weights are randomized
each time (that's normal — see [What "training" means](#what-training-means)),
so if you run these yourself, your exact numbers — and even which lesson
comes out ahead on any single run — will differ from what's quoted here,
sometimes by a lot with only 20 inference samples to measure against. What
should stay consistent is the *pattern* each lesson is demonstrating, not
the exact digits; where a comparison didn't come out the way you might
expect, that's called out honestly below rather than smoothed over.

### At a glance

| Lesson | batch size | learning rate | dropout | patience | input noise | best epoch's val loss / val acc | inference (of 20) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 — baseline | 64 | 1e-3 | none | none | none | 0.1811 / 92.10% *(checked once, at the end)* | 17/20 (85.0%) |
| 2 — + validation | 64 | 1e-3 | none | none | none | 0.1545 / 93.50% *(best was the final epoch)* | 18/20 (90.0%) |
| 3 — + epochs & patience | 64 | 1e-3 | none | 15 | none | 0.1468 / 93.80% (epoch 11 of 26) | 18/20 (90.0%) |
| 4 — + dropout | 64 | 1e-3 | 0.1 | 15 | none | 0.1644 / 93.50% (epoch 20 of 35) | 18/20 (90.0%) |
| 5 — + tuned batch/LR | 32 | 1e-4 | 0.1 | 15 | none | 0.1468 / 94.20% (epoch 109 of 124) | 18/20 (90.0%) |
| 6 — + input noise | 32 | 1e-4 | 0.1 | 15 | 0.1 | 0.1707 / 92.90% (epoch 100 of 115) | 17/20 (85.0%) |
| 7 — CPU vs. GPU, batch 32 *(fixed 30 epochs, no patience — see below)* | 32 | 1e-4 | 0.1 | none | 0.1 | 0.2199 / 92.40% (CPU) · 0.2224 / 92.00% (GPU) | 90.0% (CPU) · 95.0% (GPU) |
| 7 — CPU vs. GPU, batch 2048 *(same lesson, larger batch + scaled-up LR — see below)* | 2048 | 8e-3 | 0.1 | none | 0.1 | 0.1851 / 92.10% (CPU) · 0.1833 / 92.50% (GPU) | 90.0% (CPU) · 90.0% (GPU) |

### Lesson 1: the naive baseline

**Settings:** `batch_size=64`, `learning_rate=1e-3`, `num_epochs=10`, no
dropout, no early stopping. Trains blind for exactly 10 epochs, then checks
the held-out data exactly once, after training has already finished.

**What we measured:** by epoch 10, train accuracy reached 95.65% (loss
0.1225). The one-time check afterward showed 92.10% accuracy (loss 0.1811)
on data the model never trained on — a real gap between "how it did on
training data" and "how it does on new data." Inference on 20 unseen
examples: 17/20 correct (85.0%).

**Takeaway:** this *looks* fine, and for only 10 epochs it mostly is — but
that's partly luck. Nothing here would have warned us if training had gone
wrong, because we never looked until it was too late to do anything about
it.

### Lesson 2: watch validation loss and accuracy every epoch

**What's new:** identical settings to lesson 1 — same batch size, learning
rate, and 10 epochs. The only change is checking the held-out data after
*every* epoch instead of once at the end, and printing train and validation
numbers side by side.

**What we measured:** train accuracy climbed to 95.28% by epoch 10 (loss
0.1266), similar to lesson 1. Watching every epoch shows validation loss
dropping steadily the whole way — 0.78 at epoch 1 down to 0.1545 at epoch
10, its lowest point of the run. Inference: 18/20 correct (90.0%).

(Since lessons 1 and 2 use identical hyperparameters, the differences in
their exact numbers — including the inference count — come from random
weight initialization, not from anything this lesson changed. The point of
lesson 2 isn't better numbers; it's better *visibility* into what's
happening.)

**Takeaway:** in this particular run, 10 epochs wasn't yet enough to see
validation loss turn around — it was still improving right up to the last
epoch. That's a useful, honest result in itself: whether overfitting shows
up within any given number of epochs depends on the random starting point,
so you can't just eyeball a handful of epochs and conclude everything's
fine. Lesson 3 trains for much longer specifically to make the pattern
show up reliably, regardless of how a particular run happens to start.

### Lesson 3: raise the epoch ceiling, and add patience (early stopping)

**What's new:** `num_epochs` raised from 10 to 200 (an upper bound, not a
target), plus **patience=15**: after every epoch, if validation loss is the
best seen so far, the model's weights are saved in memory; if 15 epochs pass
with no new best, training stops, and the *best* epoch's weights (not the
final epoch's) are what get written to disk.

**What we measured:** training ran for 26 epochs before patience triggered.
By then, train loss had fallen all the way to 0.0660 (97.80% train
accuracy) — far better than lesson 2 ever reached — but validation loss had
climbed back up to 0.1784, clearly worse than its best. That best validation
loss (0.1468, 93.80% accuracy) happened much earlier, at epoch 11, which is
what got saved. Inference: 18/20 correct (90.0%).

**Takeaway:** this is overfitting laid bare — given 200 epochs to work with,
train performance kept improving long after validation performance peaked
and started getting *worse*. Patience caught it automatically and rolled
the saved model back to epoch 11, without needing a person watching the
numbers and deciding when to stop by hand.

### Lesson 4: add dropout layers

**What's new:** two `nn.Dropout(0.1)` layers added to the model, right
after each hidden layer's ReLU. During training only, each one randomly
zeroes out 10% of that layer's outputs on every batch, which discourages the
network from depending too heavily on any single neuron. Everything else
(batch size, learning rate, patience) is unchanged from lesson 3.

**What we measured:** training ran longer before stopping (35 epochs, best
at epoch 20) than lesson 3 did, and its best validation loss (0.1644, 93.50%
accuracy) was actually very slightly *worse* than lesson 3's (0.1468,
93.80%) in this particular run — worth saying plainly, since it's not the
result you'd expect. But look at the *gap* between train and validation
performance at each one's best epoch: lesson 3's was train loss 0.1181 vs.
val loss 0.1468 (a gap of 0.029); lesson 4's was train loss 0.1601 vs. val
loss 0.1644 (a gap of only 0.004) — dramatically smaller. The same shows up
in how each model behaved *after* its best epoch: over the next 15 epochs,
lesson 3's validation loss rose by 0.032, while lesson 4's rose by only
0.012. Inference: 18/20 correct (90.0%).

**Takeaway:** dropout didn't win on the single headline number this time,
but it did exactly what it's supposed to do structurally — it kept train and
validation performance much closer together, and made the climb back up
after the best epoch noticeably slower. That's a more reliable signal of
"less overfitting" than the raw validation loss number alone, and a good
reminder to look at more than one metric before deciding whether a change
helped.

### Lesson 5: tune the batch size and learning rate

**What's new:** `batch_size` lowered from 64 to 32 (each weight update sees
half as many examples, so gradients are noisier) and `learning_rate` lowered
from 1e-3 to 1e-4 (each update moves the weights 10x less). The epoch
ceiling was raised to 300 to give the slower-moving optimizer room to work.
Dropout and patience are unchanged from lesson 4.

**What we measured:** training took much longer to find its best point —
109 epochs, versus 20 for lesson 4 — and stopped at epoch 124. This time it
paid off clearly: the best validation loss (0.1468, 94.20% accuracy) matched
lesson 3's raw loss number while beating every lesson so far on validation
accuracy. Just as notable, the whole validation-loss curve declined smoothly
and steadily across all 109+ epochs with no sharp reversals, unlike lesson
3's spikier climb back up after its peak. Inference: 18/20 correct (90.0%).

**Takeaway:** tuning batch size and learning rate down bought two things at
once here: a genuinely strong validation accuracy, and a much smoother,
more stable training curve — at the cost of needing roughly 5x as many
epochs (and wall-clock time) to get there. That tradeoff — slower but
steadier — is a common one when tuning these two hyperparameters, and it's
exactly why "hyperparameter tuning" usually means trying several
combinations and comparing validation results rather than picking values by
intuition alone.

### Lesson 6 (bonus): input noise injection for robustness

**What's new:** during training only, a small amount of random noise
(`noise_std=0.1`) is added to each batch of inputs before the model sees
them. Validation and inference still use the data exactly as generated, with
no noise added. This wasn't one of the four original concepts — it's
included because it answers a different, practical question: how well does
the model hold up on inputs that are a little imprecise, the way real-world
sensor or measurement data often is?

**What we measured:** training stopped at epoch 115 (best at epoch 100),
with validation loss 0.1707 (92.90% accuracy) — worse than lesson 5's on
both counts. Train accuracy at the best epoch was also noticeably lower than
lesson 5's (90.88% vs. 93.27%), which makes sense: the model was being
trained on deliberately corrupted inputs, so doing well on the exact
training batch is expected to get harder. Inference on the 20 clean, unseen
examples: 17/20 correct (85.0%), the weakest of any lesson.

To actually test the robustness claim rather than just guess, lesson 5's
and lesson 6's saved models were both run on a separate batch of 2,000
examples with varying amounts of noise added at *test* time (something
neither script does by default) — and this was done twice, against two
independently-trained pairs of checkpoints, to see if the result was a
fluke:

| Test-time noise (std) | Lesson 5 accuracy (no noise training) | Lesson 6 accuracy (noise training) | Difference |
| --- | --- | --- | --- |
| 0.00 | 93.65% | 92.70% | −0.95pp |
| 0.05 | 93.45% | 92.50% | −0.95pp |
| 0.10 | 92.45% | 91.30% | −1.15pp |
| 0.15 | 90.95% | 89.95% | −1.00pp |
| 0.20 | 89.10% | 88.45% | −0.65pp |
| 0.30 | 83.90% | 84.05% | +0.15pp |

**Takeaway — and this is an important one:** across two separate training
runs, the numbers consistently show *no* robustness benefit from noise
injection here; if anything, lesson 6 was slightly worse at almost every
noise level tested, both on clean data and on noisy data. This doesn't mean
input noise injection is a bad idea in general — it's a well-established
technique — but on this particular small, synthetic, already-regularized
(thanks to dropout) dataset, at this noise level, it simply didn't help.
That's the real lesson: a technique sounding reasonable, or even being
"well-established," isn't the same as it helping *your* model on *your*
data. The only way to know is to measure it, exactly like this — twice, if
the first result surprises you — not to assume it worked because the idea
makes sense.

### Lesson 7 (bonus): CPU vs. GPU — does the GPU actually help here?

**What's new:** this lesson trains the exact same model twice per round,
once forced onto the CPU and once on the GPU, using the same starting
weights and the same batch order for both (so any difference in the numbers
is real device behavior, not random luck), timing every epoch on both.
Unlike lessons 3-6, there's no early stopping here — within a round, CPU
and GPU always train for the same fixed 30 epochs, so the *timing*
comparison is apples-to-apples rather than one run stopping earlier than
the other.

It then does that TWICE — once with a **small batch size (32)**, same as
every earlier lesson, and again with a **large batch size (2048)** — because
batch size turns out to be the whole story on speed. But raising the batch
size 64x also means 64x fewer weight-update steps per epoch (2 vs. 125),
so round 2 also raises `learning_rate` from 1e-4 to 8e-3 to compensate —
the same batch-size/learning-rate coupling lesson 5 introduced, applied
here so round 2 trains a model actually worth comparing, not just a fast
but broken one. That value wasn't guessed: a handful of learning rates were
tried at batch size 2048 and checked against validation accuracy, exactly
the "measure, don't guess" approach the rest of this series uses.

**Round 1 (batch size 32, learning rate 1e-4):**

| metric | CPU | GPU |
| --- | --- | --- |
| total time (30 epochs) | 3.00 s | 6.45 s |
| avg time per epoch (after warm-up) | 90.9 ms | 203.5 ms |
| final val loss / acc | 0.2199 / 92.40% | 0.2224 / 92.00% |
| inference accuracy (20 samples) | 90.0% | 95.0% |

CPU was **2.24x faster per epoch** than the GPU. Every batch sent to the GPU
has to be copied over from system memory, and every operation pays a small,
fixed cost just to launch on the GPU, before any actual computation
happens. With only 32 tiny examples per batch, that fixed overhead is
bigger than the real work — the GPU spends most of its time waiting, not
calculating.

**Round 2 (batch size 2048, learning rate 8e-3):**

| metric | CPU | GPU |
| --- | --- | --- |
| total time (30 epochs) | 1.03 s | 0.64 s |
| avg time per epoch (after warm-up) | 33.7 ms | 21.3 ms |
| final val loss / acc | 0.1851 / 92.10% | 0.1833 / 92.50% |
| inference accuracy (20 samples) | 90.0% | 90.0% |

This time the **GPU was 1.58x faster per epoch** — the crossover actually
happened — and, with the learning rate scaled to match, round 2's model is
genuinely competitive with round 1's (92.1-92.5% val accuracy vs.
92.0-92.4%), not just fast. With 2048 examples bundled into each batch,
there's finally enough parallel work per batch that the GPU's fixed
per-batch overhead is worth paying, and its ability to crunch through
thousands of examples at once starts to win out over the CPU doing the
same work more sequentially.

(For comparison, the first version of this lesson left `learning_rate` at
1e-4 for round 2 too. With 64x fewer weight-update steps and no larger
steps to compensate, that version's round 2 only reached 35-36% validation
accuracy — a broken-looking result that had nothing to do with CPU vs. GPU
and everything to do with an untuned learning rate. Fixing that, rather
than just footnoting it away, is what turned this into a fair comparison.)

**Takeaway:** a GPU's advantage comes from doing enormous numbers of
calculations in parallel, which only pays off once there's enough work per
batch to fill that parallelism — and getting real benefit from a larger
batch size takes a matching learning rate, not just the batch size change
on its own. Same model, same data, same code — what changed between the two
rounds was the batch size and, deliberately, the learning rate that goes
with it, and that combination was enough to flip which device won while
keeping both rounds' models genuinely comparable. This is exactly why this
whole tutorial series has used `get_device()` to automatically pick "the
best available" hardware rather than hard-coding GPU-or-nothing: "best"
genuinely depends on the size of the job, and the only way to know which
one is faster for *your* model is what this lesson just did — measure it
directly, the same way lesson 6 measured (and didn't just assume) whether
noise injection helped.

Try it yourself with `python train7.py`. The two batch sizes and learning
rates are set at the top of `main()` in [train7.py](train7.py)
(`small_batch_size`/`small_learning_rate` and
`large_batch_size`/`large_learning_rate`) — try other values, or change
`hidden_features` too, to see where the crossover point falls on your own
machine.
