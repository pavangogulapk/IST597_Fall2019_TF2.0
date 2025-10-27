#!/usr/bin/env python3
"""
forgetting_mlp_unique.py
Unique implementation for CS 599 assignment: Measuring Catastrophic Forgetting
Author: (You can put your name)
Date: (Today's date)

Requirements:
- Python 3.8+
- tensorflow >= 2.0
- numpy, matplotlib
- scikit-learn (for shuffling/permute utilities) -- optional but included

Usage examples:
python forgetting_mlp_unique.py --depth 3 --optimizer adam --loss nll --dropout 0.2 --seed 42
python forgetting_mlp_unique.py --depth 2 --optimizer sgd --loss l1l2 --dropout 0.0 --seed 123
"""

import os
import argparse
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------------
# Reproducibility helpers
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# ---------------------------
# Permuted MNIST generator
# ---------------------------
def make_permuted_mnist_tasks(num_tasks=10, seed=0):
    """
    Returns a list of (x_train, y_train), (x_test, y_test) pairs for each task,
    where each task is the same MNIST data but with a fixed random pixel permutation.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalize and flatten
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train.reshape(-1, 28*28)
    x_test = x_test.reshape(-1, 28*28)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    rng = np.random.RandomState(seed)
    perms = []
    tasks = []
    for t in range(num_tasks):
        perm = rng.permutation(28*28)
        perms.append(perm)
        xt = x_train[:, perm]
        xv = x_test[:, perm]
        tasks.append(((xt, y_train), (xv, y_test)))
    return tasks, perms

# ---------------------------
# Model factory
# ---------------------------
def build_mlp(input_dim=28*28, hidden_units=256, depth=2, dropout=0.0, l1=0.0, l2=0.0, num_classes=10):
    """
    Create an MLP with given depth (# hidden layers), each hidden_units wide.
    Regularization by L1/L2 applied to kernel weights.
    """
    assert depth >= 1
    regs = None
    if l1 > 0 or l2 > 0:
        regs = regularizers.L1L2(l1=l1, l2=l2)
    inputs = layers.Input(shape=(input_dim,), name="input_flat")
    x = inputs
    for i in range(depth):
        x = layers.Dense(hidden_units, activation='relu', kernel_regularizer=regs, name=f"hidden_{i+1}")(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout, name=f"drop_{i+1}")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="softmax_out")(x)
    model = models.Model(inputs=inputs, outputs=outputs, name=f"mlp_depth{depth}_u{hidden_units}")
    return model

# ---------------------------
# Loss selection helper
# ---------------------------
def get_loss_and_regularizers(loss_name):
    """
    loss_name: 'nll', 'l1', 'l2', 'l1l2' where
      - nll -> categorical_crossentropy (softmax model)
      - l1 -> categorical_crossentropy + model kernel L1 (we set via build_mlp l1)
      - l2 -> categorical_crossentropy + kernel L2
      - l1l2 -> both
    We'll return loss function and l1/l2 values for model building.
    """
    loss_name = loss_name.lower()
    if loss_name == 'nll':
        return 'categorical_crossentropy', 0.0, 0.0
    elif loss_name == 'l1':
        return 'categorical_crossentropy', 1e-5, 0.0
    elif loss_name == 'l2':
        return 'categorical_crossentropy', 0.0, 1e-4
    elif loss_name in ('l1l2', 'l1+l2'):
        return 'categorical_crossentropy', 1e-6, 1e-5
    else:
        raise ValueError("Unknown loss_name: choose from nll, l1, l2, l1l2")

# ---------------------------
# Training schedule & evaluation
# ---------------------------
def train_sequence(model, tasks, optimizer, loss, initial_task_epochs=50, subsequent_task_epochs=20, batch_size=128, verbose=2):
    """
    Train model sequentially on tasks list.
    tasks: list of ((x_train,y_train),(x_test,y_test)) pairs
    Returns:
      - R matrix: shape T x T where R[t_test, t_trained] = accuracy on test of task t_test after training on task t_trained
      - histories: list of per-task validation accuracy history dicts for plotting
    """
    T = len(tasks)
    R = np.zeros((T, T), dtype=np.float32)
    # We'll evaluate after training each task j on all tasks i (test sets).
    histories = []  # list of dicts storing val_acc series for each task training period

    # compile model initially
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    for j in range(T):
        (x_tr, y_tr), (x_te, y_te) = tasks[j]
        epochs = initial_task_epochs if j == 0 else subsequent_task_epochs

        # Fit on current task
        hist = model.fit(x_tr, y_tr, validation_data=(x_te, y_te), epochs=epochs,
                         batch_size=batch_size, verbose=verbose)
        histories.append(hist.history)

        # Evaluate on all tasks and fill R[:, j] column
        for i in range(T):
            _, (x_te_i, y_te_i) = tasks[i]
            eval_res = model.evaluate(x_te_i, y_te_i, verbose=0)
            acc = eval_res[1]  # accuracy
            R[i, j] = acc
            # optionally: print(f"After training task {j+1}, test on task {i+1} acc={acc:.4f}")
    return R, histories

# ---------------------------
# Metrics: ACC, BWT, TBWT, CBWT
# ---------------------------
def compute_ACC_BWT(R):
    """
    R: T x T matrix as defined:
      R[i,j] = test accuracy on task i after training task j (1-indexed conceptual)
    ACC = mean_{i=1..T} R[i, T]
    BWT = (1/(T-1)) * sum_{i=1..T-1} (R[i, T] - R[i, i])
    """
    T = R.shape[0]
    final_col = R[:, T-1]
    ACC = np.mean(final_col)
    BWT = np.sum(final_col[:T-1] - np.diag(R)[:T-1]) / (T - 1)
    return ACC, BWT

def compute_TBWT_CBWT(R):
    """
    TBWT (Task-averaged backward transfer) and CBWT (Cumulative backward transfer) as an example:
    TBWT: average of (R[i,T] - max_j<=i R[i,j]) maybe — implement one reasonable variant:
    We'll compute:
      TBWT_i = R[i,T] - max_{j<=i} R[i,j]
    TBWT = mean_i TBWT_i
    CBWT: sum over i of (R[i,T] - R[i,i]) / (T-1) (similar to BWT but cumulative difference)
    Return TBWT, CBWT and per-task TBWT_i
    """
    T = R.shape[0]
    tbwt_per_task = []
    for i in range(T):
        max_after_or_before = np.max(R[i, :i+1])  # best seen up to when task i finishes
        tbwt_per_task.append(R[i, T-1] - max_after_or_before)
    TBWT = np.mean(tbwt_per_task)
    CBWT = np.sum(R[:T-1, T-1] - np.diag(R)[:T-1]) / (T-1)
    return TBWT, CBWT, np.array(tbwt_per_task)

# ---------------------------
# Plotting helpers
# ---------------------------
def plot_validation_histories(histories, out_dir, title_prefix="val_acc"):
    """
    histories: list of Keras history.history dicts (one per trained task)
    Each history.history has keys 'val_accuracy' or 'val_acc' depending on TF version.
    We'll create a plot that shows the validation accuracy curves for each task training block.
    """
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for idx, h in enumerate(histories):
        # TF sometimes uses 'val_accuracy', sometimes 'val_acc'
        val_key = 'val_accuracy' if 'val_accuracy' in h else 'val_acc'
        x = np.arange(1, len(h[val_key]) + 1) + sum(len(histories[i][val_key]) for i in range(idx)) if False else np.arange(1, len(h[val_key]) + 1)
        plt.plot(x, h[val_key], marker='o', label=f"Task {idx+1} (epochs={len(h[val_key])})")
    plt.xlabel("Epochs (per task block)")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation accuracy per task training block")
    plt.legend(loc='lower right')
    plt.grid(True)
    fname = os.path.join(out_dir, f"{title_prefix}{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    return fname

# ---------------------------
# Main CLI runner
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP sequentially on permuted MNIST tasks and measure forgetting.")
    parser.add_argument("--depth", type=int, default=3, help="Hidden layer depth (2,3,4 typical).")
    parser.add_argument("--hidden_units", type=int, default=256, help="Hidden units per layer.")
    parser.add_argument("--num_tasks", type=int, default=10, help="Total number of permuted-MNIST tasks.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--loss", type=str, default="nll", choices=['nll','l1','l2','l1l2'], help="Loss variant.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout value (0.0-0.5).")
    parser.add_argument("--optimizer", type=str, default="adam", choices=['adam','sgd','rmsprop'], help="Optimizer.")
    parser.add_argument("--initial_epochs", type=int, default=50, help="Epochs for first task.")
    parser.add_argument("--subsequent_epochs", type=int, default=20, help="Epochs for subsequent tasks.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--out_dir", type=str, default="results_unique", help="Directory to save results & plots.")
    parser.add_argument("--verbose", type=int, default=2, help="Verbosity for fit()")
    return parser.parse_args()

def get_optimizer(opt_name, lr=1e-3):
    opt_name = opt_name.lower()
    if opt_name == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=lr)
    elif opt_name == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    elif opt_name == 'rmsprop':
        return tf.keras.optimizers.RMSprop(learning_rate=lr)
    else:
        raise ValueError("Unknown optimizer")

def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Starting experiment: depth={args.depth}, hidden={args.hidden_units}, loss={args.loss}, dropout={args.dropout}, opt={args.optimizer}, seed={args.seed}")

    # prepare tasks
    tasks, perms = make_permuted_mnist_tasks(num_tasks=args.num_tasks, seed=args.seed)

    # losses and regularizers
    loss_fn_name, l1, l2 = get_loss_and_regularizers(args.loss)
    if loss_fn_name == 'categorical_crossentropy':
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
    else:
        loss_fn = loss_fn_name  # fallback if needed

    # build model
    model = build_mlp(input_dim=28*28, hidden_units=args.hidden_units, depth=args.depth,
                      dropout=args.dropout, l1=l1, l2=l2, num_classes=10)

    # optimizer
    opt = get_optimizer(args.optimizer)

    # Train sequentially
    R, histories = train_sequence(model, tasks, optimizer=opt, loss=loss_fn,
                                  initial_task_epochs=args.initial_epochs,
                                  subsequent_task_epochs=args.subsequent_epochs,
                                  batch_size=args.batch_size,
                                  verbose=args.verbose)

    # Compute metrics
    ACC, BWT = compute_ACC_BWT(R)
    TBWT, CBWT, tbwt_per_task = compute_TBWT_CBWT(R)

    # Save R matrix and metrics
    np.save(os.path.join(args.out_dir, f"R_depth{args.depth}_loss{args.loss}_opt{args.optimizer}_seed{args.seed}.npy"), R)
    metrics_txt = os.path.join(args.out_dir, f"metrics_depth{args.depth}_loss{args.loss}_opt{args.optimizer}_seed{args.seed}.txt")
    with open(metrics_txt, "w") as f:
        f.write(f"Experiment parameters: {args}\\n")
        f.write(f"ACC: {ACC:.6f}\\n")
        f.write(f"BWT: {BWT:.6f}\\n")
        f.write(f"TBWT: {TBWT:.6f}\\n")
        f.write(f"CBWT: {CBWT:.6f}\\n")
        f.write("R matrix (rows=task_i test, cols=after training task_j):\\n")
        np.savetxt(f, R, fmt="%.6f")
    print(f"Saved metrics to {metrics_txt}")

    # Plot validation histories
    plot_path = plot_validation_histories(histories, args.out_dir)
    print(f"Saved validation plot to {plot_path}")

    # Print summary
    print("Summary:")
    print(f"ACC (mean final accuracies) = {ACC:.4f}")
    print(f"BWT (backward transfer) = {BWT:.4f}")
    print(f"TBWT = {TBWT:.4f}")
    print(f"CBWT = {CBWT:.4f}")
    print(f"R matrix shape: {R.shape}")

    # Save model (optional)
    model_save_path = os.path.join(args.out_dir, f"final_model_depth{args.depth}_loss{args.loss}_opt{args.optimizer}_seed{args.seed}.h5")
    model.save(model_save_path)
    print(f"Saved final model to {model_save_path}")

if _name_ == "_main_":
    main()
