import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

# Seed initialization
seed = sum([ord(c) for c in "Pavan"])
tf.random.set_seed(seed)
np.random.seed(seed)

# Generate synthetic data
n_samples = 100
X = np.linspace(0, 10, n_samples)
y = 3 * X + 7 + np.random.normal(0, 2, n_samples)

# Convert to tensors
X_train = tf.convert_to_tensor(X, dtype=tf.float32)
y_train = tf.convert_to_tensor(y, dtype=tf.float32)

# Initialize parameters
W = tf.Variable(tf.random.normal([1], dtype=tf.float32))
b = tf.Variable(tf.zeros([1], dtype=tf.float32))

# Hyperparameters
learning_rate = 0.01
epochs = 1000
patience = 50
prev_loss = float('inf')
no_improvement = 0

# Loss function (Mean Squared Error)
def compute_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Training loop
train_losses = []
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        y_pred = W * X_train + b
        loss = compute_loss(y_train, y_pred)
    
    gradients = tape.gradient(loss, [W, b])
    W.assign_sub(learning_rate * gradients[0])
    b.assign_sub(learning_rate * gradients[1])
    
    train_losses.append(loss.numpy())
    
    # Early stopping with patience
    if loss.numpy() > prev_loss:
        no_improvement += 1
        if no_improvement >= patience:
            learning_rate /= 2
            no_improvement = 0
    else:
        no_improvement = 0
    prev_loss = loss.numpy()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.numpy()}")

# Plotting
plt.plot(train_losses)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
