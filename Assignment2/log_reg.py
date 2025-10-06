import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Seed initialization
seed = sum([ord(c) for c in "Pavan"])
tf.random.set_seed(seed)
np.random.seed(seed)

# Load Fashion MNIST dataset
mnist = fetch_openml("Fashion-MNIST", version=1)
X = np.array(mnist.data) / 255.0  # Normalize
y = np.array(mnist.target.astype(int))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Convert to tensors
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

# Initialize parameters
W = tf.Variable(tf.random.normal([X_train.shape[1], 10], dtype=tf.float32))
b = tf.Variable(tf.zeros([10], dtype=tf.float32))

# Hyperparameters
learning_rate = 0.01
epochs = 100
batch_size = 64

# Loss function (Sparse Categorical Crossentropy)
def compute_loss(y_true, y_pred):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

# Training loop
train_losses = []
for epoch in range(epochs):
    epoch_loss = 0
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        
        with tf.GradientTape() as tape:
            logits = tf.matmul(X_batch, W) + b
            loss = compute_loss(y_batch, logits)
        
        gradients = tape.gradient(loss, [W, b])
        W.assign_sub(learning_rate * gradients[0])
        b.assign_sub(learning_rate * gradients[1])
        
        epoch_loss += loss.numpy()
    
    train_losses.append(epoch_loss / (len(X_train) // batch_size))
    print(f"Epoch {epoch}: Loss = {train_losses[-1]}")

# Plotting
plt.plot(train_losses)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
