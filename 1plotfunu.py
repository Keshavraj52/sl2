import numpy as np
import matplotlib.pyplot as plt

# Define Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def swish(x):
    return x * sigmoid(x)

# Generate input values
x = np.linspace(-10, 10, 500)

# Compute outputs
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_swish = swish(x)

# Create subplots
plt.figure(figsize=(12, 8))

# Sigmoid
plt.subplot(2, 3, 1)
plt.plot(x, y_sigmoid, label='Sigmoid', color='blue')
plt.title('Sigmoid')
plt.grid(True)

# Tanh
plt.subplot(2, 3, 2)
plt.plot(x, y_tanh, label='Tanh', color='green')
plt.title('Tanh')
plt.grid(True)

# ReLU
plt.subplot(2, 3, 3)
plt.plot(x, y_relu, label='ReLU', color='red')
plt.title('ReLU')
plt.grid(True)

# Leaky ReLU
plt.subplot(2, 3, 4)
plt.plot(x, y_leaky_relu, label='Leaky ReLU', color='purple')
plt.title('Leaky ReLU')
plt.grid(True)

# Swish
plt.subplot(2, 3, 5)
plt.plot(x, y_swish, label='Swish', color='orange')
plt.title('Swish')
plt.grid(True)

# Layout adjustment
plt.tight_layout()
plt.suptitle('Common Activation Functions', fontsize=16, y=1.05)
plt.show()
