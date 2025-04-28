import numpy as np

# Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of Sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Training Data (XOR Problem for simplicity)
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Network Architecture
input_neurons = 2    # Input features
hidden_neurons = 4   # Hidden layer neurons
output_neurons = 1   # Output layer

# Initialize Weights and Biases
np.random.seed(1)
W_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
b_hidden = np.random.uniform(size=(1, hidden_neurons))
W_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))
b_output = np.random.uniform(size=(1, output_neurons))

# Training Parameters
epochs = 10000
learning_rate = 0.1

# Training Process
for epoch in range(epochs):
    # Forward Propagation
    hidden_input = np.dot(X, W_input_hidden) + b_hidden
    hidden_output = sigmoid(hidden_input)
    
    final_input = np.dot(hidden_output, W_hidden_output) + b_output
    final_output = sigmoid(final_input)
    
    # Loss Calculation (MSE)
    loss = np.mean((y - final_output) ** 2)
    
    # Back Propagation
    error = y - final_output
    d_output = error * sigmoid_derivative(final_output)

    error_hidden = d_output.dot(W_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)
    
    # Updating Weights and Biases
    W_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    b_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    
    W_input_hidden += X.T.dot(d_hidden) * learning_rate
    b_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate
    
    # Print loss occasionally
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} - Loss: {loss:.5f}")

print("\nTraining complete!")

# Testing
print("\nFinal predictions after training:")
print(final_output)
