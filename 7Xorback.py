import numpy as np

# Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of Sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# XOR Inputs and Outputs
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Seed random numbers for reproducibility
np.random.seed(42)

# Network Architecture
input_neurons = 2
hidden_neurons = 2
output_neurons = 1

# Initialize Weights and Biases
W_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
b_hidden = np.random.uniform(size=(1, hidden_neurons))
W_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))
b_output = np.random.uniform(size=(1, output_neurons))

# Training Parameters
epochs = 10000
learning_rate = 0.1

# Training Loop
for epoch in range(epochs):
    # Forward Propagation
    hidden_input = np.dot(X, W_input_hidden) + b_hidden
    hidden_output = sigmoid(hidden_input)
    
    final_input = np.dot(hidden_output, W_hidden_output) + b_output
    final_output = sigmoid(final_input)
    
    # Compute Error
    error = y - final_output

    # Backward Propagation
    d_output = error * sigmoid_derivative(final_output)

    error_hidden_layer = d_output.dot(W_hidden_output.T)
    d_hidden = error_hidden_layer * sigmoid_derivative(hidden_output)

    # Update Weights and Biases
    W_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    b_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    
    W_input_hidden += X.T.dot(d_hidden) * learning_rate
    b_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate
    
    # Print loss occasionally
    if epoch % 1000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch} - Loss: {loss:.5f}")

# Final Output
print("\nTraining complete!")

# Testing
print("\nPredictions after training:")
for i in range(len(X)):
    hidden_input = np.dot(X[i], W_input_hidden) + b_hidden
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, W_hidden_output) + b_output
    final_output = sigmoid(final_input)
    print(f"Input: {X[i]} Output: {np.round(final_output)}")
