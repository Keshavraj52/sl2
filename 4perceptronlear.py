import numpy as np
import matplotlib.pyplot as plt

# define input feature set for a binary classification problem (AND logic gate)
X = np.array([[0, 0],
              [1, 0],
              [0, 1],
              [1, 1]])

# defining corresponding target labels (-1 for one class, 1 for another)
Y = np.array([-1, -1, -1, 1])

w = np.zeros(X.shape[1])  # w = [0, 0]
b = 0

# using perceptron learning algorithm
for _ in range(6):
    for i in range(X.shape[0]):  # Loop for each training sample
        y_pred = np.sign(np.dot(X[i], w) + b)
        
        if y_pred != Y[i]:  
            w += 0.3 * Y[i] * X[i]
            b += 0.3 * Y[i]

# defining the graph range
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# generating a mesh grid for plotting
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

Z = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b)  # computing perceptron output
Z = Z.reshape(xx.shape)  # reshape to match mesh grid

plt.contourf(xx, yy, Z, alpha=0.8, cmap='RdYlGn')

plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='bwr', edgecolors='k')

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Perceptron Decision Regions')

plt.show()
