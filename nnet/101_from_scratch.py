#
# Based on http://iamtrask.github.io/2015/07/12/basic-python-network/
#
# Heavily modified to incorporate full zoom on the mechanics
# of backprop. No hand-waving permitted!

import numpy as np

# Input data
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# Output labels
y = np.array([[0],
              [0],

              [1],
              [1]])


# initialize weights randomly with mean 0
np.random.seed(1)
W0 = 2 * np.random.random((3, 1)) - 1

f       = lambda x: 1.0/(1.0 + np.exp(-x))    # sigmoid "non-linearity"
f_prime = lambda x: x * (1 - x)               # derivative of sigmoid

print("X shape: " + str(X.shape))
print("y shape: " + str(y.shape))
print("W0 shape: " + str(W0.shape))
print("np.dot shape: " + str(np.dot(X, W0).shape))

for iter in range(60000):

    # forward propagation
    y_hat = f(np.dot(X, W0))
    loss = y - y_hat     # how much did we miss?

    # print("W0: \n" + str(W0))
    if (iter % 1000) == 0:
        print("Loss: " + str(np.mean(np.abs(loss))))
        # print("np.dot X, W0: \n" + str(np.dot(X, W0)))

    # multiply how much we missed by the
    # slope of the sigmoid at the values in y_prime
    loss_delta = loss * f_prime(y_hat)

    # update weights
    # W0 += np.dot(X.T, loss_delta)
    W0 += np.dot(loss_delta.T, X).T
    # print("deltas \n" + str(np.dot(X.T, loss_delta)) + "\n\n")

print("Output After Training:")
print(y_hat)