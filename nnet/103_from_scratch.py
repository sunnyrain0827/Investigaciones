#
# Based on http://iamtrask.github.io/2015/07/12/basic-python-network/
#
# Heavily modified to incorporate full zoom on the mechanics
# of backprop. No hand-waving permitted!

import numpy as np

num_iterations = 60000

# Input data
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# Output labels
y = np.array([[0],
              [1],
              [1],
              [0]])


# randomly initialize our weights with mean 0
np.random.seed(1337)
W1 = 2 * np.random.random((3, 4)) - 1
W2 = 2 * np.random.random((4, 1)) - 1

f       = lambda x: 1.0/(1.0 + np.exp(-x))         # sigmoid "non-linearity"
# f_prime = lambda x: x * (1 - x)                  # derivative of sigmoid
f_prime = lambda x: np.exp(-x)/((1+np.exp(-x))**2) # derivative of sigmoid

# for iter in range(num_iterations):
#
#     # forward propagation, 2 layers
#     z1 = np.dot(X, W1)
#     a1 = f(z1)
#
#     z2 = np.dot(a1, W2)
#     y_hat = f(z2)
#
#     # how much did we miss the target value?
#     loss = y - y_hat
#
#     if (iter % 10000) == 0:
#         print("Loss: " + str(np.mean(np.abs(loss))))
#
#     # in what direction is the target value?
#     loss_delta = loss * f_prime(y_hat)
#
#     # how much did each l1 value contribute to the l2 error (according to the weights)?
#     l1_error = loss_delta.dot(W2.T)
#
#     # in what direction is the target h1?
#     h1_delta = l1_error * f_prime(a1)
#
#     W2 += (a1.T).dot(loss_delta)
#     W1 += (X.T).dot(h1_delta)
#
# print(y_hat)


for iter in range(num_iterations):

    # forward propagation, 2 layers
    z1 = np.dot(X, W1)
    a1 = f(z1)
    z2 = np.dot(a1, W2)
    y_hat = f(z2)

    # loss = y - y_hat
    loss = 0.5 * sum((y - y_hat) ** 2)

    if (iter % 10000) == 0:
        print("Loss: " + str(np.mean(np.abs(loss))))

        print("X : " + str(X.shape))
        print("X.T: " + str(X.T.shape))
        print("W2 : " + str(W2.shape))
        print("W2.T : " + str(W2.T.shape))

        print("a1.T : " + str(a1.T.shape))
        print("-1 * (y - y_hat) * f_prime(z2) : " + str(-1 * (y - y_hat) * f_prime(z2)))
        print("f_prime(z1) : " + str(f_prime(z1)))
        print("f_prime(z2) : " + str(f_prime(z2)))

    dJdW2 = a1.T.dot(       -1 * (y - y_hat) * f_prime(z2))
    # dJdW1 = X.T.dot((np.dot(-1 * (y - y_hat) * f_prime(z2), W2.T) * f_prime(z1)))
    temp = np.dot(-1 * (y - y_hat) * f_prime(z2), W2.T) * f_prime(z1)
    dJdW1 = X.T.dot((np.dot(-1 * (y - y_hat) * f_prime(z2), W2.T) * f_prime(z1)))

    W2 = W2 - dJdW2
    W1 = W1 - dJdW1

print(dJdW1)
print(dJdW2)

print("yhat: \n" + str(y_hat))