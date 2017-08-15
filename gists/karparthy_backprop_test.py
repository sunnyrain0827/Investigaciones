# from https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b#.3fr1z5uva
#
# This RNN is unrolled for T time steps, you’ll see that the gradient signal going backwards 
# in time through all the hidden states is always being multiplied by the same matrix (the 
# recurrence matrix Whh), interspersed with non-linearity backprop.  What happens when you 
# take one number a and start multiplying it by some other number b (i.e. a*b*b*b*b*b*b…)? 
# This sequence either goes to zero if |b| < 1, or explodes to infinity when |b|>1. The same 
# thing happens in the backward pass of an RNN, except b is a matrix and not just a number, 
# so we have to reason about its largest eigenvalue instead.

import numpy as np

H = 5
T = 50

Whh = np.random.randn(H, H)

# forward pass of an RNN (ignoring the inputs x)
hs = {}
ss = {}
hs[-1] = np.random.randn(H)

for t in range(T):
    ss[t] =  np.dot(Whh, hs[t-1])
    hs[t] = np.maximum(0, ss[t]) # ReLU

# backward pass -- backprop of the RNN
dhs = {}
dss = {}
dhs[T-1] = np.random.randn(H) # start off the chain with a random gradient
for t in reversed(range(T)):
    dss[t] = (hs[t] > 0) * dhs[t]    # backprop through the non-linearity (ReLU)
    print(dss[t])
    dhs[t-1] = np.dot(Whh.T, dss[t]) # backprop into the previous hidden state
    print(dhs[t-1])