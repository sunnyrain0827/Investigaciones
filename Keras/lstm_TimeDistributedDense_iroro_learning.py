from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributedDense, TimeDistributed

import numpy as np

inputdata_dim = 156 #-- flipped to simulate pitch network on biaxial network
timesteps = 20
nb_classes = 156

output_dim = 300
nb_samples = 164   # -- flipped to simulate pitch network on biaxial network

model = Sequential()
model.add(LSTM(output_dim, return_sequences=True, input_dim = 2))
model.add(TimeDistributedDense(4))

model.compile(optimizer='rmsprop',loss='mse')

ins = np.random.random((nb_samples, timesteps, 2))
outs = np.random.random((nb_samples, timesteps, 4))
model.fit(x=ins,y=outs, batch_size=nb_samples, nb_epoch=50)

# now you can give it a test sequence of any duration. Here, its 1000 timesteps.
print(model.predict(x=np.random.random((1, 10, 2))))

# score = model.evaluate(X_test, Y_test, batch_size=16)
# print score
# print model.predict(X_test, batch_size=16)
