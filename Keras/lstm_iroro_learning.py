from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

inputdata_dim = 164 # 156 -- flipped to simulate pitch network on biaxial network
timesteps = 20
nb_classes = 156

output_dim = 300
nb_samples = 156 #164   -- flipped to simulate pitch network on biaxial network

# expected input data shape: (batch_size or nb_samples, timesteps, inputdata_dim)
model = Sequential()
model.add(LSTM(output_dim, return_sequences=True, input_length=timesteps, input_dim=inputdata_dim))  # returns a sequence of vectors of dimension 32
model.add(LSTM(output_dim))  # return a single vector of dimension 32

# output layer with nb_classes classes
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# generate dummy training data
x_train = np.random.random((nb_samples, timesteps, inputdata_dim))
y_train = np.random.random((nb_samples, nb_classes))

print("size x_train: " + str(x_train.shape))
print("size y_train: " + str(y_train.shape))

# generate dummy validation data
x_val = np.random.random((nb_samples*10, timesteps, inputdata_dim))
y_val = np.random.random((nb_samples*10, nb_classes))

print("size x_val: " + str(x_val.shape))
print("size y_val: " + str(y_val.shape))
print("")

model.fit(x_train, y_train, nb_epoch=15, validation_data=(x_val, y_val))