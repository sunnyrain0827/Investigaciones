from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model
import numpy as np

input_dim = 156
timesteps = 20
latent_dim = 32
nb_samples = 164

inputs = Input(shape=(timesteps, input_dim))
encoded = LSTM(latent_dim)(inputs)

decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(input_dim, return_sequences=True)(decoded)

sequence_autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)

# generate dummy training data
x_train = np.random.random((nb_samples, timesteps, input_dim))
x_test = np.random.random((nb_samples, timesteps, input_dim))

sequence_autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

sequence_autoencoder.fit(x_train, x_train,
                         nb_epoch=100,
                         batch_size=256,
                         shuffle=True,
                         validation_data=(x_test, x_test))