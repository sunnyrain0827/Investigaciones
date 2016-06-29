from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats


# "encoded" is the encoded representation of the input
# encoded = Dense(encoding_dim, activation='relu')(input_img)
# encoded = Dense(encoding_dim, activation='relu',
#                 activity_regularizer=regularizers.activity_l1(10e-5))(input_img)

# "decoded" is the lossy reconstruction of the input
# decoded = Dense(784, activation='sigmoid')(encoded)

# this is our input placeholder
input_img = Input(shape=(784,))

encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)   # this is no longer the top and bottom of the decoder (takes 128->784)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                nb_epoch=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

encoder = Model(input=input_img, output=encoded)
autoencoder.summary()
encoder.summary()

# for the shallow autoencoder

# encoded_input = Input(shape=(encoding_dim,))
# decoder = Model(input=encoded_input, output=decoded)

# decoder_layer = autoencoder.layers[-1]
# decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

# encode and decode some digits
# note that we take them from the *test* set
# encoded_imgs = encoder.predict(x_test)
# decoded_imgs = decoder.predict(encoded_imgs)

# for the deep autoencoder
decoded_imgs = autoencoder.predict(x_test)


# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()