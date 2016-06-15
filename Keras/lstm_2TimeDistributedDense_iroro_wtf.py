from keras.models import Sequential
from keras.layers.core import TimeDistributedDense, Activation, Dropout
from keras.layers.recurrent import GRU, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
import numpy as np
maxlen = 2

batch_size = 1
nb_word = 4
nb_tag = 2

X_train = [[1,2],[1,3]] #two sequences, one is [1,2] and another is [1,3]
# Y_train = [[[0,1],[1,0]],[[0,1],[1,0]]] #the output should be 3D and one-hot for softmax output with categorical_crossentropy
Y_train = [[[0,1],[1,0]],[[0,1],[1,0]]] #the output should be 3D and one-hot for softmax output with categorical_crossentropy

X_test = [[1,2],[1,3]]
Y_test = [[[0,1],[1,0]],[[0,1],[1,0]]]

X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

Y_train = np.asarray(Y_train, dtype='int32')
Y_test = np.asarray(Y_test, dtype='int32')
print(Y_train.shape)

model = Sequential()
# model.add(Embedding(nb_word, 128))
model.add(Embedding(nb_word, 128, input_length=2))
model.add(LSTM(128, return_sequences=True))
model.add(TimeDistributedDense(nb_tag))
model.add(Activation('softmax'))

rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms)
# model.compile(optimizer='rmsprop',loss='mse')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=100, show_accuracy=True)
res = model.predict_classes(X_test)
#res = model.predict(X_test)
print('res',res)