import itertools
import numpy as np

sentences = '''
sam is red
hannah not red
hannah is green
bob is green
bob not red
sam not green
sarah is red
sarah not green'''.strip().split('\n')
is_green = np.asarray([[0, 1, 1, 1, 1, 0, 0, 0]], dtype='int32').T

lemma = lambda x: x.strip().lower().split(' ')
sentences_lemmatized = [lemma(sentence) for sentence in sentences]
words = set(itertools.chain(*sentences_lemmatized))
# set(['boy', 'fed', 'ate', 'cat', 'kicked', 'hat'])

# dictionaries for converting words to integers and vice versa
word2idx = dict((v, i) for i, v in enumerate(words))
idx2word = list(words)

# convert the sentences a numpy array
to_idx = lambda x: [word2idx[word] for word in x]
sentences_idx = [to_idx(sentence) for sentence in sentences_lemmatized]
sentences_array = np.asarray(sentences_idx, dtype='int32')
print("Sentences array, i.e. integerized words (word2idx'd) : ")
print(sentences_array)
print("\n")

# parameters for the model
sentence_maxlen = 3
n_words = len(words)
n_embed_dims = 3

# put together a model to predict
from keras.layers import Input, Embedding, SimpleRNN, Dense, LSTM
from keras.models import Model

input_sentence = Input(shape=(sentence_maxlen,), dtype='int32')
input_embedding = Embedding(n_words, n_embed_dims)(input_sentence)
color_prediction = LSTM(3)(input_embedding)
output = Dense(1, activation='sigmoid')(color_prediction)

predict_green = Model(input=[input_sentence], output=[output])
predict_green.compile(optimizer='sgd', loss='binary_crossentropy')

# fit the model to predict what color each person is
predict_green.fit([sentences_array], [is_green], nb_epoch=15000, verbose=1)
embeddings = predict_green.layers[1].W.get_value()


##############################################################################
# print out the embedding vector associated with each word

xs = []
ys = []
zs = []

for i in range(n_words):
    print('{}: {}'.format(idx2word[i], embeddings[i]))
    xs.append(embeddings[i][0])
    ys.append(embeddings[i][1])
    zs.append(embeddings[i][2])

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plotlabels = []

# these are okay if we didn't have to link the labels
# xs,ys,zs = zip(*embeddings)
# xs, ys, zs = np.split(points, 3, axis=1)
sc = ax.scatter(xs, ys, zs)

def update_position(e,fig,ax,labels_and_points):
    for label, x, y, z in labels_and_points:
        x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
        label.xy = x2,y2
        label.update_positions(fig.canvas.renderer)
    fig.canvas.draw()

for txt, x, y, z in zip(idx2word, xs, ys, zs):
    x2, y2, _ = proj3d.proj_transform(x,y,z, ax.get_proj())
    label = plt.annotate(
        txt, xy = (x2, y2), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    plotlabels.append(label)
fig.canvas.mpl_connect('button_release_event', lambda event: update_position(event,fig,ax,zip(plotlabels, xs, ys, zs)))
plt.show()




