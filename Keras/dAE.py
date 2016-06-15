from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence, text
from keras.utils.np_utils import accuracy
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Graph, Sequential, model_from_yaml
from keras.layers.core import Dense, Dropout, Activation, AutoEncoder
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import rmsprop
from keras.layers import containers
from keras.constraints import unitnorm
from keras.regularizers import WeightRegularizer, ActivityRegularizer
from keras.optimizers import *
import string
import pandas as pd
import datetime
import json
import re
from collections import defaultdict
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Convolution1D, MaxPooling1D

current_model = datetime.datetime.now().strftime("dAE.%Y%m%dT%H")
DICTIONARY = "/site/aspen/common/sentiment/keras_sentiment/data/new_dictionary_20160204T14.json"
exclude = re.escape(re.sub(r"[\-\_\']", "", string.punctuation))
batch_size = 128
max_features = 5003
# maxlen = 45  # cut texts after this number of words (among top max_features most common words)
def default_value():
    return 1

words2int = defaultdict(default_value, json.loads(open(DICTIONARY).read()))


def get_file(file_path):
    data = open(file_path).read().strip().split("\n")
    try:
        dd = [z for x in data for z in json.loads(x)['caller'] if z != 'no utterances']
    except:
        return None
    df = pd.DataFrame(dd)
    df['transcript'] = df.transcript.astype(str).str.lower() \
        .str.replace(r"[%s]" % exclude, "").str.replace("[_-]", ' ')
    v = CountVectorizer(token_pattern='(?u)[^ ]+', vocabulary=words2int.keys())
    x = np.log1p(v.transform(df.transcript.values)).toarray()
    return x

def batch_generator(batch_size=128):
    dir = "/site/aspen/common/sentiment/keras_sentiment/data/predicted_sentiment/"
    files = os.listdir(dir)[:20]
    numfiles = len(files)
    i = 0
    data_pool = get_file(dir+files[i])
    while True:
        if data_pool.shape[0] <= batch_size:
            while data_pool.shape[0] <= batch_size:
                i += 1
                ix = i % numfiles
                # print(files[ix])
                new_data = get_file(dir+files[ix])
                if new_data is not None:
                    data_pool = np.vstack((data_pool, new_data))
        out = data_pool[:batch_size, :].copy()
        # print(out.sum(axis=1))
        data_pool = data_pool[batch_size:, :]
        yield out, out


# Pre-defined vocabulary dict you can create one with sentiment.preprocessing.clean_dictionary.py
# The Kaldi Dictionary would be even better, but it's an order of magnitude bigger and will blow up the input layer

print("Loading data...")

encoder = containers.Sequential([Dense(2000, input_dim=5000),
                                 LeakyReLU(0.2),
                                 Dense(500, activation='relu'),
                                 LeakyReLU(0.2),
                                 Dense(250, activation='relu'),
                                 LeakyReLU(0.2),
                                 Dense(10, activation='relu')])
decoder = containers.Sequential([Dense(250, activation='relu', input_dim=10),
                                 LeakyReLU(0.2),
                                 Dense(500, activation='relu', ),
                                 LeakyReLU(0.2),
                                 Dense(2000, activation='relu', ),
                                 LeakyReLU(0.2),
                                 Dense(5000, activation='relu', )])

ae = AutoEncoder(encoder=encoder, decoder=decoder)

model = Sequential()
model.add(ae)
model.layers[0].build()
model.compile(optimizer=rmsprop(), loss='mse')
dir = "/site/aspen/common/sentiment/keras_sentiment/data/predicted_sentiment/"
files = os.listdir(dir)
X = np.vstack([get_file(dir+files[i]) for i in range(25, 30) if get_file(dir+files[i]) is not None])

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

checkpoint = ModelCheckpoint("weights.{epoch:02d}-{val_loss:.3f}.hdf5", monitor='val_loss', verbose=0,
                             save_best_only=True, mode='auto')
print("Train...", datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),"\n")

try:
    model.fit_generator(batch_generator(), 
        samples_per_epoch=10000000, 
        validation_data=(X,X), 
        nb_epoch=10,
        callbacks=[checkpoint, early_stopping])

except KeyboardInterrupt:
    pass

model.layers[0].output_reconstruction = False  # the autoencoder has to be recompiled after modifying this property
model.layers[0].build()

model.compile(optimizer=rmsprop(), loss='mse')
# representations = autoencoder.predict(X_test)



def toDF(fname):
    d = [json.loads(call) for call in open(fname).read().strip().split("\n")]
    index = [(x['product_id'], x['product']) for x in d]
    # agent = pd.concat([pd.DataFrame(row['agent']) for row in d], keys=index).reset_index().rename(columns={"level_0":'product_id', 'level_1':"product"})
    caller = pd.concat([pd.DataFrame(row['caller']) for row in d], keys=index).reset_index().rename(columns={
        "level_0": 'product_id', 'level_1':"product"}).drop(0, axis=1)
    # df = pd.concat([agent, caller], keys=['agent','caller']).reset_index().rename(columns={"level_0":'channel'}).drop(0, axis=1)
    return caller.ix[caller.transcript.notnull()]


X = np.vstack([get_file(dir+files[i]) for i in range(10)])
code = model.predict(X)
df = pd.concat([toDF(dir+files[i]) for i in range(10)])
print(df.head())
df.join(pd.DataFrame(code, index=df.index)).to_csv(datetime.datetime.now().strftime("coded_data_%Y%m%d_%H.csv"))