#%%
import time
import math
import sys
import argparse
import cPickle as pickle
import copy
import os
import six

import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
from VRAE import LSTMVRAE #, make_initial_state

import dataset

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',      type=str,   default="dataset")
parser.add_argument('--output_dir',     type=str,   default="model")
parser.add_argument('--dataset',        type=str,   default="midi")
parser.add_argument('--init_from',      type=str,   default="")
parser.add_argument('--clip_grads',     type=int,   default=5)
parser.add_argument('--gpu',            type=int,   default=-1)

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)


if args.dataset == 'midi':
    # midi = dataset.load_midi_data('%s/midi/sample1.mid' % args.data_path)
    midi = dataset.load_midi_data('%s/midi/Suteki-Da-Ne.mid' % args.data_path)
    # midi = dataset.load_midi_data('%s/midi/example.mid' % args.data_path)
    # midi = dataset.load_midi_data('%s/midi/haydn_7_1.mid' % args.data_path)
    train_x = midi[:120].astype(np.float32)

    n_x = train_x.shape[1]
    n_hidden = [500]
    n_z = 2
    n_y = n_x

    frames  = train_x.shape[0]
    n_batch = 6
    seq_length = frames / n_batch

    split_x = np.vsplit(train_x, n_batch)

    n_epochs = 500
    continuous = False

    print("seq_length: " + str(seq_length))
    print("train_x.shape: " + str(train_x.shape))


n_hidden_recog = n_hidden
n_hidden_gen   = n_hidden
n_layers_recog = len(n_hidden_recog)
n_layers_gen   = len(n_hidden_gen)

layers = {}

# # Recognition model.
# rec_layer_sizes = [(train_x.shape[1], n_hidden_recog[0])]
# rec_layer_sizes += zip(n_hidden_recog[:-1], n_hidden_recog[1:])
# rec_layer_sizes += [(n_hidden_recog[-1], n_z)]
#
# layers['recog_in_h'] = F.Linear(train_x.shape[1], n_hidden_recog[0], nobias=True)
# layers['recog_h_h']  = F.Linear(n_hidden_recog[0], n_hidden_recog[0])
#
# layers['recog_mean'] = F.Linear(n_hidden_recog[-1], n_z)
# layers['recog_log_sigma'] = F.Linear(n_hidden_recog[-1], n_z)
#
# # Generating model.
# gen_layer_sizes = [(n_z, n_hidden_gen[0])]
# gen_layer_sizes += zip(n_hidden_gen[:-1], n_hidden_gen[1:])
# gen_layer_sizes += [(n_hidden_gen[-1], train_x.shape[1])]
#
# layers['z'] = F.Linear(n_z, n_hidden_gen[0])
# layers['gen_in_h'] = F.Linear(train_x.shape[1], n_hidden_gen[0], nobias=True)
# layers['gen_h_h']  = F.Linear(n_hidden_gen[0], n_hidden_gen[0])
#
# layers['output']   = F.Linear(n_hidden_gen[-1], train_x.shape[1])

if args.init_from == "":
    print("Initializing model from code spec")
                   # n_input, n_hidden, n_latent, loss_func
    model = LSTMVRAE(train_x.shape[1], n_hidden_recog[0], n_z, F.mean_squared_error)
else:
    print("Initializing model from pickle file: " + args.init_from)
    model = pickle.load(open(args.init_from))

if args.gpu >= 0:
    cuda.init(args.gpu)
    model.to_gpu()


# use Adam
optimizer = optimizers.Adam()
optimizer.setup(model)

total_losses = np.zeros(n_epochs, dtype=np.float32)

for epoch in xrange(1, n_epochs + 1):
    print('epoch', epoch)

    t1 = time.time()
    total_rec_loss = 0.0
    total_kl_loss = 0.0
    total_loss = 0.0
    outputs = np.zeros(train_x.shape, dtype=np.float32)

    for i in xrange(n_batch):
        state = model.make_initial_state()
        x_batch = split_x[i]

        if args.gpu >= 0:
            x_batch = cuda.to_gpu(x_batch)

        # print("x_batch shape: " + str(x_batch.shape ))
        # x_batch.shape == (20, 88)
        output, rec_loss, kl_loss, state = model.forward(x_batch, state)

        outputs[i*seq_length:(i+1)*seq_length, :] = output.squeeze(1) # output but with removed single dimensions

        print("\n\nx_batch shape: " + str(x_batch.shape ))
        print("x_batch[0]: " + str(x_batch[0]))
        print("x_batch shape sum: " + str(sum(sum(x) for x in x_batch)))
        print("output shape: " + str(output.shape ))
        print("output[0]: " + str(output[0]))

        loss = rec_loss + kl_loss
        total_loss += loss
        total_rec_loss += rec_loss
        total_losses[epoch-1] = total_loss.data

        optimizer.zero_grads()
        loss.backward()
        loss.unchain_backward()
        optimizer.clip_grads(args.clip_grads)
        optimizer.update()

        # if epoch == 1 and i == 0:
        #     with open("graph.dot", "w") as o:
        #         o.write(c.build_computational_graph((loss, )).dump())

    saved_output = outputs

    print("{}/{}, train_loss = {}, total_rec_loss = {}, time = {}".format(epoch, n_epochs, total_loss.data,
                                                                          total_rec_loss.data, time.time()-t1))

    if epoch % 100 == 0:
        model_path = "%s/VRAE_%s_%d.pkl" % (args.output_dir, args.dataset, epoch)
        with open(model_path, "w") as f:
            pickle.dump(copy.deepcopy(model).to_cpu(), f)
