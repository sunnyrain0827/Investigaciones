import os
import time
import numpy as np
import argparse
import cPickle as pickle
from scipy import misc

from chainer import cuda, Variable, function, FunctionSet, optimizers
from chainer import functions as F

parser = argparse.ArgumentParser()
parser.add_argument('--model',          type=str,   required=True)
parser.add_argument('--output_dir',     type=str,   default="generated")
parser.add_argument('--dataset',        type=str,   default="mnist")
parser.add_argument('--gpu',            type=int,   default=-1)

args = parser.parse_args()

# if args.dataset == 'faces':
#     im_size = (28,20)
#
# elif args.dataset == 'mnist':
#     im_size = (28, 28)

model = pickle.load(open(args.model, "rb"))

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

n_sample = 100
n_z = model.recog_log_sigma.W.data.shape[0]
print("n_z: " + str(n_z))

# sampleN = np.random.standard_normal((100, n_z)).astype(np.float32)
# n_layers_gen = 2
# def generate    (latent_data, n_layers_gen, nonlinear_p='relu'):
# generated_output = model.generate(sampleN, n_layers_gen, 'relu')

seq_length_per_z = 20 # from training - x_data.shape[0]
sample_z = np.random.standard_normal((100, n_z)).astype(np.float32)

# def generate_z_x(seq_length_per_z, sample_z, nonlinear_q='tanh', nonlinear_p='tanh', output_f='sigmoid', gpu=-1):
generated_output = model.generate_z_x(seq_length_per_z, sample_z)

print("generated_output.data.shape: " + str(generated_output.shape))

# save off samples as images -- we want something different for MIDI
# for i in range(n_sample):
#     im = np.ones(im_size).astype(np.float32) - generated_output.data[i].reshape(im_size)
#     misc.imsave('%s/%d.jpg'% (args.output_dir, i), im)