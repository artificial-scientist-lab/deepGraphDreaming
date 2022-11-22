# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 16:47:54 2022

@author: freem
"""

import argparse
import numpy as np
import pickle
import random
import os
from pytheus import help_functions as hf, fancy_classes as fc, theseus as th
import yaml
from yaml import Loader

from datagen import generatorGraphFidelity
from neuralnet import prep_data, train_model

# We compute the fidelity of the final state of each quantum graph with respect to the GHZ state.
stream = open("config_train.yaml", 'r')
cnfg = yaml.load(stream, Loader=Loader)

kets = hf.makeState(cnfg['state'])
state = fc.State(kets, normalize=True)
dims = th.stateDimensions(state.kets)

num_of_examples = int(float(cnfg['num_of_examples']))  # Training set size
learnRate = float(cnfg['learnRate'])  # Learning rate
model_prefix = cnfg['model_prefix']  # when we save the neural network as a .pt, this is the name that it inherits
l2Lambda = float(cnfg['l2Lambda'])  # Lambda parameter for L2 Regularization
isL2Reg = float(cnfg['isL2Reg'])  # Do we want to introduce L2 Regularization in the training process?
nnType = cnfg['nnType']  # What type of neural network do we want to train on

print(f"Let's a go! Number of examples: {num_of_examples}")
print(f"Learning Rate: {learnRate}")

seed = random.randint(1000, 9999)
print(f'seed: {seed}')
cnfg['seed'] = seed
random.seed(cnfg['seed'])

# Generate a sample graph to extract additional properties (like the number of edges for our chosen graph shape)
input_edges, ket_amplitudes, output_fidelity = generatorGraphFidelity(dims, state, num_edges=None,
                                                                      short_output=False)
input_edge_weights = input_edges.weights

# Load up the training dataset
with open('data_train.pkl', 'rb') as f:
    data_full, res_full = pickle.load(f)

data = data_full[0:num_of_examples]
res = res_full[0:num_of_examples]

weights_train, weights_test, result_train, result_test = prep_data(data, res, 0.95)
NN_INPUT = len(input_edge_weights)
NN_OUTPUT = 1

# Prepare saving the model
direc = os.getcwd() + f'/models/{model_prefix}_seed{seed}.pt'
print(direc, flush=True)

# train the model
stream = open(f'models/config{seed}.yaml', 'w')
yaml.dump(cnfg, stream)

train_model(NN_INPUT, NN_OUTPUT, weights_train, result_train, weights_test, result_test, learnRate, direc, model_prefix,
            num_of_examples, nnType, cnfg['batch_size'], l2Lambda, isL2Reg)
