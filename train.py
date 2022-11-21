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

from datagen import generatorGraphFidelity
from neuralnet import prep_data, train_model

# We compute the fidelity of the final state of each quantum graph with respect to the GHZ state. 
desired_state_2 = {((0, 0), (1, 0), (2, 0), (3, 0)): (1 / np.sqrt(2)),
                   ((0, 1), (1, 1), (2, 1), (3, 1)): (1 / np.sqrt(2))}

parser = argparse.ArgumentParser(description='generating lots of graphs')
parser.add_argument('--ii', dest='ii', type=int,
                    default=None, help='')
args = parser.parse_args()
shift = args.ii
print(shift)

# We can train different kinds of neural network in parallel
parameter_list = [[10000000, 1 * 10 ** -3, '.pt', '_4PartGHZ', desired_state_2, [2] * 4, 0.00001, False, 1, True],
                  [10000000, 1 * 10 ** -3, 'ZERO.pt', '_4PartGHZ', desired_state_2, [2] * 4, 0.00001, True, 1, True]]

num_of_examples = parameter_list[shift - 1][0]  # Training set size
learnRate = parameter_list[shift - 1][1]  # Learning rate
end_of_string = parameter_list[shift - 1][2]  # file extension for the dataset of graphs we need to train on
nn_case = parameter_list[shift - 1][3]  # when we save the neural network as a .pt, this is the name that it inherits
desired_state = parameter_list[shift - 1][4]  # with respect to what state we compute the fidelity?
DIM = parameter_list[shift - 1][5]  # shape  of graph
l2Lambda = parameter_list[shift - 1][6]  # Lambda parameter for L2 Regularization
isL2Reg = parameter_list[shift - 1][7]  # Do we want to introduce L2 Regularization in the training process?
nnType = parameter_list[shift - 1][8]  # What type of neural network do we want to train on
isZero = parameter_list[shift - 1][
    9]  # Sets the fidelities in the training set to zero if true. Useful if we want to set a baseline for

print(f"Let's a go! Number of examples: {num_of_examples}")
print(f"Learning Rate: {learnRate}")

random.seed(666)  # Why around 42 specifically? Cheeky joke? Apparently, 666 is another seed that works.

# Generate a sample graph to extract additional properities (like the number of edges for our chosen graph shape)
input_edges, ket_amplitudes, output_fidelity = generatorGraphFidelity(DIM, desired_state, num_edges=None,
                                                                      short_output=False)
input_edge_weights = np.array(list(input_edges.values()))

# Load up the training dataset
with open(f'graph_simple_fidelity_{num_of_examples}.pkl', 'rb') as f:
    data_full, res_full = pickle.load(f)

data = data_full[0:num_of_examples]
res = res_full[0:num_of_examples]

# Prepare the data for training. Zero out the output fidelities in the training set if isZero is true
vals_train_np, vals_test_np, res_train_np, res_test_np = prep_data(num_of_examples, data, res, 0.95)
if (isZero):
    vals_train_np = np.zeros(vals_train_np.shape)
    vals_test_np = np.zeros(vals_test_np.shape)
NN_INPUT = len(input_edge_weights)
NN_OUTPUT = 1

# Prepare saving the model
direc = os.getcwd() + f'/GraphDreamForward2_{num_of_examples}' + nn_case + end_of_string
print(os.getcwd() + f'/GraphDreamForward2_{num_of_examples}' + nn_case + end_of_string)

# train the model
train_model(NN_INPUT, NN_OUTPUT, vals_train_np, res_train_np, vals_test_np, res_test_np, learnRate, direc, nn_case,
            num_of_examples, nnType, l2Lambda, isL2Reg)
