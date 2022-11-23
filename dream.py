import argparse
import numpy as np
import pickle
import random
import time
import os
import yaml
from yaml import Loader
from pytheus import fancy_classes as fc, theseus as th, help_functions as hf
import csv

import torch

from datagen import generatorGraphFidelity, constructGraph
from neuralnet import prep_data, load_model, dream_model

# All the neuron sets below correspond to the trained neural network consisting of 15 hidden layers
neuron_index_sets = []
neuron_index_sets.append(np.arange(0, 30, 1))
neuron_index_sets.append(np.arange(0, 400, 2))
neuron_index_sets.append(np.arange(0, 200, 2))
neuron_index_sets.append(np.arange(0, 100, 2))
neuron_index_sets.append(np.arange(0, 60, 2))
neuron_index_sets.append(np.arange(0, 50, 2))
neuron_index_sets.append(np.arange(0, 30, 1))

stream = open("config_dream.yaml", 'r')
cnfg = yaml.load(stream, Loader=Loader)

num_of_examples = cnfg['num_of_examples']  # training set size
learnRate = cnfg['learnRate']  # learning rate of inverse training
num_of_examples_fixed = num_of_examples
num_of_epochs = cnfg['num_of_epochs']  # for how many epochs should we run the inverse training?
layer_indices = cnfg['layer_indices']  # The indices corresponding to the hidden layers of the neural network
neuron_indices = neuron_index_sets[cnfg['neuron_index_set']]  # the neuron indices for each hidden layer

layer = cnfg['layer']  # The indices corresponding to the hidden layers of the neural network
neuron = cnfg['neuron']  # the neuron indices for each hidden layer

nnType = cnfg['nnType']  # the type of neural network we wish to examine
modelname = cnfg['modelname']

print(f"Let's a go! Number of examples: {num_of_examples}")
print(f"Learning rate: {learnRate}")

seed = random.randint(1000, 9999)
print(f'seed: {seed}')
cnfg['seed'] = seed
random.seed(cnfg['seed'])

kets = hf.makeState(cnfg['state'])
state = fc.State(kets, normalize=True)
cnfg['dims'] = th.stateDimensions(state.kets)

# We generate a graph for the purposes of obtaining some additional properties about the graphs we are generating (e.g. we have 24 edge)
input_graph, ket_amplitudes, output_fidelity = generatorGraphFidelity(cnfg['dims'], state, num_edges=None,
                                                                      short_output=False)

# Load up a dataset of generated graphs
with open(f'data_train.pkl', 'rb') as f:
    data_full, res_full = pickle.load(f)

data = data_full[0:num_of_examples_fixed]
res = res_full[0:num_of_examples_fixed]
vals_train_np, vals_test_np, res_train_np, res_test_np = prep_data(data, res, 0.95)
best_graph = np.argmax(res_train_np)  # Index pertaining to the graph with the highest fidelity in the dataset

NN_INPUT = len(input_graph.weights)
NN_OUTPUT = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load up our trained neural network
direc = os.getcwd() + f'/models/{modelname}'
model = load_model(direc, device, NN_INPUT, NN_OUTPUT, nnType)

# We proceed to generate an initial set of edges from the dreaming process. We sample 3 graphs from our dataset

if cnfg['start_graph'] == 'best':
    ind = best_graph
else:
    ind = random.randint(0, len(res_test_np))

*_, start_graph = constructGraph(vals_train_np[ind], cnfg['dims'], state)
start_res = float(res_train_np[ind])
start_pred = model(torch.tensor(input_graph.weights, dtype=torch.float))
with open(cnfg['dream_file'], 'a') as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow([start_res, start_pred, start_graph.weights])

final_prop_list = []  # the fidelity of the final dreamed graphs
start_time = time.time()

name_of_zip = f'intermediategraphs/zip_test_graph.zip'
dream_model(model, state, start_graph, cnfg)

'''
plt.hist(final_prop_list, 25, alpha=0.5, label = 'Dreamed Fidelity')
plt.hist(initial_prop_list, 25, alpha=0.5, label = 'Initial Fidelity')
plt.legend(loc='upper right')
plt.savefig('final_prop_dist_' + str(num_of_examples) + '.png')
'''

print("Wake up, sleepyhead!")
print("--- %s seconds ---" % (time.time() - start_time))
