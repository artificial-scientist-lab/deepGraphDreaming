import argparse
import numpy as np
import pickle
import random
import time
import os
import yaml
from yaml import Loader
from pytheus import fancy_classes as fc, theseus as th, help_functions as hf

import torch

from datagen import generatorGraphFidelity, constructGraph
from neuralnet import prep_data, load_model, dream_model

# All the neuron sets below correspond to the trained neural network consisting of 15 hidden layers
neuron_index_sets = []
neuron_index_sets.append(np.arange(0, 30, 1))
neuron_index_sets.append(np.arange(0, 400, 2))
neuron_index_sets.append(np.arange(0, 200, 2))
neuron_index_sets.append( np.arange(0, 100, 2))
neuron_index_sets.append(np.arange(0, 60, 2))
neuron_index_sets.append(np.arange(0, 50, 2))
neuron_index_sets.append(np.arange(0, 30, 1))

stream = open("config_train.yaml", 'r')
cnfg = yaml.load(stream, Loader=Loader)

num_of_examples = cnfg['num_of_examples']  # training set size
learnRate = cnfg['learnRate']  # learning rate of inverse training
num_of_examples_fixed = num_of_examples
num_of_epochs = cnfg['num_of_epochs']  # for how many epochs should we run the inverse training?
layer_indices = cnfg['layer_indices']  # The indices corresponding to the hidden layers of the neural network
neuron_indices = neuron_index_sets[cnfg['neuron_index_set']]  # the neuron indices for each hidden layer
nnType = cnfg['nnType']  # the type of neural network we wish to examine

print(f"Let's a go! Number of examples: {num_of_examples}")
print(f"Learning rate: {learnRate}")

seed = random.randint(1000, 9999)
print(f'seed: {seed}')
cnfg['seed'] = seed
random.seed(cnfg['seed'])

kets = hf.makeState(cnfg['state'])
state = fc.State(kets, normalize=True)
dims = th.stateDimensions(state.kets)

# We generate a graph for the purposes of obtaining some additional properties about the graphs we are generating (e.g. we have 24 edge)
input_graph, ket_amplitudes, output_fidelity = generatorGraphFidelity(dims, state, num_edges=None,
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
direc = os.getcwd() + f'models/GraphDreamForward2_{num_of_examples}_4PartGHZ_small.pt'  # small indicates that we are going for the trained "small (5 layer, 30 neurons)" neural network
model_fidelity = load_model(direc, device, NN_INPUT, NN_OUTPUT, nnType)

# We proceed to generate an initial set of edges from the dreaming process. We sample 3 graphs from our dataset

rando = random.sample(range(0, len(res_test_np)), 3)
start_res = []
start_graphs = []
for r in rando:  # Construct the graphs associated to the edge weights we've chosen
    *_, randGraph = constructGraph(vals_test_np[r], dims, state)
    start_graphs.append(randGraph)

# Add the highest fidelity graph to the initial set 
*_, randGraph = constructGraph(vals_train_np[best_graph], dims, state)
start_graphs.append(randGraph)
start_res = np.append(res_test_np[rando], res_train_np[best_graph])

initial_prop_list = []
final_prop_list = []  # the fidelity of the final dreamed graphs
percent_valid_transforms = []  # number of vaild transformation steps taken during the dreaming process
start_time = time.time()

# We proceed dreaming on each graph in the data set
for i in range(0, len(start_res)):

    # the way it works is that we do dreaming on the graph using the trained neural network up to a specific layer and using the chosen neuron as the output neuron.
    # This is done by creating a new neural network with all the weights and biases of the original neural network up to and including the layer/neuron pair.
    # We also save how the graph looks in each step of the dreaming process as a png in a zip file. These are used to make the movies.
    for j in range(0, len(layer_indices)):
        for k in range(0, len(neuron_indices)):
            name_of_zip = f'Intermediate Graphs V2/zip_test_graph_{i}_{num_of_examples}_{layer_indices[j]}_{neuron_indices[k]}.zip'
            initial_prop_list.append(float(start_res[i]))
            final_prop, interm_graph, loss_prediction, interm_prop, nn_prop, gradDec, percent_valid_transform, *_ = dream_model(
                dims, model_fidelity, state, start_graphs[i], learnRate, num_of_epochs,
                name_of_zip, layer_indices[j], neuron_indices[k], display=True)
            final_prop_list.append(final_prop)
            percent_valid_transforms.append(percent_valid_transform)
            with open(
                    f'Dreamed Graph Pickles V2/dream_graph_{i}_{num_of_examples}_{layer_indices[j]}_{neuron_indices[k]}.pkl',
                    'wb') as f:
                pickle.dump([interm_graph, interm_prop, nn_prop, gradDec, loss_prediction], f)

# You can uncomment this if you want to plot the histograph of dreamed graph fidelities. 
'''
plt.hist(final_prop_list, 25, alpha=0.5, label = 'Dreamed Fidelity')
plt.hist(initial_prop_list, 25, alpha=0.5, label = 'Initial Fidelity')
plt.legend(loc='upper right')
plt.savefig('final_prop_dist_' + str(num_of_examples) + '.png')
'''

print("Wake up, sleepyhead!")
print("--- %s seconds ---" % (time.time() - start_time))
