import argparse
import numpy as np
import pickle
import random
import time
import os

import torch

from datagen import generatorGraphFidelity, constructGraph
from neuralnet import prep_data, load_model, dream_model

parser = argparse.ArgumentParser(description='generating lots of graphs')
parser.add_argument('--ii', dest='ii', type=int,
                    default=None, help='')
args = parser.parse_args()
shift = args.ii
print(shift)

# All the neuron sets below correspond to the trained neural network consisting of 15 hidden layers (details can be found in the slides)
# For the sake of time, we look at every other neuron
neuron_index_set_2 = np.arange(0, 400, 2)
neuron_index_set_3 = np.arange(0, 200, 2)
neuron_index_set_4 = np.arange(0, 100, 2)
neuron_index_set_5 = np.arange(0, 60, 2)
neuron_index_set_6 = np.arange(0, 50, 2)

# This is for a smaller neural network consisting of 5 hidden layers and 30 neurons in each layer. 
neuron_index_set_1 = np.arange(0, 30, 1)

# This is to test if we can also probe the behavior of the input layer
neuron_index_set_7 = np.array([0])

# We do the deep dreaming for section(s) (that is, the set of hidden layers in the network with the same number of neurons) of the neural network in parallel. These are the parameters that we choose for each
parameter_list = [[10000000, 1 * 10 ** -4, 30000, [2, 4, 6, 8], neuron_index_set_1, 1],
                  [10000000, 1 * 10 ** -4, 30000, [2], neuron_index_set_2, 2],
                  [10000000, 1 * 10 ** -4, 30000, [4], neuron_index_set_3, 2],
                  [10000000, 1 * 10 ** -4, 30000, [6], neuron_index_set_4, 2],
                  [10000000, 1 * 10 ** -4, 30000, [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28], neuron_index_set_5, 2],
                  [10000000, 1 * 10 ** -4, 30000, [30], neuron_index_set_6, 2],
                  [10000000, 1 * 10 ** -4, 10000, [32], neuron_index_set_7, 2]]

num_of_examples = parameter_list[shift - 1][0]  # training set size
learnRate = parameter_list[shift - 1][1]  # learning rate of inverse training
num_of_examples_fixed = parameter_list[3][0]
num_of_epochs = parameter_list[shift - 1][2]  # for how many epochs should we run the inverse training?
layer_indices = parameter_list[shift - 1][
    3]  # The indices corresponding to the hidden layers of the neural network (the odd layers are the nonlinear activation function layers)
neuron_indices = parameter_list[shift - 1][4]  # the neuron indices for each hidden layer
nnType = parameter_list[shift - 1][
    5]  # the type of neural network we wish to examine (the big one we looked at before, or the way smaller neural network)

print(f"Let's a go! Number of examples: {num_of_examples}")
print(f"Learning rate: {learnRate}")

random.seed(666)  # Why around 42 specifically? Cheeky joke? Apparently, 666 is another seed that works.

# We compute the fidelity of the final state of each quantum graph with respect to the GHZ state. 
desired_state_2 = {
    ((0, 0), (1, 0), (2, 0), (3, 0)): (1 / np.sqrt(2)),
    ((0, 1), (1, 1), (2, 1), (3, 1)): (1 / np.sqrt(2))
}
# [dimension of each graph vertex] * number of vertices in the graph
DIM = [2] * 4

# We generate a graph for the purposes of obtaining some additional properties about the graphs we are generating (e.g. we have 24 edge)
input_edges, ket_amplitudes, output_fidelity = generatorGraphFidelity(DIM, desired_state_2, num_edges=None,
                                                                      short_output=False)
input_edge_weights = np.array(list(input_edges.values()))

# Load up a dataset of generated graphs 

with open(f'graph_simple_fidelity_{num_of_examples_fixed}.pkl', 'rb') as f:
    data_full, res_full = pickle.load(f)

data = data_full[0:num_of_examples_fixed]
res = res_full[0:num_of_examples_fixed]
vals_train_np, vals_test_np, res_train_np, res_test_np = prep_data(data, res, 0.95)
best_graph = np.argmax(res_train_np)  # Index pertaining to the graph with the highest fidelity in the dataset

NN_INPUT = len(input_edge_weights)
NN_OUTPUT = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load up our trained neural network
direc = os.getcwd() + f'models/GraphDreamForward2_{num_of_examples}_4PartGHZ_small.pt'  # small indicates that we are going for the trained "small (5 layer, 30 neurons)" neural network
model_fidelity = load_model(direc, device, NN_INPUT, NN_OUTPUT, num_of_examples, nnType)

# We proceed to generate a initial set of edges from the dreaming process. We sample 3 graphs from our dataset

rando = random.sample(range(0, len(res_test_np)), 3)
prop_test = []
graph_test = []
for r in rando:  # Construct the graphs associated to the edge weights we've chosen
    *_, randGraph = constructGraph(vals_test_np[r], DIM, desired_state_2)
    graph_test.append(randGraph)

# Add the highest fidelity graph to the initial set 
*_, randGraph = constructGraph(vals_train_np[best_graph], DIM, desired_state_2)
graph_test.append(randGraph)
prop_test = np.append(res_test_np[rando], res_train_np[best_graph])

'''
# Out of curiousity, let's see what happens when we start from a perfect GHZ state. 

GHZ_PLUS = [0]*24
GHZ_PLUS[3] = 1
GHZ_PLUS[8] = 1
GHZ_PLUS[12] = 1
GHZ_PLUS[23] = 1

GHZ_MINUS = [0]*24
GHZ_MINUS[3] = -1
GHZ_MINUS[8] = 1
GHZ_MINUS[12] = 1
GHZ_MINUS[23] = 1

fidelity1, randGraph1 = constructGraph(GHZ_PLUS, DIM, desired_state_2)
prop_test = np.append(prop_test,fidelity1)
graph_test.append(randGraph1)

fidelity2, randGraph2 = constructGraph(GHZ_MINUS, DIM, desired_state_2)
prop_test = np.append(prop_test,fidelity2)
graph_test.append(randGraph2)
'''

initial_prop_list = []
final_prop_list = []  # the fidelity of the final dreamed graphs
percent_valid_transforms = []  # number of vaild transformation steps taken during the dreaming process
start_time = time.time()

# We proceed dreaming on each graph in the data set
for i in range(0, len(prop_test)):

    # the way it works is that we do dreaming on the graph using the trained neural network up to a specific layer and using the chosen neuron as the output neuron.
    # This is done by creating a new neural network with all the weights and biases of the original neural network up to and including the layer/neuron pair.
    # We also save how the graph looks in each step of the dreaming process as a png in a zip file. These are used to make the movies.
    for j in range(0, len(layer_indices)):
        for k in range(0, len(neuron_indices)):
            name_of_zip = f'Intermediate Graphs V2/{shift}/zip_test_graph_{i}_{num_of_examples}_{layer_indices[j]}_{neuron_indices[k]}.zip'
            initial_prop_list.append(float(prop_test[i]))
            final_prop, interm_graph, loss_prediction, interm_prop, nn_prop, gradDec, percent_valid_transform, *_ = dream_model(
                DIM, model_fidelity, num_of_examples, desired_state_2, graph_test[i], learnRate, num_of_epochs,
                name_of_zip, layer_indices[j], neuron_indices[k], display=True)
            final_prop_list.append(final_prop)
            percent_valid_transforms.append(percent_valid_transform)
            with open(
                    f'Dreamed Graph Pickles V2/{shift}/dream_graph_{i}_{num_of_examples}_{layer_indices[j]}_{neuron_indices[k]}.pkl',
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
