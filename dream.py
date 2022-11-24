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
import pandas as pd
import torch

from datagen import generatorGraphFidelity, constructGraph
from neuralnet import prep_data, load_model, dream_model

stream = open("config_dream.yaml", 'r')
cnfg = yaml.load(stream, Loader=Loader)

learnRate = cnfg['learnRate']  # learning rate of inverse training
num_of_epochs = cnfg['num_of_epochs']  # for how many epochs should we run the inverse training?
nnType = cnfg['nnType']  # the type of neural network we wish to examine
modelname = cnfg['modelname']
num_start_graphs = cnfg['num_start_graphs'] if cnfg['start_graph'] == 'best' else 1

# seed = random.randint(1000, 9999)
# print(f'seed: {seed}')
# cnfg['seed'] = seed
seed = cnfg['seed']
random.seed(seed)

# load data
if cnfg['datafile'].split('.')[-1] == 'pkl':
    # Load up the training dataset
    with open(cnfg['datafile'], 'rb') as f:
        data_full, res_full = pickle.load(f)

    data = data_full[:]
    res = res_full[:]
else:
    df = pd.read_csv(cnfg['datafile'], names=['weights', 'res'], delimiter=";")
    data = np.array([eval(graph) for graph in df['weights']])
    res = df['res'].to_numpy()
vals_train_np, vals_test_np, res_train_np, res_test_np = prep_data(data, res, 0.95)
best_graph = np.argmax(res_train_np)  # Index pertaining to the graph with the highest fidelity in the dataset
randinds = []
for ii in range(num_start_graphs):
    randinds.append(random.randint(0, len(res_train_np)))

# parse through slurm array
parser = argparse.ArgumentParser()
parser.add_argument(dest='ii')
args = parser.parse_args()
proc_id = args.ii

# choose start graph
start_graph_id = proc_id % num_start_graphs
if cnfg['start_graph'] == 'best':
    ind = best_graph
else:
    ind = randinds[start_graph_id]

# choose neuron from array given in config
neuron_id = proc_id // num_start_graphs
neuron_array = eval(cnfg['neuron_array'])
cnfg['layer'], cnfg['neuron'] = neuron_array[neuron_id]

cnfg['dream_file'] += f'{seed}/{start_graph_id}_{neuron_id}.csv'
print(cnfg['dream_file'])

kets = hf.makeState(cnfg['state'])
state = fc.State(kets, normalize=True)
cnfg['dims'] = th.stateDimensions(state.kets)

# We generate a graph for the purposes of obtaining some additional properties about the graphs we are generating (e.g. we have 24 edge)
input_graph, ket_amplitudes, output_fidelity = generatorGraphFidelity(cnfg['dims'], state, num_edges=None,
                                                                      short_output=False)
NN_INPUT = len(input_graph.weights)
NN_OUTPUT = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load up our trained neural network
direc = os.getcwd() + f'/models/{modelname}'
model = load_model(direc, device, NN_INPUT, NN_OUTPUT, nnType)

# We proceed to generate an initial set of edges from the dreaming process. We sample 3 graphs from our dataset


*_, start_graph = constructGraph(vals_train_np[ind], cnfg['dims'], state)
start_res = float(res_train_np[ind])
start_pred = model(torch.tensor(input_graph.weights, dtype=torch.float).to(device)).item()
with open(cnfg['dream_file'], 'a') as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow([start_res, start_pred, start_graph.weights])

start_time = time.time()
dream_model(model, state, start_graph, cnfg)

print(f"--- done in {time.time() - start_time} seconds ---")
