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
import re

from datagen import generatorGraphFidelity, constructGraph
from neuralnet import prep_data, load_model, dream_model, neuron_selector

def maxNElems(listor, N):
    final_max = []
    tempList = listor
    
    for i in range(0,N):
        maxTemp = 0
        maxIndex = 0
        for j in range(len(tempList)):
            if tempList[j] > maxTemp:
                maxTemp = tempList[j]
                maxIndex = j
                
        tempList[maxIndex] = 0
        final_max.append(maxIndex)
            
            
    return final_max

stream = open("configs/dream.yaml", 'r')
cnfg = yaml.load(stream, Loader=Loader)

learnRate = cnfg['learnRate']  # learning rate of inverse training
num_of_epochs = cnfg['num_of_epochs']  # for how many epochs should we run the inverse training?
nnType = cnfg['nnType']  # the type of neural network we wish to examine
modelname = cnfg['modelname']
num_start_graphs = cnfg['num_start_graphs'] if cnfg['start_graph'] == 'random' else 1

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
    df = pd.read_csv(cnfg['datafile'], names=['weights', 'res'], delimiter=";", nrows=cnfg['num_of_examples_fixed'])
    try:
        data = np.array([eval(graph) for graph in df['weights']])
    except:
        data = np.array(
            [eval(re.sub(r"  *", ',', graph.replace('\n', '').replace('[ ', '['))) for graph in df['weights']])
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
proc_id = int(args.ii)

print("spiffy" )
# choose start graph
start_graph_id = proc_id % num_start_graphs
if cnfg['start_graph'] == 'best':
    ind = best_graph
else:
    ind = randinds[start_graph_id]

# choose neuron from array given in config
neuron_id = proc_id // num_start_graphs
neuron_array = eval(cnfg['neuron_array'])
neuron_id = neuron_id % len(neuron_array)
cnfg['layer'], cnfg['neuron'] = neuron_array[neuron_id]

cnfg['dream_file'] += f"_layer{cnfg['layer']}"
cnfg['dream_file'] += f'_{seed}'
dreamfolder = cnfg['dream_file']
cnfg['dream_file'] += f'/dream{start_graph_id}_{neuron_id}.csv'
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

# Here, we look over our training examples and choose the one
# which activates the neuron the most.  

startPred = np.zeros(len(vals_train_np))

if (cnfg['bestExamples']):
    intermediateModel = neuron_selector(model,device, cnfg['layer'],cnfg['neuron'])
    for ii in range(len(vals_train_np)):
        fid, temp_graph = constructGraph(vals_train_np[ii], cnfg['dims'], state)
        # Evaluate starting prediction 
        startPred[ii] = intermediateModel(torch.tensor(temp_graph.weights, dtype=torch.float).to(device))

    # If best examples is enabled, we choose the graph that triggers the maximum activation on the neuron. 
    bestInds = maxNElems(startPred,num_start_graphs) 
    ind = bestInds[start_graph_id]
    print(ind)
    print(bestInds) 
    print(start_graph_id)

# We proceed to generate an initial set of edges from the dreaming process. We sample 3 graphs from our dataset

if cnfg['start_graph'] == 'zero':
    fid, start_graph = constructGraph([0] * len(input_graph), cnfg['dims'], state)
else:
    fid, start_graph = constructGraph(vals_train_np[ind], cnfg['dims'], state)
start_res = fid
start_pred = model(torch.tensor(start_graph.weights, dtype=torch.float).to(device)).item()
if not os.path.exists(dreamfolder):
    os.makedirs(dreamfolder)
with open(cnfg['dream_file'], 'a') as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow([start_res, start_pred, start_graph.weights])

start_time = time.time()
dream_model(model, state, start_graph, cnfg)

print(f"--- done in {time.time() - start_time} seconds ---")
