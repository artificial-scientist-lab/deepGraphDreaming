import pickle
from random import randrange

import numpy as np
import pytheus.help_functions as hf
import pytheus.fancy_classes as fc
from pytheus import theseus as th
from pytheus.lossfunctions import fidelity, make_lossString_entanglement
import yaml
from yaml import Loader
import csv
import math
import argparse



def discretize_weight(weight):
    if abs(weight) < 0.33:
        return 0
    elif weight > 0:
        return 1
    else:
        return -1


def thresholded_linear(weight):
    if abs(weight) < 0.33:
        return 0
    else:
        return weight


def generatorGraphFidelity(dimensions, desired_state, num_edges=None, short_output=True, discretize=False):
    """
    Generates graphs and computes their fidelity with respect to some desired state
    """

    if num_edges == None:
        alledges = th.buildAllEdges(dimensions)
        randweights = 2 * np.random.rand(len(alledges)) - 1
        if discretize:
            if discretize == 'thr_linear':
                weights = list(map(thresholded_linear, randweights))
            else:
                weights = list(map(discretize_weight, randweights))
        else:
            weights = randweights
        rand_graph = fc.Graph(alledges, weights=weights)  # full graph
        rand_graph.getState()
        rand_state = rand_graph.state
        rand_state.normalize()
    fidelity = abs(rand_state @ desired_state) ** 2
    if math.isnan(fidelity):
        fidelity = 0

    if short_output:  # array of the edges' weights (includes 0 valued edges) and fidelity
        return weights, fidelity
    else:  # dictionaries with edges names and values, generated kets, and fidelity
        return rand_graph, rand_state.amplitudes, fidelity


def quickgenerate(func, numargs, isZero, prop):
    """
    Generates graphs and computes their fidelity with respect to some desired state
    """
    
    if(isZero):
        randweights = np.zeros(numargs)
    else:
        randweights = 2 * np.random.rand(numargs) - 1
    if discretize:
        if discretize == 'thr_linear':
            weights = [thresholded_linear(weight) for weight in randweights]
        else:
            weights = [discretize_weight(weight) for weight in randweights]
    else:
        weights = randweights
    
    
    if(prop == 'concurrence'):
        try:
            fidelity = func(weights)  
        except ZeroDivisionError:
            fidelity = 0
            print(weights, flush=True)
    else:
        try:
            fidelity= 1 - func(weights)  
        except ZeroDivisionError:
            fidelity = 0
            print(weights, flush=True)

    return weights, fidelity


def constructGraph(neoEdgeWeights, dimensions, func, prop):
    
    # We update our graph now with potentially new weight values and recompute the fidelity
    
    graph_neo = th.buildAllEdges(dimensions)
    neoEdgeWeights = [float(item) for item in neoEdgeWeights]
    graph_neo = fc.Graph(graph_neo, weights=neoEdgeWeights)
    graph_neo.getState()
    state_neo = graph_neo.state
    state_neo.normalize()
    
    fidelity = 1 - func(neoEdgeWeights) # This should be what you do for both fidelity AND concurrence

    return fidelity, graph_neo


def edit_graph(graph, upper_bound):
    """Replaces all zeroes with a random float in the range [0,upper_bound]"""
    # t1=time.clock()
    for edge in graph:
        graph[edge] += upper_bound * randrange(-1, 1)
        if (graph[edge] > 1):
            graph[edge] = 1
        elif (graph[edge] < -1):
            graph[edge] = -1
    return graph


if __name__ == '__main__':


    stream = open(f"configs/datagen.yaml", 'r')
    cnfg = yaml.load(stream, Loader=Loader)
				
    DIM = eval(cnfg['dim'])
    kets = hf.makeState(cnfg['state'])
    state = fc.State(kets, normalize=True)
    sysDict = hf.get_sysdict(DIM)
    print(state)
    num_of_examples = cnfg['num_of_examples']
    file_type = cnfg['file_type']
    filename = cnfg['file_name']
    isBinary = False
    
    toBoolean = {'True': True, 'False':False}
    
    discretize = toBoolean[cnfg['discretize']]
    print(discretize)

    cnfgfid = {"heralding_out": False, "imaginary": False}
    alledges = th.buildAllEdges(DIM)
    graph = fc.Graph(alledges)
    if(cnfg['prop'] == 'concurrence'): # Computes the concurrence of a random graph
        fid = make_lossString_entanglement(graph,sysDict)
    else: # Computes the fidelity, though this can be reconfigured for any other quantum property we wanna measure
        fid = fidelity(graph,state,cnfgfid)
    numargs = len(th.buildAllEdges(DIM))
    threshold = cnfg['threshold']

    print("Training Data...")
    
    if file_type == 'pkl':
        data = np.zeros((num_of_examples, numargs))
        res = np.zeros((num_of_examples, 1))

    for ii in range(num_of_examples):
        # generate sample
        weights, output_fidelity = quickgenerate(fid, numargs,isBinary, cnfg['prop'])
        while(output_fidelity > threshold):
            weights, output_fidelity = quickgenerate(fid, numargs,isBinary, cnfg['prop'])
            
        if file_type == 'csv':  # if csv, write line by line
            with open(filename + '.csv', 'a') as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow([list(weights), output_fidelity])
        else:
            data[ii, :] = weights
            res[ii] = output_fidelity
        if ii % 1000 == 0:
            print('Training data: ', ii, '/', num_of_examples)
            print(f'Samnple Weights: {weights}')
    if file_type == 'pkl':
        with open(filename + '.pkl', 'wb') as f:
            pickle.dump([data, res], f)

    print("Done!")
