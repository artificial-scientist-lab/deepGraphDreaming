import pickle
from random import randrange

import numpy as np
import pytheus.help_functions as hf
import pytheus.fancy_classes as fc
from pytheus import theseus as th


# # Training data generator
# # Fidelity generating function (Tareq)
def generatorGraphFidelity(dimensions, desired_state, num_edges=None, short_output=True):
    """
    Generates graphs and computes their fidelity with respect to some desired state
    """

    if num_edges == None:
        rand_graph = fc.Graph(th.buildAllEdges(dimensions))  # full graph
        rand_graph.getState()
        rand_state = rand_graph.state
        rand_state.normalize()
    fidelity = abs(rand_state @ desired_state) ** 2

    if short_output:  # array of the edges' weights (includes 0 valued edges) and fidelity
        return rand_graph.weights, fidelity
    else:  # dictionaries with edges names and values, generated kets, and fidelity
        return rand_graph, rand_state.amplitudes, fidelity


def constructGraph(neoEdgeWeights, dimensions, desired_state):
    # We update our graph now with potentially new weight values and recompute the fidelity
    graph_neo = th.buildAllEdges(dimensions)
    graph_neo = fc.Graph(graph_neo, weights=neoEdgeWeights)
    graph_neo.getState()
    state_neo = graph_neo.state
    state_neo.normalize()

    fidelity = abs(state_neo @ desired_state)

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

    DIM = [2] * 6
    kets = hf.makeState('000000+111111')
    state = fc.State(kets, normalize=True)
    print(state)
    num_of_examples = 10000000
    input_graph, ket_amplitudes, output_fidelity = generatorGraphFidelity(DIM, state, short_output=False)

    print("Training Data...")
    data = np.zeros((num_of_examples, len(input_graph)))
    res = np.zeros((num_of_examples, 1))

    for ii in range(num_of_examples):
        input_graph, ket_amplitudes, output_fidelity = generatorGraphFidelity(DIM, state, short_output=False)
        data[ii, :] = input_graph.weights
        res[ii] = output_fidelity
        if ii % 100 == 0:
            print('Training data: ', ii, '/', num_of_examples)

    with open('data_train.pkl', 'wb') as f:
        pickle.dump([data, res], f)

    print("Done!")
