import pickle
from random import randrange

import pytheus.theseus as th
import numpy as np
from pytheus import theseus as th


# # Training data generator
# # Fidelity generating function (Tareq)

def compute_fidelity(q_state, desired_state):
    ' For now, let us just see what is is exactly that we are feeding this '

    """
    Computes the fidelity of a graph's resulting quantum state with respect to some desired quantum state

    Parameters
    ----------
    state : numpy array
        state list in state_format (see top) [1,0,0,0.7,1.2,...]
    sys_dict : dict
        that stores essential infos of quantuum system (see top.get_sysdict).

    Returns
    -------
    TYPE
        concorrence:  C( |Psi> ) = √( 2 * ( 1 - TR_M( <Psi|Psi> ) ) ) where TR_M is partial trace (in subsystem M)
        and return is sum over all possible bipartion

    """

    AllEquations = []
    TargetEquations = []
    for state1 in q_state:
        newEq = q_state[state1]
        for state2 in desired_state:
            if (state1 == state2):
                TargetEquations.append(newEq)
        AllEquations.append(newEq)

    NormalisationConstant = np.sum(np.array(AllEquations) ** 2)

    Fidelity = np.abs(np.sum(np.array(TargetEquations))) ** 2 / (len(TargetEquations) * NormalisationConstant)
    return Fidelity


def generatorGraphFidelity(dimensions, desired_state, num_edges=None, short_output=True):
    """
    Generates graphs and computes their fidelity with respect to some desired state

    Parameters
    ----------
    dimensions : numpy array
        dimensionality of the graph we wwish to generate
    desired_state : dict
        This is a dictionary with the desired kets of our state as the keys and their weights as values
    num_edges : int
        Total number of edges that the graph can have
    short_output : boolean
        ????

    Returns
    -------
    TYPE
        concorrence:  C( |Psi> ) = √( 2 * ( 1 - TR_M( <Psi|Psi> ) ) ) where TR_M is partial trace (in subsystem M)
        and return is sum over all possible bipartion

    """

    # Dictionary with all possible kets given the input dimensions
    all_kets_dict = {ket: [] for ket in th.allEdgeCovers(dimensions, order=0)}
    if num_edges == None:
        rand_graph = th.buildAllEdges(dimensions)  # full graph
        possible_kets = th.stateCatalog(th.findPerfectMatchings(rand_graph))
    else:
        perfect_matching = False
        count_perfect_matchings = 0
        while not perfect_matching:  # Check to guarantee at least one perfect matching
            rand_graph = th.buildRandomGraph(dimensions=dimensions, num_edges=num_edges)
            possible_kets = th.stateCatalog(th.findPerfectMatchings(rand_graph))
            count_perfect_matchings = len(possible_kets)
            if len(possible_kets) > 0: perfect_matching = True
    '''
    print(all_kets_dict.items())
    time.sleep(5)
    '''
    all_kets_dict.update(possible_kets)
    # Now the dictionary includes the perfect matchings from the random graph
    '''
    print(all_kets_dict.items())
    time.sleep(5)
    '''
    # Dictionary with edge values (randomly assigned)
    edge_weights = {edge: 0 for edge in th.buildAllEdges(dimensions)}
    for edge in rand_graph:
        edge_weights[edge] = 2 * np.random.rand() - 1

    # Dictionary with the amplitudes for each of the possible kets
    ket_amplitudes = {ket: 0 for ket in all_kets_dict.keys()}
    for ket, graph_list in all_kets_dict.items():
        for graph in graph_list:
            term = 1
            for edge in graph:
                term *= edge_weights[edge]
            ket_amplitudes[ket] += term

    # Generation of concurrence with Jan's functions
    ket_coeffs = np.array(list(ket_amplitudes.values()))
    fidelity = compute_fidelity(ket_amplitudes, desired_state)

    if short_output:  # array of the edges' weights (includes 0 valued edges) and fidelity
        return np.array(list(edge_weights.values())), fidelity
    else:  # dictionaries with edges names and values, generated kets, and fidelity
        return edge_weights, ket_amplitudes, fidelity

def constructGraph(neoEdgeWeights, dimensions, desired_state):
    # We update our graph now with potentially new weight values and recompute the fidelity
    graph_neo = th.buildAllEdges(dimensions)
    all_kets_dict = {ket: [] for ket in th.allEdgeCovers(dimensions, order=0)}
    possible_kets = th.stateCatalog(th.findPerfectMatchings(graph_neo))
    all_kets_dict.update(possible_kets)

    edge_weights = {edge: 0 for edge in th.buildAllEdges(dimensions)}
    ii = 0
    for edge in graph_neo:
        edge_weights[edge] = neoEdgeWeights[ii]
        ii += 1

    ket_amplitudes = {ket: 0 for ket in all_kets_dict.keys()}
    for ket, graph_list in all_kets_dict.items():
        for graph in graph_list:
            term = 1
            for edge in graph:
                term *= edge_weights[edge]
            ket_amplitudes[ket] += term

    fidelity = compute_fidelity(ket_amplitudes, desired_state)

    return fidelity, edge_weights


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

    DIM = [2] * 4
    # 4-particle GHZ state. Let's focus on this for now

    desired_state_2 = {
        ((0, 0), (1, 0), (2, 0), (3, 0)): (1 / np.sqrt(2)),
        ((0, 1), (1, 1), (2, 1), (3, 1)): (1 / np.sqrt(2))
    }
    num_of_examples = 10000000
    input_edges, ket_amplitudes, output_fidelity = generatorGraphFidelity(DIM, desired_state_2, num_edges=None,
                                                                          short_output=False)
    input_edge_weights = np.array(list(input_edges.values()))

    # input_edges returns an array of floats between +- 1 .... are these the weights of the edges?

    create_training_data_new = True

    if create_training_data_new == True:
        print("Training Data...")
        data = np.zeros((num_of_examples, len(input_edges)))
        res = np.zeros((num_of_examples, 1))

        for ii in range(num_of_examples):
            input_edges, ket_amplitudes, output_fidelity = generatorGraphFidelity(DIM, desired_state_2, num_edges=None,
                                                                                  short_output=False)
            input_edge_weights = np.array(list(input_edges.values()))
            data[ii, :] = input_edge_weights
            res[ii] = output_fidelity
            if ii % 100 == 0:
                print('Training data: ', ii, '/', num_of_examples)

        with open('graph_examples_fidelity_neo.pkl', 'wb') as f:
            pickle.dump([data, res], f)

    else:
        with open('graph_simple_fidelity_10000000.pkl', 'rb') as f:
            data_full, res_full = pickle.load(f)

        data = data_full[0:num_of_examples]
        res = res_full[0:num_of_examples]

    # split data into train and test
    print("Done!")