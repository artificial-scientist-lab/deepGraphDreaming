
import zipfile
import torch
import yaml
import os

from torch import nn, rand
from random import shuffle, randrange

'''


from utilities import data_loader
from utilities import plot_utils

from utilities.mol_utils import multiple_selfies_to_hot, edit_hot, lst_of_logP, multiple_hot_to_indices
from utilities.utils import make_dir, change_str, use_gpu

'''

import theseus as th
import numpy as np
import itertools
import time
import io
import graphplot as gp


# # Concurrence tools from Jan

def ptrace(u, keep, dims, optimize=False):
    """Calculate the partial trace of an outer product

    ρ_a = Tr_b(|u><u|)

    from: https://scicomp.stackexchange.com/questions/30052/calculate-partial-trace-of-an-outer-product-in-python?rq=1
    Parameters
    ----------
    u : array
        Vector to use for outer product
    keep : array
        An array of indices of the spaces to keep after
        being traced. For instance, if the space is
        A x B x C x D and we want to trace out B and D,
        keep = [0,2]
    dims : array
        An array of the dimensions of each space.
        For instance, if the space is A x B x C x D,
        dims = [dim_A, dim_B, dim_C, dim_D]

    Returns
    -------
    ρ_a : 2D array
        Traced matrix
    """
    keep = np.asarray(keep)
    dims = np.asarray(dims)
    Ndim = dims.size
    Nkeep = np.prod(dims[keep])

    idx1 = list(range(Ndim))
    idx2 = [Ndim+i if i in keep else i for i in range(Ndim)]
    u = u.reshape(dims)
    rho_a = np.einsum(u, idx1, u.conj(), idx2, optimize=optimize)
    return rho_a.reshape(Nkeep, Nkeep)


def get_all_bi_partions(dim: int):
    """
    returns all possible bi-partions as a generatos for a given dimension:

        e.g. : dim = 3 : [([0], [1, 2]), ([1], [0, 2]), ([2], [0, 1])]

    """

    S = { i for i in range( dim) }
    doubles = []
    for l in range(1,int(len(S)/2)+1):
        combinations = set(itertools.combinations(S,l))
        for oneC in combinations:
            if sorted(list(oneC)) not in doubles:
                yield (sorted(list(oneC)), sorted(list(S-set(oneC))))
            doubles.append(sorted(list(S-set(oneC))) )

def get_sysdict(dim: list):
    dd = dict()
    dd['dimensions'] = dim
    dd['all_biparations'] = get_all_bi_partions(len(dim))
    return dd


def compute_concurrence(qstate, sys_dict):
    """

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
    
    #print(qstate)
    #input()
    

    dimi = np.array(sys_dict['dimensions'])
    qstate *=   1/(np.linalg.norm(qstate))
    calc_con  = lambda mat, par:  np.sqrt(2 * ( 1 - min(np.trace(( np.linalg.matrix_power(ptrace(mat,par,dimi) ,2)  ) ),1) ) )
    spiffy = abs(sum([calc_con(qstate,par[0]) for par in sys_dict['all_biparations'] ]))
    
    #print(spiffy)
    #input()
    
    
    return spiffy


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
        
    NormalisationConstant = np.sum(np.array(AllEquations)**2)
    Fidelity = np.sum(np.array(TargetEquations))**2/(len(TargetEquations)*NormalisationConstant)
    
    return Fidelity
    
    '''
    AllEquations = []
    TargetEquations = []
    
    for comb in self.combinations:
        newEq = np.sum([i for i in self.TriggerableState.args if str(comb) in str(i)]).subs([(comb,1)])
        for state in desired_state:
            if comb == self.state_symbol(state):
                # This term is part of the quantum state we want
                TargetEquations.append(newEq)
        AllEquations.append(newEq)
    # Run the Optimization 
    self.TargetEquation = TargetEquations
    NormalisationConstant2 = np.sum(np.array(AllEquations)**2)
    self.NormalisationConstant = NormalisationConstant2
    Fidelity = np.sum(np.array(TargetEquations))**2/(len(TargetEquations)*NormalisationConstant2)
    return Fidelity
 '''
 
# # Training data generator

def generatorGraphConcurrence(dimensions, num_edges = None, short_output = True):
    # Dictionary with all possible kets given the input dimensions
    all_kets_dict = {ket:[] for ket in th.allEdgeCovers(dimensions, order=0)}
    if num_edges == None: 
        rand_graph = th.buildAllEdges(dimensions) # full graph
        possible_kets = th.stateCatalog(th.findPerfectMatchings(rand_graph))
    else:
        perfect_matching = False
        count_perfect_matchings = 0
        while not perfect_matching: # Check to guarantee at least one perfect matching 
            rand_graph = th.buildRandomGraph(dimensions=dimensions, num_edges=num_edges)
            possible_kets = th.stateCatalog(th.findPerfectMatchings(rand_graph))
            count_perfect_matchings = len(possible_kets)
            if len(possible_kets) > 0: perfect_matching = True
    all_kets_dict.update(possible_kets)
    # Now the dictionary includes the perfect matchings from the random graph
    
    
    # Dictionary with edge values (randomly assigned)
    edge_weights = {edge:0 for edge in th.buildAllEdges(dimensions)}
    for edge in rand_graph: 
        edge_weights[edge] = 2 * np.random.rand() - 1
    
    # Dictionary with the amplitudes for each of the possible kets 
    ket_amplitudes = {ket:0 for ket in all_kets_dict.keys()}
    for ket, graph_list in all_kets_dict.items():
        for graph in graph_list:
            term = 1
            for edge in graph:
                term *= edge_weights[edge]
            ket_amplitudes[ket] += term
    
    
    # Generation of concurrence with Jan's functions
    ket_coeffs = np.array(list(ket_amplitudes.values()))
    
    concurrence = compute_concurrence(ket_coeffs, get_sysdict(dimensions))
    
    #if (concurrence == 0):
        # print(" ** ZERO CONCURRENCE ** ")
       # print(" All Kets ")
       #  print(possible_kets)
        # input()
       # print(" All edge weights ")
      #  print(edge_weights)
       # input()
      #  print(" All ket coefficients ")
       #  print(ket_coeffs)
       # input()
        
    if short_output: # array of the edges' weights (includes 0 valued edges) and concurrence
        return np.array(list(edge_weights.values())), concurrence
    else: # dictionaries with edges names and values, generated kets, and concurrence
        return edge_weights, ket_amplitudes, concurrence

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
        
    NormalisationConstant = np.sum(np.array(AllEquations)**2)

    Fidelity = np.abs(np.sum(np.array(TargetEquations)))**2/(len(TargetEquations)*NormalisationConstant)
    return Fidelity
    

def generatorGraphFidelity(dimensions,desired_state, num_edges = None, short_output = True):
    
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
    all_kets_dict = {ket:[] for ket in th.allEdgeCovers(dimensions, order=0)}
    if num_edges == None: 
        rand_graph = th.buildAllEdges(dimensions) # full graph
        possible_kets = th.stateCatalog(th.findPerfectMatchings(rand_graph))
    else:
        perfect_matching = False
        count_perfect_matchings = 0
        while not perfect_matching: # Check to guarantee at least one perfect matching 
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
    edge_weights = {edge:0 for edge in th.buildAllEdges(dimensions)}
    for edge in rand_graph: 
        edge_weights[edge] = 2 * np.random.rand() - 1
    
    # Dictionary with the amplitudes for each of the possible kets 
    ket_amplitudes = {ket:0 for ket in all_kets_dict.keys()}
    for ket, graph_list in all_kets_dict.items():
        for graph in graph_list:
            term = 1
            for edge in graph:
                term *= edge_weights[edge]
            ket_amplitudes[ket] += term
    
    
    # Generation of concurrence with Jan's functions
    ket_coeffs = np.array(list(ket_amplitudes.values()))
    fidelity = compute_fidelity(ket_amplitudes, desired_state)
    
    if short_output: # array of the edges' weights (includes 0 valued edges) and fidelity
        return np.array(list(edge_weights.values())), fidelity
    else: # dictionaries with edges names and values, generated kets, and fidelity
        return edge_weights, ket_amplitudes, fidelity


def edit_graph(graph, upper_bound):
    """Replaces all zeroes with a random float in the range [0,upper_bound]"""
    #t1=time.clock()
    for edge in graph:        
        graph[edge] += upper_bound*randrange(-1,1)
        if (graph[edge] > 1):
            graph[edge] = 1
        elif (graph[edge] < -1):
            graph[edge] = -1
        
    return graph

def neuron_selector(model, device, layer,  neuron):
    
    '''
    This creates a new model with the same weights and biases as the input model
    but only includes the trained layers up to the layer containing the neuron
    we want to analyze. 
    
    The way this'll work is that we initialize a new model consisting of every layer of the trained neural network 
    up to the layer containing the neuron we want to analyze. We then create a new 'output layer' consisting solely of that particular neuron. 
    The weights/biases of the neuron must be the same as it was in the original. 
    '''
    
    total_model = list(model.mynn.children())
    
    old_output_layer = total_model[layer] # The weights here are always the same! (as they should...)
    in_features = old_output_layer.in_features
    
    new_output_layer = nn.Linear(in_features,1) # The initial weights/biases here are random, 'uninitialized'values.
    
    with torch.no_grad():
     new_output_layer.weight[0] = old_output_layer.weight[neuron]
     new_output_layer.bias[0] = old_output_layer.bias[neuron]
    
    
    if (layer == 0):
        print("gobble gobble")
        new_model = new_output_layer.to(device)
    else:
        print(f"gobble gobble {layer}")
        middle_model = total_model[:layer-1]
        new_model = nn.Sequential(*middle_model,nn.ReLU(), new_output_layer).to(device)
        
    new_model.eval()
    
   
    return new_model


def dream_model(dimensions, model, num_of_examples, desired_state, data_train, lr, num_epochs, name_of_zip, layer_index, neuron_index,  display=True):
    
    """
    Inverse trains the model by freezing the weights and biases and optimizing instead for the input.
    In particular, we want to find the input that maximizes our output from whatever neuron we are interested in looking over
    
    PARAMETERS
    dimensions - dimension of our input (and desired graph)
    model - our trained model
    num_of_examples - number of examples in our dataset
    desired state - we compute the fidelity of the dreamed graph states with respect to the desired state
    data_train - input graph
    lr - learning rate 
    num_epochs - maximum number of epochs that we run the dreaming
    name_of_zip - name of the zip file that we save our graph movie frames
    layer_index - the layer of our model that contains the neuron that we want to analyze
    neuron_index - the neuron that we want to analyze
    display - makes the output longer or shorter
    
    """
    
    loss_prediction=[]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prop = 1
    
    #data_train = edit_graph(data_train, upper_bound)
    
    graph_edge_weights = torch.tensor(np.array(list(data_train.values())), dtype=torch.float).to(device)
    data_train_prop=torch.tensor(prop, dtype=torch.float).to(device)
    data_train_var=torch.autograd.Variable(graph_edge_weights, requires_grad=True)
    
    #initiailize list of intermediate property values and molecules
    interm_prop = []
    nn_prop = []
    gradDec = []
    interm_graph = [data_train]
    
    epoch_transformed = [0]
    steps = 0
    valid_steps = 0
    
    # initialize an instance of the model
    optimizer_encoder = torch.optim.Adam([data_train_var], lr=lr)
    interm_model = neuron_selector(model,device,layer_index,neuron_index)

    for epoch in range(num_epochs):

        # feedforward step
        
        calc_properties = interm_model(data_train_var)
        nn_prop.append(calc_properties.cpu().detach().numpy())

        # mean squared error between target and calculated property
        calc_properties = calc_properties.reshape(1)
        #criterion = nn.MSELoss()
        #real_loss=criterion(calc_properties, data_train_prop) # So we calculate the mean squared error between the predicted fidelity and the target one
        real_loss = -calc_properties
        loss = torch.clamp(real_loss, min = -50000, max = 50000.).double()
        # backpropagation step
        
        optimizer_encoder.zero_grad()
        loss.backward()
        optimizer_encoder.step()
        
        real_loss=loss.cpu().detach().numpy()
        loss_prediction.append(float(real_loss))
        
        input_grad = data_train_var.grad.cpu().detach().numpy()
        input_grad_norm = np.linalg.norm(input_grad, ord=2)
        gradDec.append(input_grad_norm)
        
        if epoch%100==0:
              print('epoch: ',epoch,', gradient: ', input_grad_norm)

        # We update our graph now with potentially new weight values and recompute the fidelity
        neo_edge_weights = data_train_var.cpu().detach().numpy()
        fidelity,edge_weights = constructGraph(neo_edge_weights, dimensions, desired_state)
        

        if len(interm_prop)==0 or interm_prop[len(interm_prop)-1] != fidelity:
            
            # collect intermediate graphs
            interm_graph.append(edge_weights)
            interm_prop.append(fidelity)
            
            steps+=1
            epoch_transformed.append(epoch+1)

            if len(interm_prop)>1:

                # determine validity of transformation
                previous_prop = interm_prop[len(interm_prop)-2]
                current_prop = fidelity
             
                valid = (prop > previous_prop and current_prop > previous_prop) \
                        or (prop < previous_prop and current_prop < previous_prop)
                
                if valid:
                    valid_steps += 1
        
        
        if len(gradDec)>1000:
            if gradDec[-1] < 1e-7 and 0.99*gradDec[-100]<=gradDec[-1]:
              print('The gradient is very near zero at this point, stop dreaming at epoch ', epoch)
              break
            else: 
             if nn_prop[-1] - nn_prop[-100] < 1e-7:
              print('Our predictions arent changing much, maybe our gradient is going back and forth? Stop dreaming at epoch ', epoch)
              break
        
 
    # Make a plot for the intermediate graph and save in a zip file. 

    if display:
       print("Creating archive: {:s}".format(name_of_zip))
    
       with zipfile.ZipFile(name_of_zip, mode="w") as zf: 
         for i in range(0,len(interm_graph),int(len(interm_graph)/10)):
            # First, we reformat the interm graph into something compatible with the graph plotting functions
            graph_to_go = []
            
            temp = interm_graph[i]
            temp_keys = list(temp.keys())
            temp_vals = list(temp.values())
            for j in range(len(temp_keys)):
                graph_to_go.append(temp_keys[j]+tuple([temp_vals[j]]))
         
            # Now save the interm plot to the zip
            interm_graph_plot = gp.graphPlot(graph_to_go,1,i,interm_prop[j],  scaled_weights=True, show=False, max_thickness=10, multiple_graphs = True)
            #input()
            buf = io.BytesIO()
            interm_graph_plot.savefig(buf)
            img_name = "graph_fig_{:02d}.png".format(i)
            zf.writestr(img_name,buf.getvalue())
                
    percent_valid_transform = None
    
    if steps > 0:
        percent_valid_transform = valid_steps / steps *100
        
    return interm_prop[-1], interm_graph, loss_prediction, interm_prop, nn_prop,gradDec, percent_valid_transform, epoch_transformed


def constructGraph(neoEdgeWeights,dimensions, desired_state):
    # We update our graph now with potentially new weight values and recompute the fidelity
    graph_neo = th.buildAllEdges(dimensions)
    all_kets_dict = {ket:[] for ket in th.allEdgeCovers(dimensions, order=0)}
    possible_kets = th.stateCatalog(th.findPerfectMatchings(graph_neo))
    all_kets_dict.update(possible_kets)
    
    edge_weights = {edge:0 for edge in th.buildAllEdges(dimensions)}
    ii = 0
    for edge in graph_neo:
        edge_weights[edge] = neoEdgeWeights[ii]
        ii += 1
    
    ket_amplitudes = {ket:0 for ket in all_kets_dict.keys()}
    for ket, graph_list in all_kets_dict.items():
        for graph in graph_list:
            term = 1
            for edge in graph:
                term *= edge_weights[edge]
            ket_amplitudes[ket] += term
         
    fidelity = compute_fidelity(ket_amplitudes, desired_state)
    
    return fidelity, edge_weights

    

def constructGraph2(neoEdgeWeights,dimensions, desired_state):
    # We update our graph now with potentially new weight values and recompute the fidelity
    graph_neo = th.buildAllEdges(dimensions)
    all_kets_dict = {ket:[] for ket in th.allEdgeCovers(dimensions, order=0)}
    possible_kets = th.stateCatalog(th.findPerfectMatchings(graph_neo))
    all_kets_dict.update(possible_kets)
    
    edge_weights = {edge:0 for edge in th.buildAllEdges(dimensions)}
    ii = 0
    for edge in graph_neo:
        edge_weights[edge] = neoEdgeWeights[ii]
        ii += 1
    
    ket_amplitudes = {ket:0 for ket in all_kets_dict.keys()}
    for ket, graph_list in all_kets_dict.items():
        for graph in graph_list:
            term = 1
            for edge in graph:
                term *= edge_weights[edge]
            ket_amplitudes[ket] += term
         
    fidelity = compute_fidelity(ket_amplitudes, desired_state)
    
    return fidelity, edge_weights, ket_amplitudes




