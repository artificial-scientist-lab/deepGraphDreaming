import csv
import os
import numpy as np
import random
import time

import torch
from torch import nn

import matplotlib.pyplot as plt

from datagen import constructGraph
#a

class ff_network(nn.Module):
    def __init__(self, size_of_input, size_of_output, type):
        super(ff_network, self).__init__()

        if (type == 1):  # small neural network with 5 hidden layers and 30 neurons per layer
            self.mynn = nn.Sequential(
                nn.Linear(size_of_input, 15 * 2),
                nn.ReLU(),
                nn.Linear(15 * 2, 15 * 2),
                nn.ReLU(),
                nn.Linear(15 * 2, 15 * 2),
                nn.ReLU(),
                nn.Linear(15 * 2, 15 * 2),
                nn.ReLU(),
                nn.Linear(15 * 2, 15 * 2),
                nn.ReLU(),
                nn.Linear(15 * 2, size_of_output)
            )

        if (type == 2):  # This was the neural network we had been dreaming with before
            self.mynn = nn.Sequential(
                nn.Linear(size_of_input, 576 * 2),
                nn.ReLU(),
                nn.Linear(576 * 2, 200 * 2),
                nn.ReLU(),
                nn.Linear(200 * 2, 100 * 2),
                nn.ReLU(),
                nn.Linear(100 * 2, 50 * 2),
                nn.ReLU(),
                nn.Linear(50 * 2, 30 * 2),
                nn.ReLU(),
                nn.Linear(30 * 2, 30 * 2),
                nn.ReLU(),
                nn.Linear(30 * 2, 30 * 2),
                nn.ReLU(),
                nn.Linear(30 * 2, 30 * 2),
                nn.ReLU(),
                nn.Linear(30 * 2, 30 * 2),
                nn.ReLU(),
                nn.Linear(30 * 2, 30 * 2),
                nn.ReLU(),
                nn.Linear(30 * 2, 30 * 2),
                nn.ReLU(),
                nn.Linear(30 * 2, 30 * 2),
                nn.ReLU(),
                nn.Linear(30 * 2, 30 * 2),
                nn.ReLU(),
                nn.Linear(30 * 2, 30 * 2),
                nn.ReLU(),
                nn.Linear(30 * 2, 30 * 2),
                nn.ReLU(),
                nn.Linear(30 * 2, 25 * 2),
                nn.ReLU(),
                nn.Linear(25 * 2, size_of_output)
            )

        if (type == 3):
            constantsize = 100
            self.mynn = nn.Sequential(
                nn.Linear(size_of_input, constantsize),
                nn.ReLU(),
                nn.Linear(constantsize, constantsize),
                nn.ReLU(),
                nn.Linear(constantsize, constantsize),
                nn.ReLU(),
                nn.Linear(constantsize, constantsize),
                nn.ReLU(),
                nn.Linear(constantsize, constantsize),
                nn.ReLU(),
                nn.Linear(constantsize, size_of_output)
            )

        if (type == 4):
            constantsize = 200
            self.mynn = nn.Sequential(
                nn.Linear(size_of_input, constantsize),
                nn.ReLU(),
                nn.Linear(constantsize, constantsize),
                nn.ReLU(),
                nn.Linear(constantsize, constantsize),
                nn.ReLU(),
                nn.Linear(constantsize, constantsize),
                nn.ReLU(),
                nn.Linear(constantsize, constantsize),
                nn.ReLU(),
                nn.Linear(constantsize, constantsize),
                nn.ReLU(),
                nn.Linear(constantsize, constantsize),
                nn.ReLU(),
                nn.Linear(constantsize, size_of_output)
            )

        if (type == 5):
            constantsize = 200
            self.mynn = nn.Sequential(
                nn.Linear(size_of_input, constantsize),
                nn.ReLU(),
                nn.Linear(constantsize, constantsize),
                nn.ReLU(),
                nn.Linear(constantsize, constantsize),
                nn.ReLU(),
                nn.Linear(constantsize, constantsize),
                nn.ReLU(),
                nn.Linear(constantsize, constantsize),
                nn.ReLU(),
                nn.Linear(constantsize, constantsize),
                nn.ReLU(),
                nn.Linear(constantsize, 20),
                nn.ReLU(),
                nn.Linear(20, size_of_output)
            )
            
        if (type == 6): # "Wider but narrower approach" for 4-qubits.
                self.mynn = nn.Sequential(
                    nn.Linear(size_of_input, size_of_input),
                    nn.ReLU(),
                    nn.BatchNorm1d(size_of_input),
                    nn.Linear(size_of_input, 16),
                    nn.ReLU(),
                    nn.BatchNorm1d(16),
                    nn.Linear(16, size_of_output)
                )
                
        if (type == 7): #  Same neural network, but with duplicate layers
                        self.mynn = nn.Sequential(
                            nn.Linear(size_of_input, size_of_input),
                            nn.ReLU(),
                            nn.Linear(size_of_input, size_of_input),
                            nn.ReLU(),
                            nn.Linear(size_of_input, size_of_input),
                            nn.ReLU(),
                            nn.Linear(size_of_input, 16),
                            nn.ReLU(),
                            nn.Linear(16,16),
                            nn.ReLU(),
                            nn.Linear(16,16),
                            nn.ReLU(),
                            nn.Linear(16, 2),
                            nn.ReLU(),
                            nn.Linear(2,size_of_output)
                        )
                        
        if (type == 8): # Same neural network again, but with more duplicate layers
                self.mynn = nn.Sequential(
                    nn.Linear(size_of_input, size_of_input),
                    nn.ReLU(),
                    nn.Linear(size_of_input, size_of_input),
                    nn.ReLU(),
                    nn.Linear(size_of_input, size_of_input),
                    nn.ReLU(),
                    nn.Linear(size_of_input, size_of_input),
                    nn.ReLU(),
                    nn.Linear(size_of_input, size_of_input),
                    nn.ReLU(),
                    nn.Linear(size_of_input, 16),
                    nn.ReLU(),
                    nn.Linear(16,16),
                    nn.ReLU(),
                    nn.Linear(16,16),
                    nn.ReLU(),
                    nn.Linear(16,16),
                    nn.ReLU(),
                    nn.Linear(16,16),
                    nn.ReLU(),
                    nn.Linear(16, 2),
                    nn.ReLU(),
                    nn.Linear(2,size_of_output)
                    )
            
                
        if (type == 9): # "Wider but narrower approach" for 6-qubits.
                        self.mynn = nn.Sequential(
                            nn.Linear(size_of_input, 64),
                            nn.ReLU(),
                            nn.Linear(64, size_of_input),
                            nn.ReLU(),
                            nn.Linear(size_of_input, size_of_output)
                        )
                        
        if (type == 10): # Same number of neurons
                        self.mynn = nn.Sequential(
                            nn.Linear(size_of_input, 64),
                            nn.ReLU(),
                            nn.Linear(64, 64),
                            nn.ReLU(),
                            nn.Linear(64, 64),
                            nn.ReLU(),
                            nn.Linear(64, size_of_input),
                            nn.ReLU(),
                            nn.Linear(size_of_input, size_of_input),
                            nn.ReLU(),
                            nn.Linear(size_of_input, size_of_input),
                            nn.ReLU(),
                            nn.Linear(size_of_input, 2),
                            nn.ReLU(), 
                            nn.Linear(2,size_of_output))
                        
        if (type == 11):  # same kind of neural network, but with more layers
        
                    self.mynn = nn.Sequential(
                        nn.Linear(size_of_input, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, size_of_input),
                        nn.ReLU(),
                        nn.Linear(size_of_input, size_of_input),
                        nn.ReLU(),
                        nn.Linear(size_of_input, size_of_input),
                        nn.ReLU(),
                        nn.Linear(size_of_input, size_of_input),
                        nn.ReLU(),
                        nn.Linear(size_of_input, size_of_input),
                        nn.ReLU(),
                        nn.Linear(size_of_input, 2),
                        nn.ReLU(), 
                        nn.Linear(2,size_of_output) 
                        )
        if (type == 12): # NN Architecture Type #1 suggested by Mario
                self.mynn = nn.Sequential(
                    nn.Linear(size_of_input, 20),
                    nn.ReLU(),
                    nn.Linear(20,20),
                    nn.ReLU(),
                    nn.Linear(20,20),
                    nn.ReLU(),
                    nn.Linear(20,size_of_output)
                    )
        if (type==13): # NN Architecture Type #2 suggested by Mario
                self.mynn = nn.Sequential(
                    nn.Linear(size_of_input, 50),
                    nn.ReLU(),
                    nn.Linear(50,40),
                    nn.ReLU(),
                    nn.Linear(40,30),
                    nn.ReLU(),
                    nn.Linear(30,20),
                    nn.ReLU(),
                    nn.Linear(20,10),
                    nn.ReLU(),
                    nn.Linear(10,size_of_output)
                    )
        if (type==14): # Smaller neural network of type 13
                self.mynn = nn.Sequential(
                     nn.Linear(size_of_input,40),
                     nn.ReLU(),
                     nn.Linear(40,30),
                     nn.ReLU(),
                     nn.Linear(30,20),
                     nn.ReLU(),
                     nn.Linear(20,10),
                     nn.ReLU(),
                     nn.Linear(10,size_of_output)
                      )
        if (type==15): # Bigger neural network version of type #12
            self.mynn = nn.Sequential(
                    nn.Linear(size_of_input,100),
                    nn.ReLU(),
                    nn.Linear(100,100),
                    nn.ReLU(),
                    nn.Linear(100,100),
                    nn.ReLU(),
                    nn.Linear(100,100),
                    nn.ReLU(),
                    nn.Linear(100,size_of_output)
                    )
        if(type==16): # Double Neurons
            self.mynn = nn.Sequential(
                nn.Linear(size_of_input,200),
                nn.ReLU(),
                nn.Linear(200,200),
                nn.ReLU(),
                nn.Linear(200,200),
                nn.ReLU(),
                nn.Linear(200,200),
                nn.ReLU(),
                nn.Linear(200,size_of_output)
                )
            
        if(type==17): # Triple Neurons
                self.mynn = nn.Sequential(
                    nn.Linear(size_of_input,300),
                    nn.ReLU(),
                    nn.Linear(300,300),
                    nn.ReLU(),
                    nn.Linear(300,300),
                    nn.ReLU(),
                    nn.Linear(300,300),
                    nn.ReLU(),
                    nn.Linear(300,size_of_output)
                    )
                    
        if(type==18): #Quadruple Neurons
                self.mynn = nn.Sequential(
                    nn.Linear(size_of_input,400),
                    nn.ReLU(),
                    nn.Linear(400,400),
                    nn.ReLU(),
                    nn.Linear(400,400),
                    nn.ReLU(),
                    nn.Linear(400,400),
                    nn.ReLU(),
                    nn.Linear(400,size_of_output)
                    )
                
                      
    def forward(self, x):
        res = self.mynn(x)
        return res


def load_model(file_name, device, size_of_input, size_of_output, nnType):
    """
    Load existing model state dict from file
    
    PARAMETERS 
    ---
    file_name - string - directory from where the trained neural network is found
    device - string - device from which the training is being done
    size_of_input - int 
    size_of_output - int 
    num_of_examples - int - size of dataset
    nnType - int - type of the neural network
    
    
    RETURNS
    ---
    model -- ff_network -- the loaded up model
    
    """
    model = ff_network(size_of_input, size_of_output, nnType).to(device=device)
    if (torch.cuda.is_available()):
        model.load_state_dict(torch.load(file_name))
    else:
        model.load_state_dict(torch.load(file_name, map_location=torch.device('cpu')))
    model.eval()
    return model


def prep_data(data, res, train_test_split, zeroInput = False, randomNoise = False):
    idx_traintest = int(len(data) * train_test_split)
    vals_train_np = data[0:idx_traintest]  # Input edges .. so these are our graphs
    # input_edges_train = input_edges[0:idx_traintest]
    res_train_np = res[0:idx_traintest]  # Output concurrence corresponding to each input graph
    vals_test_np = data[idx_traintest:]
    # input_edges_test = input_edges[idx_traintest:]
    res_test_np = res[idx_traintest:]
    if (zeroInput):
        vals_train_np=np.zeros(vals_train_np.shape)
        vals_test_np=np.zeros(vals_test_np.shape)
    if(randomNoise): # Introduce random noise onto the weights of graphs that exhibit high fidelity
        for ii in range(len(vals_train_np)):
            if res_train_np[ii] > 0.5:
                print("High Fidelity Example found! Adding random noise to weights...")
                vals_train_np[ii] = vals_train_np[ii] +  np.random.uniform(-0.005,0.005,len(vals_train_np[ii]))
    return vals_train_np, vals_test_np, res_train_np, res_test_np


def train_model(NN_INPUT_SIZE, NN_OUTPUT_SIZE, vals_train_np, res_train_np, vals_test_np, res_test_np, save_direc,
                suffix, config, l2Regularization=False, save_fig=False):
    """
    Trains the neural netowork. Implementation of Type L2 Regularization onto the training process is a WIP
    
    PARAMETERS 
    ---
    NN_INPUT_SIZE, NN_OUTPUT_SIZE -- int -- size of input vector and output vector of neural network, respectively
    vals_train_np, res_train_np -- np.array -- training data consisting of the input graph weights and the graph's corresponding fidelity wrt the desired state.
    vals_test_np, res_test_np -- np.array -- testing data on which the network is validated
    learnRate -- double -- the learning rate of the neural network
    save_direc -- string -- directory on which the trained neural network will be saved
    suffix -- string -- the suffix we append to the saving directories. It is used to distinguish among trained neural networks
    num_of_examples -- int -- size of dataset
    nnType -- int -- type of neural network (see initialization class) that we train
    l2Lambda -- double -- value of the lambda parameter in the L2 regularization process
    l2Regularization -- boolean -- toggles the application of L2 regularization to the training process
    
    """

    num_of_examples = int(float(config['num_of_examples']))  # Training set size
    learnRate = float(config['learnRate'])  # Learning rate
    l2Lambda = float(config['l2Lambda'])  # Lambda parameter for L2 Regularization
    epochToSaturate = float(config['epochToSaturate'])
    nnType = config['nnType']  # What type of neural network do we want to train on

    batch_size = min(len(vals_train_np), config['batch_size'])
    print('batch_size: ', batch_size)
    print('lr_enc: ', learnRate)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists('losses'):
        os.makedirs('losses')
    seed = config['seed']
    loss_file = f'losses/loss{seed}.csv'

    start_time = time.time()
    print("Training neural network...")
    model = ff_network(size_of_input=NN_INPUT_SIZE, size_of_output=NN_OUTPUT_SIZE, type=nnType).to(device)
    optimizer_predictor = torch.optim.Adam(model.parameters(), lr=learnRate)

    vals_train = torch.tensor(vals_train_np, dtype=torch.float).to(device)
    res_train = torch.tensor(res_train_np, dtype=torch.float).reshape([len(res_train_np), 1]).to(device)

    vals_test = torch.tensor(vals_test_np, dtype=torch.float).to(device)
    res_test = torch.tensor(res_test_np, dtype=torch.float).reshape([len(res_test_np), 1]).to(device)

    criterion = torch.nn.MSELoss()

    train_loss_evolution = []
    test_loss_evolution = []

    start_time = time.time()
    
    
    # Let's set up an adaptive learnimg rate 
    
    lmbda = lambda epoch: config['learnRateFactor']
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer_predictor, lr_lambda=lmbda,verbose=True)
    lrUpdate = config['epochLRUpdate']

    print('Everything prepared, lets train')

    for epoch in range(100000):  # should be much larger, with good early stopping criteria
        clamp_loss = 0
        model.train()

        x = [i for i in range(len(vals_train))]  # random shuffle input    

        random.shuffle(x)
        total_train_loss = 0
        num_episodes = int(len(vals_train) // batch_size)
        for batch in range(num_episodes):

            batch_indices = x[int(batch * batch_size):int((batch + 1) * batch_size)]
            vals_train_batch = vals_train[batch_indices].to(device)
            res_train_batch = res_train[batch_indices].to(device)

            prediction = model(vals_train_batch)
            real_loss = criterion(prediction, res_train_batch)
            # If L2 Regularization is turned on, modify the MSE loss accordingly
            if (l2Regularization):
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                real_loss = real_loss + l2Lambda * l2_norm

            clamp_loss = torch.clamp(real_loss, min=0., max=5000000.).double()
            total_train_loss += real_loss.detach().cpu().numpy()

            optimizer_predictor.zero_grad()
            clamp_loss.backward()
            optimizer_predictor.step()

        if epoch % 10 == 0:
            torch.save(model.state_dict(), save_direc)

        # Evaluating the current quality.
        with torch.no_grad():
            model.eval()
            train_loss = total_train_loss / num_episodes
            train_loss_evolution.append(train_loss)

            prediction = model(vals_test)
            test_loss = criterion(prediction, res_test).detach().cpu().numpy()
            test_loss_evolution.append(test_loss)

            if epoch % 1 == 0:
                print(str(epoch), ' - train: ', train_loss, '; test: ', test_loss, flush=True)
                with open(loss_file, 'a') as f:
                    writer = csv.writer(f, delimiter=";")
                    writer.writerow(
                        [train_loss, test_loss])

            if len(test_loss_evolution) - np.argmin(test_loss_evolution) > epochToSaturate: # This was set to 50 in the original code
                print('    Early stopping kicked in: test loss', np.argmin(test_loss_evolution),
                      len(test_loss_evolution))
                break

            if (time.time() - start_time) > 72 * 60 * 60:
                print('    Early stopping kicked in: too much time has elasped')
                break
            
            if epoch % lrUpdate and len(test_loss_evolution) - np.argmin(test_loss_evolution) > lrUpdate:
                print("The test loss doesn't seem to be changing much, so let's change the learn rate")
                scheduler.step()
                

        if save_fig and epoch % 50 == 0:
            plot_loss(num_of_examples, suffix, test_loss_evolution, train_loss_evolution, vals_test_np,
                  vals_train_np,epoch, config['plotFolder'])

    print('Best test MSE: ', min(test_loss_evolution))
    print("...DONE")
    print("--- %s seconds ---" % (time.time() - start_time))

    return True


def plot_loss(num_of_examples, suffix, test_loss_evolution, train_loss_evolution, vals_test_np,
              vals_train_np,epoch, plotFolder):
    plt.plot(train_loss_evolution, label='train')
    plt.clf()
    plt.plot(train_loss_evolution, label='train')
    plt.plot(test_loss_evolution, label='test')
    plt.yscale('log')
    plt.grid()
    plt.title(
        f'Evolution of train and test loss \nexamples -- train: {len(vals_train_np)} test: {len(vals_test_np)} \nBest test MSE: {min(test_loss_evolution)}')
    plt.ylabel('loss')
    plt.xlabel('episode')
    plt.legend(loc="lower left")
    print(plotFolder)
    plt.savefig(plotFolder+'/nn_train_results_' + str(num_of_examples) + '_' + suffix + '_'+str(epoch)+'.png')
    plt.show()


def neuron_selector(model, device, layer, neuron):
    '''
    This creates a new model with the same weights and biases as the input model
    but only includes the trained layers up to the layer containing the neuron
    we want to analyze.

    The way this'll work is that we initialize a new model consisting of every layer of the trained neural network
    up to the layer containing the neuron we want to analyze. We then create a new 'output layer' consisting solely of that particular neuron.
    The weights/biases of the neuron must be the same as it was in the original.
    '''

    total_model = list(model.mynn.children())

    old_output_layer = total_model[layer]  # The weights here are always the same! (as they should...)
    in_features = old_output_layer.in_features

    new_output_layer = nn.Linear(in_features, 1)  # The initial weights/biases here are random, 'uninitialized'values.

    with torch.no_grad():
        new_output_layer.weight[0] = old_output_layer.weight[neuron]
        new_output_layer.bias[0] = old_output_layer.bias[neuron]

    if (layer == 0):
        print(f"Inverse training on input layer and neuron {neuron}")
        new_model = new_output_layer.to(device)
    else:
        print(f"Inverse training on layer {layer}, neuron {neuron}")
        middle_model = total_model[:layer - 1]
        new_model = nn.Sequential(*middle_model, nn.ReLU(), new_output_layer).to(device)

    new_model.eval()

    return new_model


def dream_model(model, desired_state, start_graph, cnfg):
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
    dimensions = cnfg['dims']
    lr = float(cnfg['learnRate'])
    num_epochs = cnfg['num_of_epochs']
    layer_index = cnfg['layer']
    neuron_index = cnfg['neuron']

    loss_prediction = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prop = 1

    # data_train = edit_graph(data_train, upper_bound)

    graph_edge_weights = torch.tensor(np.array(list(start_graph.weights)), dtype=torch.float).to(device)
    data_train_prop = torch.tensor(prop, dtype=torch.float).to(device)
    data_train_var = torch.autograd.Variable(graph_edge_weights, requires_grad=True)

    # initialize list of intermediate property values and molecules
    fidelity_evolution = []
    activation_evolution = []
    gradDec = []
    interm_graph = [start_graph]

    epoch_transformed = [0]

    # initialize an instance of the model
    optimizer_encoder = torch.optim.Adam([data_train_var], lr=lr)
    interm_model = neuron_selector(model, device, layer_index, neuron_index)

    for epoch in range(num_epochs):

        # feedforward step
        activation = interm_model(data_train_var)
        activation_evolution.append(activation.cpu().detach().numpy())

        # mean squared error between target and calculated property
        activation = activation.reshape(1)
        # criterion = nn.MSELoss()
        # real_loss=criterion(activation, data_train_prop) # So we calculate the mean squared error between the predicted fidelity and the target one
        real_loss = -activation
        loss = torch.clamp(real_loss, min=-50000, max=50000.).double()
        # backpropagation step

        optimizer_encoder.zero_grad()
        loss.backward()
        optimizer_encoder.step()

        real_loss = loss.cpu().detach().numpy()
        loss_prediction.append(float(real_loss))

        input_grad = data_train_var.grad.cpu().detach().numpy()
        input_grad_norm = np.linalg.norm(input_grad, ord=2)
        gradDec.append(input_grad_norm)

        if epoch % 100 == 0:
            # We update our graph now with potentially new weight values and recompute the fidelity
            modified_edge_weights = data_train_var.cpu().detach().numpy()
            fidelity, dream_graph = constructGraph(modified_edge_weights, dimensions, desired_state)
            activation = interm_model(data_train_var).item()
            #print(f'epoch: {epoch} gradient: {input_grad_norm} fidelity {fidelity} activation {activation}', flush=True)
            with open(cnfg['dream_file'], 'a') as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow([fidelity, activation, dream_graph.weights])

        if len(gradDec) > 1000:
            if gradDec[-1] < 1e-7 and 0.99 * gradDec[-100] <= gradDec[-1]:
                print('The gradient is very near zero at this point, stop dreaming at epoch ', epoch)
                break
            else:
                if activation_evolution[-1] - activation_evolution[-100] < 1e-7:
                    print(
                        'Our predictions arent changing much, maybe our gradient is going back and forth? Stop dreaming at epoch ',
                        epoch)
                    break

    return 0
