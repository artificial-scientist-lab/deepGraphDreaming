import io
import zipfile

import numpy as np
import random
import time

import torch
from pytheus import graphplot as gp
from torch import nn

import matplotlib.pyplot as plt

from datagen import constructGraph


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


def prep_data(data, res, train_test_split):
    idx_traintest = int(len(data) * train_test_split)
    vals_train_np = data[0:idx_traintest]  # Input edges .. so these are our graphs
    # input_edges_train = input_edges[0:idx_traintest]
    res_train_np = res[0:idx_traintest]  # Output concurrence corresponding to each input graph

    vals_test_np = data[idx_traintest:]
    # input_edges_test = input_edges[idx_traintest:]
    res_test_np = res[idx_traintest:]
    return vals_train_np, vals_test_np, res_train_np, res_test_np


def train_model(NN_INPUT_SIZE, NN_OUTPUT_SIZE, vals_train_np, res_train_np, vals_test_np, res_test_np, learnRate,
                save_direc, suffix, num_of_examples, nnType, batch_size, l2Lambda=0.001, l2Regularization=False,
                save_fig=False):
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

    batch_size = min(len(vals_train_np), batch_size)
    print('batch_size: ', batch_size)
    print('lr_enc: ', learnRate)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    print('Everything prepared, lets train')

    for epoch in range(100000):  # should be much larger, with good early stopping criteria
        clamp_loss = 0
        model.train()

        x = [i for i in range(len(vals_train))]  # random shuffle input    

        random.shuffle(x)
        total_train_loss = 0
        num_episodes = int(len(vals_train) / batch_size)
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

            if len(test_loss_evolution) - np.argmin(test_loss_evolution) > 50:
                print('    Early stopping kicked in: test loss', np.argmin(test_loss_evolution),
                      len(test_loss_evolution))
                break

            if (time.time() - start_time) > 24 * 60 * 60:
                print('    Early stopping kicked in: too much time has elasped')
                break
    if save_fig:
        plot_loss(num_of_examples, suffix, test_loss_evolution, train_loss_evolution, vals_test_np,
                  vals_train_np)

    print('Best test MSE: ', min(test_loss_evolution))
    print("...DONE")
    print("--- %s seconds ---" % (time.time() - start_time))

    return True


def plot_loss(num_of_examples, suffix, test_loss_evolution, train_loss_evolution, vals_test_np,
              vals_train_np):
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
    plt.show()
    plt.savefig('nn_train_results_' + str(num_of_examples) + suffix + '_.png')


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
        print("gobble gobble")
        new_model = new_output_layer.to(device)
    else:
        print(f"gobble gobble {layer}")
        middle_model = total_model[:layer - 1]
        new_model = nn.Sequential(*middle_model, nn.ReLU(), new_output_layer).to(device)

    new_model.eval()

    return new_model


def dream_model(dimensions, model, desired_state, data_train, lr, num_epochs, name_of_zip, layer_index,
                neuron_index, display=True):
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

    loss_prediction = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prop = 1

    # data_train = edit_graph(data_train, upper_bound)

    graph_edge_weights = torch.tensor(np.array(list(data_train.values())), dtype=torch.float).to(device)
    data_train_prop = torch.tensor(prop, dtype=torch.float).to(device)
    data_train_var = torch.autograd.Variable(graph_edge_weights, requires_grad=True)

    # initiailize list of intermediate property values and molecules
    interm_prop = []
    nn_prop = []
    gradDec = []
    interm_graph = [data_train]

    epoch_transformed = [0]
    steps = 0
    valid_steps = 0

    # initialize an instance of the model
    optimizer_encoder = torch.optim.Adam([data_train_var], lr=lr)
    interm_model = neuron_selector(model, device, layer_index, neuron_index)

    for epoch in range(num_epochs):

        # feedforward step

        calc_properties = interm_model(data_train_var)
        nn_prop.append(calc_properties.cpu().detach().numpy())

        # mean squared error between target and calculated property
        calc_properties = calc_properties.reshape(1)
        # criterion = nn.MSELoss()
        # real_loss=criterion(calc_properties, data_train_prop) # So we calculate the mean squared error between the predicted fidelity and the target one
        real_loss = -calc_properties
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
            print('epoch: ', epoch, ', gradient: ', input_grad_norm)

        # We update our graph now with potentially new weight values and recompute the fidelity
        neo_edge_weights = data_train_var.cpu().detach().numpy()
        fidelity, edge_weights = constructGraph(neo_edge_weights, dimensions, desired_state)

        if len(interm_prop) == 0 or interm_prop[len(interm_prop) - 1] != fidelity:

            # collect intermediate graphs
            interm_graph.append(edge_weights)
            interm_prop.append(fidelity)

            steps += 1
            epoch_transformed.append(epoch + 1)

            if len(interm_prop) > 1:

                # determine validity of transformation
                previous_prop = interm_prop[len(interm_prop) - 2]
                current_prop = fidelity

                valid = (prop > previous_prop and current_prop > previous_prop) \
                        or (prop < previous_prop and current_prop < previous_prop)

                if valid:
                    valid_steps += 1

        if len(gradDec) > 1000:
            if gradDec[-1] < 1e-7 and 0.99 * gradDec[-100] <= gradDec[-1]:
                print('The gradient is very near zero at this point, stop dreaming at epoch ', epoch)
                break
            else:
                if nn_prop[-1] - nn_prop[-100] < 1e-7:
                    print(
                        'Our predictions arent changing much, maybe our gradient is going back and forth? Stop dreaming at epoch ',
                        epoch)
                    break

    # Make a plot for the intermediate graph and save in a zip file.

    if display:
        print("Creating archive: {:s}".format(name_of_zip))

        with zipfile.ZipFile(name_of_zip, mode="w") as zf:
            for i in range(0, len(interm_graph), int(len(interm_graph) / 10)):
                # First, we reformat the interm graph into something compatible with the graph plotting functions
                graph_to_go = []

                temp = interm_graph[i]
                temp_keys = list(temp.keys())
                temp_vals = list(temp.values())
                for j in range(len(temp_keys)):
                    graph_to_go.append(temp_keys[j] + tuple([temp_vals[j]]))

                # Now save the interm plot to the zip
                interm_graph_plot = gp.graphPlot(graph_to_go, 1, i, interm_prop[j], scaled_weights=True, show=False,
                                                 max_thickness=10, multiple_graphs=True)
                # input()
                buf = io.BytesIO()
                interm_graph_plot.savefig(buf)
                img_name = "graph_fig_{:02d}.png".format(i)
                zf.writestr(img_name, buf.getvalue())

    percent_valid_transform = None

    if steps > 0:
        percent_valid_transform = valid_steps / steps * 100

    return interm_prop[
               -1], interm_graph, loss_prediction, interm_prop, nn_prop, gradDec, percent_valid_transform, epoch_transformed
