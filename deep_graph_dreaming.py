import numpy as np
import random
import time

import torch
from torch import nn

import matplotlib.pyplot as plt


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


def load_model(file_name, device, size_of_input, size_of_output, num_of_examples, nnType):
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


def prep_data(num_of_examples, data, res, train_test_split):
    idx_traintest = int(len(data) * train_test_split)
    vals_train_np = data[0:idx_traintest]  # Input edges .. so these are our graphs
    # input_edges_train = input_edges[0:idx_traintest]
    res_train_np = res[0:idx_traintest]  # Output concurrence corresponding to each input graph

    vals_test_np = data[idx_traintest:]
    # input_edges_test = input_edges[idx_traintest:]
    res_test_np = res[idx_traintest:]
    return vals_train_np, vals_test_np, res_train_np, res_test_np


def train_model(NN_INPUT_SIZE, NN_OUTPUT_SIZE, vals_train_np, res_train_np, vals_test_np, res_test_np, learnRate,
                save_direc, suffix, num_of_examples, nnType, l2Lambda=0.001, l2Regularization=False):
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

    batch_size = min(len(vals_train_np),
                     5e3)  # Large batch_size seems to be important (smaller number of examples the greater the number of batches)
    print('batch_size: ', batch_size)
    print('lr_enc: ', learnRate)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    start_time = time.time()
    print("Training neural network...")
    modelFidelity = ff_network(size_of_input=NN_INPUT_SIZE, size_of_output=NN_OUTPUT_SIZE, type=nnType).to(device)
    optimizer_predictor = torch.optim.Adam(modelFidelity.parameters(), lr=learnRate)

    vals_train = torch.tensor(vals_train_np, dtype=torch.float).to(device)
    res_train = torch.tensor(res_train_np, dtype=torch.float).reshape([len(res_train_np), 1]).to(device)

    vals_test = torch.tensor(vals_test_np, dtype=torch.float).to(device)
    res_test = torch.tensor(res_test_np, dtype=torch.float).reshape([len(res_test_np), 1]).to(device)

    criterion = torch.nn.MSELoss()

    train_loss_evolution = []
    test_loss_evolution = []

    start_time = time.time()

    print('Everything prepared, lets train')

    for episode in range(100000):  # should be much larger, with good early stopping criteria
        modelFidelity.train()
        clamp_loss = 0

        x = [i for i in range(len(vals_train))]  # random shuffle input    

        random.shuffle(x)
        total_train_loss = 0
        num_of_iterations = int(len(vals_train) / batch_size)
        for idx in range(num_of_iterations):

            curr_idxs = x[int(idx * batch_size):int((idx + 1) * batch_size)]
            vals_train_batch = vals_train[curr_idxs].to(device)
            res_train_batch = res_train[curr_idxs].to(device)

            calc_properties = modelFidelity(vals_train_batch)
            real_loss = criterion(calc_properties, res_train_batch)
            # If L2 Regularization is turned on, modify the MSE loss accordingly
            if (l2Regularization):
                l2_norm = sum(p.pow(2.0).sum() for p in modelFidelity.parameters())
                real_loss = real_loss + l2Lambda * l2_norm

            # real_loss = criterion(calc_properties, res_train)
            clamp_loss = torch.clamp(real_loss, min=0., max=5000000.).double()
            total_train_loss += real_loss.detach().cpu().numpy()

            optimizer_predictor.zero_grad()
            clamp_loss.backward()
            optimizer_predictor.step()

        # Evaluating the current quality.
        with torch.no_grad():
            modelFidelity.eval()
            train_loss = total_train_loss / num_of_iterations
            train_loss_evolution.append(train_loss)

            calc_properties = modelFidelity(vals_test)
            test_loss = criterion(calc_properties, res_test).detach().cpu().numpy()
            test_loss_evolution.append(test_loss)

            if episode % 5 == 0:
                print(str(episode), ' - train: ', train_loss, '; test: ', test_loss)

            # We stop the training process if the test MSE on the neural network does not reach a further minimum past 500 steps, or if the training has been running for 5 hours
            if len(test_loss_evolution) - np.argmin(test_loss_evolution) > 500:
                print('    Early stopping kicked in: test loss', np.argmin(test_loss_evolution),
                      len(test_loss_evolution))
                break

            if (time.time() - start_time) > 5 * 60 * 60:
                print('    Early stopping kicked in: too much time has elasped')
                break
                # Plot the loss evolution of the training process and save it as a png
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
    print('Best test MSE: ', min(test_loss_evolution))
    plt.savefig('nn_train_results_' + str(num_of_examples) + suffix + '_.png')

    # Save thr model as a png
    torch.save(modelFidelity.state_dict(), save_direc)
    print("...DONE")
    print("Wake up, sleepyhead!")
    print("--- %s seconds ---" % (time.time() - start_time))

    return True
