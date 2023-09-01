# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 14:39:52 2023

@author: freem
"""

from pytheus import help_functions as hf, theseus as th, fancy_classes as fc, graphplot as gp
from neuralnet import prep_data, load_model
from datagen import generatorGraphFidelity
import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
import re
from scipy import stats
import os
import argparse
import yaml
from yaml import Loader
import random


# NEEDS TO BE TESTED * 

# Loads up the training and test data. This assumes that the dataset corresponding to the trained neural network
# has been prepared using dataprep. 

def prep_data(cnfg, isTest):
    if(isTest):
        df = pd.read_csv(f"{cnfg['dataTest']}.csv", names=['weights','res'],delimiter=";")
    else:
        df = pd.read_csv(f"{cnfg['dataTrain']}.csv", names=['weights', 'res'], delimiter=";")
    try:
        weights = np.array([eval(graph) for graph in df['weights']])
    except:
        weights = np.array(
            [eval(re.sub(r"  *", ',', graph.replace('\n', '').replace('[ ', '['))) for graph in df['weights']])
    res = df['res'].to_numpy()
    return weights,res
        
    
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# parse through slurm array
parser = argparse.ArgumentParser()
parser.add_argument(dest='ii')
args = parser.parse_args()
shift = int(args.ii)

# Load up yaml files
print(shift)
stream = open(f"configs/analyze{shift}.yaml", 'r')
cnfg = yaml.load(stream, Loader=Loader)

weights_train, result_train = prep_data(cnfg,False)
weights_test, result_test = prep_data(cnfg,True)

nn_type = cnfg['nnType']
print(nn_type)
model = cnfg['model']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

mod = load_model(f'models/{model}.pt', 'cpu', 24, 1, nn_type)
DIM = [2] * 4
kets = hf.makeState('0000+1111')
state = fc.State(kets, normalize=True)
input_graph, ket_amplitudes, output_fidelity = generatorGraphFidelity(DIM, state, short_output=False)

#####
x_train = []
y_train = []
#####
x_test = []
y_test = []
#####
for edge in input_graph.edges:
    input_graph[edge] = 0
    if len(DIM) == 6:
        if edge in [(0, 1, 0, 0), (2, 3, 0, 0), (4, 5, 0, 0), (1, 2, 1, 1), (3, 4, 1, 1), (0, 5, 1, 1)]:
            input_graph[edge] = 1

    else:
        if len(DIM)==4:
            if edge in [(0, 1, 0, 0), (2, 3, 0, 0), (1, 2, 1, 1), (0, 3, 1, 1)]:
                input_graph[edge] = 1

input_graph.getState()
input_graph.state.normalize()
nnTitle = cnfg['name']


for ii in range(1000):
    input = torch.tensor(weights_train[ii], dtype=torch.float).to(device)
    x_train.append(float(mod(input)))
    y_train.append(result_train[ii])

# We compute the r^2 coefficient of a hypothetical linear fit between model predictions x and ground truths y 
 
slope, intercept, r_value, p_value, std_err = stats.linregress(x_train, y_train)

print("Results on training data")
print(f"Slope:{slope}, intercept:{intercept}, r_value:{r_value}")

 # predict fidelity of baseline (which ideally should be 1)
input = torch.tensor(input_graph.weights, dtype=torch.float).to(device)

xLine = np.linspace(0,1)
plt.scatter(x_train, y_train)
plt.scatter(float(mod(input)),1,color='green')
plt.plot(xLine*slope,xLine)
plt.title(f'True Fidelity vs. Predicted Fidelity -- {nnTitle} -- Training Data \n r^2 value:{r_value**2}')
plt.xlabel('Predicted Fidelity')
plt.ylabel('True Fidelity')
plt.savefig(f'R2_Plot_4Q_Train{model}')
plt.clf()


# Now do the same for the test data 

for ii in range(1000):
    input = torch.tensor(weights_test[ii], dtype=torch.float).to(device)
    x_test.append(float(mod(input)))
    y_test.append(result_test[ii])


slope, intercept, r_value, p_value, std_err = stats.linregress(x_test, y_test)
input = torch.tensor(input_graph.weights, dtype=torch.float).to(device)
print("Results on training data")
print(f"Slope:{slope}, intercept:{intercept}, r_value:{r_value}")

plt.scatter(x_test, y_test,color='red')
plt.scatter(float(mod(input)),1,color='green')
plt.plot(xLine*slope,xLine,color='red')
plt.title(f'True Fidelity vs. Predicted Fidelity -- {nnTitle} -- Test Data \n r^2 value:{r_value**2}')
plt.xlabel('Predicted Fidelity')
plt.ylabel('True Fidelity')
plt.savefig(f'R2_Plot_4Q_Test_{model}')
plt.clf()