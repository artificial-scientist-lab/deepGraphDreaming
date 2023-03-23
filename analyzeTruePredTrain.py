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


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# parse through slurm array
parser=argparse.ArgumentParser(description='test')
parser.add_argument('--ii', dest='ii', type=int,
    default=None, help='')
args = parser.parse_args()
shift = args.ii
print(shift)

# Load up yaml files
stream = open(f"configs/analyze{shift}.yaml", 'r')
cnfg = yaml.load(stream, Loader=Loader)

dataset = cnfg['dataset']
model = cnfg['model']
num_of_examples = cnfg['num_of_examples']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

df = pd.read_csv(f'{dataset}.csv', names=['weights', 'res'], delimiter=";")
try:
    weights = np.array([eval(graph) for graph in df['weights']])
except:
    weights = np.array(
        [eval(re.sub(r"  *", ',', graph.replace('\n', '').replace('[ ', '['))) for graph in df['weights']])
res = df['res'].to_numpy()

mod = load_model(f'models/{model}.pt', 'cpu', 24, 1, 18)
DIM = [2] * 4
kets = hf.makeState('0000+1111')
state = fc.State(kets, normalize=True)
input_graph, ket_amplitudes, output_fidelity = generatorGraphFidelity(DIM, state, short_output=False)

# Shuffle datapoints
shuffleInts = np.arange(0,num_of_examples)
np.random.shuffle(shuffleInts)
weights = weights[shuffleInts]
res = res[shuffleInts]

weights_train, weights_test, result_train, result_test = prep_data(weights, res, 0.95, zeroInput=False)
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

# predict fidelity of baseline (which ideally should be 1)
input = torch.tensor(input_graph.weights, dtype=torch.float).to(device)
x_train.append(float(mod(input)))
x_test.append(float(mod(input)))
y_train.append(1)
y_test.append(1)

for ii in range(len(result_train)):
    input = torch.tensor(weights_train[ii], dtype=torch.float).to(device)
    x_train.append(float(mod(input)))
    y_train.append(result_train[ii])

# We compute the r^2 coefficient of a hypothetical linear fit between model predictions x and ground truths y 

slope, intercept, r_value, p_value, std_err = stats.linregress(x_train, y_train)

print("Results on training data")
print(f"Slope:{slope}, intercept:{intercept}, r_value:{r_value}")

xLine = np.linspace(np.min(x_train),np.max(x_train))
plt.scatter(x_train, y_train)
plt.plot(xLine*slope,xLine)
plt.title(f'True Fidelity vs. Predicted Fidelity -- 20 M -- 4 Qubits -- Training Data \n r^2 value:{r_value**2}')
plt.xlabel('Predicted Fidelity')
plt.ylabel('True Fidelity')
plt.savefig(f'R2_Plot_4Q_{model}')
plt.clf()


# Now do the same for the test data 

for ii in range(len(result_test)):
    input = torch.tensor(weights_test[ii], dtype=torch.float).to(device)
    x_test.append(float(mod(input)))
    y_test.append(result_test[ii])


slope, intercept, r_value, p_value, std_err = stats.linregress(x_test, y_test)

print("Results on training data")
print(f"Slope:{slope}, intercept:{intercept}, r_value:{r_value}")


plt.scatter(x_test, y_test,color='red')
plt.plot(xLine*slope,xLine,color='red')
plt.title(f'True Fidelity vs. Predicted Fidelity -- 20 M -- 4 Qubits -- Test Data \n r^2 value:{r_value**2}')
plt.xlabel('Predicted Fidelity')
plt.ylabel('True Fidelity')
plt.savefig(f'R2_Plot_4Q_Test_{model}')