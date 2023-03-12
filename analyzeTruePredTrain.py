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

dataset = 'distcont_4q_1M'
model = '4q_cont_type18_20M_5_seed3339'
df = pd.read_csv(f'{dataset}.csv', names=['weights', 'res'], delimiter=";", nrows=100000)
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
    if(len(DIM)==6):
        if edge in [(0, 1, 0, 0), (2, 3, 0, 0), (4, 5, 0, 0), (1, 2, 1, 1), (3, 4, 1, 1), (0, 5, 1, 1)]:
            input_graph[edge] = 1     
    else: 
        if (len(DIM)==4):
            if edge in [(0,1,0,0),(2,3,0,0),(1,2,1,1),(3,0,1,1)]:
                input_graph[edge] = 1
        

input_graph.getState()
input_graph.state.normalize()

# predict fidelity of baseline (which ideally should be 1)
input = torch.tensor(input_graph.weights, dtype=torch.float).to('cpu')
#x_train.append(float(mod(input)))
#x_test.append(float(mod(input)))

for ii in range(len(result_train)):
    input = torch.tensor(weights_train[ii], dtype=torch.float).to('cpu')
    x_train.append(float(mod(input)))
    y_train.append(result_train[ii])
    
xLine = np.linspace(np.min(x_train),np.max(x_train),100)
plt.scatter(x_train, y_train)
plt.plot(xLine,xLine)
plt.title('True Fidelity vs. Predicted Fidelity -- 20 M -- 6 Qubits -- Training Data')
plt.xlabel('Predicted Fidelity')
plt.ylabel('True Fidelity')
plt.savefig('R2_Plot_6Q_Train')
plt.clf()

# Now do the same for the test data 

for ii in range(len(result_test)):
    input = torch.tensor(weights_test[ii], dtype=torch.float).to('cpu')
    x_test.append(float(mod(input)))
    y_test.append(result_test[ii])


plt.scatter(x_test, y_test,color='red')
plt.plot(xLine,xLine,color='red')
plt.title('True Fidelity vs. Predicted Fidelity -- 20 M -- 6 Qubits -- Test Data')
plt.xlabel('Predicted Fidelity')
plt.ylabel('True Fidelity')
plt.savefig('R2_Plot_6Q_Test')


    
    


