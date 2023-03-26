# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 12:26:08 2023

@author: Toaster
"""

import numpy as np
import pandas as pd
import yaml
from yaml import Loader
import re
import csv
import os

from neuralnet import prep_data

stream = open("configs/dataprep.yaml", 'r')
cnfg = yaml.load(stream, Loader=Loader)

dataset = cnfg['dataset']
indices = np.load(cnfg['shuffle'])
num_of_examples = eval(cnfg['num_of_examples'])
seed = cnfg['seed']


df = pd.read_csv(f'{dataset}.csv', names=['weights', 'res'], delimiter=";", nrows=num_of_examples)
try:
    weights = np.array([eval(graph) for graph in df['weights']])
except:
    weights = np.array(
        [eval(re.sub(r"  *", ',', graph.replace('\n', '').replace('[ ', '['))) for graph in df['weights']])
res = df['res'].to_numpy()


weights = weights[indices]
res = res[indices]

weights_train, weights_test, res_train, res_test = prep_data(weights, res, 0.95, zeroInput=False)
###########################################

for ii in range(len(res_train)):
    with open(os.getcwd() + f"\\{dataset}_train_{seed}.csv", 'a') as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([list(weights_train[ii]), res_train[ii]])

for ii in range(len(res_test)):
    with open(os.getcwd() + f"\\{dataset}_test_{seed}.csv", 'a') as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([list(weights_test[ii]), res_test[ii]])

##########################################
        