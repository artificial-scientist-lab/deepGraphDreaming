# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 11:25:09 2023

@author: freem
"""
from neuralnet import prep_data
import csv
import pickle
import os
import numpy as np
import random
import yaml
from yaml import Loader
import pandas as pd
import re

seed = random.randint(1000, 9999)
print(f'seed: {seed}')

stream = open(f"configs/dataprep.yaml", 'r')
cnfg = yaml.load(stream, Loader=Loader)

num_of_examples = cnfg['numExamples']
fileName = cnfg['shuffleName']

if cnfg['datafile'].split('.')[-1] == 'pkl':
    # Load up the training dataset
    with open(cnfg['datafile'], 'rb') as f:
        data_full, res_full = pickle.load(f)

    data = data_full[0:num_of_examples]
    res = res_full[0:num_of_examples]
else:
    df = pd.read_csv(cnfg['datafile'], names=['weights', 'res'], delimiter=";")
    try:
        data = np.array([eval(graph) for graph in df['weights']])
    except:
        data = np.array(
            [eval(re.sub(r"  *", ',', graph.replace('\n', '').replace('[ ', '['))) for graph in df['weights']])
    res = df['res'].to_numpy()
    data = data[0:num_of_examples]
    print(data[0], flush=True)
    res = res[0:num_of_examples]
    print(res[0], flush=True)

shuffleInts = np.arange(0,num_of_examples)
np.random.shuffle(shuffleInts)
data = data[shuffleInts]
res = res[shuffleInts]

with open(fileName + '.csv', 'a') as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerows([data, res])


