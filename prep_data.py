# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:13:08 2023

@author: freem
"""

import numpy as np
import pandas as pd
import yaml
from yaml import Loader
import re
import csv

from neuralnet import prep_data

stream = open("configs/dataprep.yaml", 'r')
cnfg = yaml.load(stream, Loader=Loader)

dataset = cnfg['dataset']
indices = np.load(cnfg['shuffle'])


df = pd.read_csv(f'{dataset}.csv', names=['weights', 'res'], delimiter=";")
try:
    weights = np.array([eval(graph) for graph in df['weights']])
except:
    weights = np.array(
        [eval(re.sub(r"  *", ',', graph.replace('\n', '').replace('[ ', '['))) for graph in df['weights']])
res = df['res'].to_numpy()


weights = weights[indices]
res = res[indices]

weights_train, weights_test, res_train, res_test = prep_data(weights, res, 0.95, zeroInput=False)

with open(f"{dataset}_train.csv", 'a') as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerows([weights_train, res_train])

with open(f"{dataset}_test.csv", 'a') as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerows([weights_test, res_test])







