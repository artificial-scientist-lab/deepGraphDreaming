# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 11:13:34 2022

@author: freem
"""

import csv
import os
import numpy as np
import pandas as pd
import pickle
import yaml
from yaml import Loader

from neuralnet import plot_loss, prep_data

loss = pd.read_csv('losses/loss111.csv', names=['train_loss', 'test_loss'], delimiter=";")
stream = open("configs/analyze.yaml", 'r')
cnfg = yaml.load(stream, Loader=Loader)

num_of_examples = int(float(cnfg['num_of_examples']))
suffix = cnfg['model_suffix']

with open(cnfg['datafile'], 'rb') as f:
    data_full, res_full = pickle.load(f)
    data = data_full[0:num_of_examples]
    res = res_full[0:num_of_examples]

# def plot_loss(num_of_examples, suffix, test_loss_evolution, train_loss_evolution, vals_test_np,
             # vals_train_np)
             
vals_train_np,vals_test_np,*_ = prep_data(data,res,0.95)
plot_loss(num_of_examples,suffix,loss['train_loss'], loss['test_loss'], vals_test_np, vals_train_np)




