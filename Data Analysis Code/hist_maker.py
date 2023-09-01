'''

This code takes a directory of dreamed graphs corresponding to each neuron 
and plots a histogram of the ket probabilities

'''
from pytheus import help_functions as hf, theseus as th, fancy_classes as fc, graphplot as gp
from neuralnet import load_model
from datagen import generatorGraphFidelity
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import glob 

import yaml 
from yaml import Loader

# Some functions for plotting the histograms 


# Loads up all the data coming from the graphs 

def loadAll(fileList):
    weightChange = []
    weightList = []
    for filename in fileList:
        df = pd.read_csv(filename, sep=";", names=['fidelity', 'activation', 'graph'])
        finalWeights = eval(df.iloc[-1, 2])
        finalWeights = [w / np.max(np.abs(finalWeights)) for w in finalWeights] # renormalization step
        weightList.append(finalWeights)
    return weightList

# This retrieves information about the resulting states from all the dreamed graphs
    
def getKets(fileList, edges):
    finalWeights = loadAll(fileList)
    dreamStates = []
    # Store all information about the ket amplitudes in seperate lists
    for subList in finalWeights:
        graph = fc.Graph(edges=edges, weights=subList)
        graph.getState()
        dreamState = graph.state
        dreamState.normalize()
        dreamStates.append(dreamState.state)
    return dreamStates

# This retrieves information about the kets and their corresponding values from each graph.  

def extractAllKets(fileList, edges, isProb):
    kets = getKets(fileList, edges)
    totalKets = {}
    for ket in kets[0].keys():
        ketList = []
        for k in kets:
            if(isProb):
                ketList.append(np.abs(k[ket])**2)
            else:
                ketList.append(k[ket])
        totalKets[ket] = ketList
    return totalKets

# Adds them up, computes variance, then plots that over histogram 
def plotKetHist(fileList, edges, neuron, cnfg):
    # Load up relevant parameters
    isProb = cnfg['isProb']
    saveDirec = cnfg['saveDirec']
    # Obtain total ket dictionary 
    totalKets = extractAllKets(fileList, edges, isProb)
    # Compute mean/variance for each ket
    meanKet = {}
    var = []
    for ket in totalKets.keys():
        meanKet[ket] = np.mean(totalKets[ket])
        var.append(np.var(totalKets[ket]))
    #Let's get ready to plot those histograms
    x_ticks = []
    for ket in totalKets.keys():
        x_ticks.append(f"|{ket[0][1]},{ket[1][1]},{ket[2][1]},{ket[3][1]}>")
    meanKetVal = list(meanKet.values())
    plt.figure(figsize=(15,15))
    plt.title(f"Histogram of Ket Probabilities (Neuron {neuron})")
    plt.bar(range(len(meanKetVal)), meanKetVal, width = 0.5, tick_label = x_ticks, yerr=var, color='red')
    plt.savefig(f"{saveDirec}/hist_{neuron}")
    
# Loads up configuration file 

stream = open("configs/anaDream.yaml", 'r')
cnfg = yaml.load(stream, Loader=Loader)

# Some required parameters

directory = cnfg['directory'] # loading directory

# Load up the csv files corresponding to the dreamed graph neuron by neuron 
# We implement a 'filelist' for each neuron with the list of csv files. 

data = []
neuron_indices = eval(cnfg['neurons'])
fileLists = []

for neuron in neuron_indices:
    fileLists.append(glob.glob(f"{directory}/*_{neuron}.csv"))

# So fileList[neuron] gives us a list of csv files for that neuron. 

# We now proceed to start generating histograms for each neuron using the complete
# quad-partite graph as reference

DIM = eval(cnfg['DIM'])
edges = th.buildAllEdges(dimensions=DIM)

for neuron in neuron_indices: 
    plotKetHist(fileLists[neuron], edges, neuron, cnfg)




