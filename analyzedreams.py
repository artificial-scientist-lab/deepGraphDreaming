import os
import pandas as pd
from pytheus import theseus as th, graphplot as gp, fancy_classes as fc
import numpy as np

directory = 'dreamfiles/dream6q20n'
data = []
for ii, filename in enumerate(os.listdir(directory)):
    df = pd.read_csv(f'{directory}/{filename}', sep=";", names=['fidelity', 'activation', 'graph'])
    weights = np.array(eval(df.iloc[-1, 2]))
    startweights = np.array(eval(df.iloc[0, 2]))
    diff = weights - startweights
    data.append([weights, filename])
    edges = th.buildAllEdges(dimensions=6 * [2])
    graph = fc.Graph(edges=edges, weights=diff)
    # gp.graphPlot(graph)
    newweights = [w / max(graph.weights) for w in graph.weights]
    newgraph = fc.Graph(edges=graph.edges, weights=newweights)
    newgraph.purge(threshold=0.3, update=True)
    newgraph = fc.Graph(newgraph.edges, weights=newgraph.weights)
    gp.graphPlot(newgraph)
    gp.leiwandPlot(newgraph, name=f'leiwandplots/graph{ii}')
