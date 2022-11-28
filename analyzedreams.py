import os
import pandas as pd
from pytheus import theseus as th, graphplot as gp, fancy_classes as fc

directory = 'dreamfiles/dream6q20n'
data = []
for filename in os.listdir(directory):
    df = pd.read_csv(f'{directory}/{filename}', sep=";", names=['fidelity', 'activation', 'graph'])
    weights = eval(df.iloc[-1, 2])
    data.append([weights, filename])
    edges = th.buildAllEdges(dimensions=6 * [2])
    graph = fc.Graph(edges=edges, weights=weights)
    # gp.graphPlot(graph)
    newweights = [w / max(graph.weights) for w in graph.weights]
    newgraph = fc.Graph(edges=graph.edges, weights=newweights)
    newgraph.purge(threshold=6e-1, update=True)
    newgraph = fc.Graph(newgraph.edges, weights=newgraph.weights)
    gp.graphPlot(newgraph)
