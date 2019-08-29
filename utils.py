import numpy as np
from numpy import linalg as LA
import networkx as nx
import itertools
from multiprocessing import Pool
from matplotlib import pylab
import matplotlib.pyplot as plt

class GGLLoader():
    """
    A helper class for preprocessing chromatin geometric embedding data and graph theoretical analysis.
    """

    def __init__(self, sample_name, embedding_mode='pe', DEBUG=False):
        """
        sample_name: sample name that matches the name in keys.*.dat and coors.*.txt file.
        embedding_mode: the metric space employed during GGL training procedure, Poincare by default.
        """
        self.name = sample_name
        self.mode = embedding_mode
        self.graph = None
        self._debug = DEBUG
        self._load()

    def _load(self):
        keys_raw = np.expand_dims(np.loadtxt('data/keys.%s.dat' % self.name), 1)
        coors_raw = np.loadtxt('data/%s.coors.%s.txt' % (self.mode, self.name))
        self.data = np.concatenate([keys_raw, coors_raw], axis=1)
        if self._debug: print(self.data)
        return self.data

    def _update_edge(self, node1, node2, distance, cutoff=None):
        if cutoff == None or distance <= cutoff:
            return self.graph.add_edge(node1, node2, weight=distance)
        else:
            pass

    def to_graph(self, cutoff=None):
        if self.graph is not None:
            raise ValueError('Graph data already exists')
        self.graph = nx.Graph()

        for i in range(self.data.shape[0]):
            self.graph.add_node(self.data[i, 0], pos=(self.data[i, 1], self.data[i, 2]))

        if self.mode == 'pe':
            metric = self.pe_distance
        else:
            raise ValueError('Not implemented')

        #with Pool(processes=8) as pool: #TODO: multiprocessing memory sync issue
        for i, j in itertools.product(range(self.data.shape[0]), range(self.data.shape[0])):
            if i == j or (abs(self.data[i, 0] - self.data[j, 0]) <= 4): continue #Remove trivial connections
            distance = metric(self.data[i, 1:3], self.data[j, 1:3])
            self._update_edge(self.data[i, 0], self.data[j, 0], distance, cutoff=cutoff)

        print('Load finished!')
        if self._debug:
            print(nx.get_edge_attributes(self.graph, 'weight'))
        return self.graph

    def pe_distance(self, coor1, coor2):
        sqnorm1 = np.sum(np.power(coor1, 2))
        sqnorm2 = np.sum(np.power(coor2, 2))
        sqeuci_d = np.sum(np.power(coor1 - coor2, 2))
        pe_d = np.arccosh(1 + 2 * sqeuci_d /
                ((1 - sqnorm1) + (1 - sqnorm2)))
        return pe_d

    def mst(self):
        """
        Generate the minimum spanning tree for hierachy analysis
        """
        self.mst = nx.minimum_spanning_tree(self.graph)
        return self.mst

    def centrality(self, mode='betweenness'):
        """
        Compute centrality within the interaction network.
        """
        if mode == 'betweenness':
            cent = nx.algorithms.centrality.betweenness_centrality(self.graph, weight='weight', k=250)
        elif mode == 'eigenvector':
            cent = nx.algorithms.centrality.eigenvector_centrality(self.graph, weight='weight', max_iter=1000, tol=1e-6)
        elif mode == 'katz':
            cent = nx.algorithms.centrality.katz_centrality(self.graph, alpha=0.5, weight='weight', max_iter=1000, tol=1e-6)

        nx.set_node_attributes(self.graph, cent, 'centrality')

        return self.graph


def save_graph(G, path='./graph.gml'):
    nx.readwrite.gml.write_gml(G, path)

def load_graph(path):
    return nx.readwrite.gml.read_gml(path)

def save_graph_vis(graph, path='./test.png', cutoff=None, centrality=False):

    plt.figure(num=None, figsize=(20, 20), dpi=80)
    fig = plt.figure(1)
    plt.axis('on')
    pos = nx.get_node_attributes(graph, 'pos')
    weights = [d['weight'] for (u, v, d) in graph.edges(data=True)]
    if not centrality:
        nx.draw_networkx_nodes(graph, pos=pos, node_color='#A0CBE2')
    else:
        cents = [d['centrality'] for (n, d) in graph.nodes(data=True)]
        nx.draw_networkx_nodes(graph, pos=pos, node_color=cents,\
                vmin=0, vmax=max(cents), cmap=plt.cm.PuRd, alpha=0.9)

    nx.draw_networkx_edges(graph, pos=pos, edge_color=weights, \
            edge_cmap=plt.cm.magma, edge_vmin=0, edge_vmax=cutoff, alpha=0.05)
    plt.savefig(path, boox_inches="tight")
    pylab.close()
    del fig


if __name__ == '__main__':

    for sample in ['LG', 'LG-cancer', 'PA', 'PA-cancer']:
        #loader = GGLLoader(sample, DEBUG=False)
        #loader.graph = load_graph('data/%s.raw.gml' % sample)
        #G = loader.centrality('betweenness')
        G = load_graph('data/%s.BC.gml' % sample)
        #test_mst = test_loader.mst()
        save_graph_vis(G, path='vis/%s_BC.png' % sample, cutoff=0.5, centrality=True)
