""" Plot evolved network.

This module contains functions for plotting the evolved network with diameters
or flow as edge width as well as some additional data to better understand the
simulation.

Notable functions
-------
uniform_hist(SimInputData, Graph, Edges, np.ndarray, np.ndarray, str, str) \
    -> None
    plot the network and histograms of key data
"""

from matplotlib import gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from config import SimInputData
from delaunay import Graph
from incidence import Edges


def uniform_hist(sid: SimInputData, graph: Graph, edges: Edges, \
    cb: np.ndarray, cc: np.ndarray, name: str, data: str) -> None:
    """ Draw the network with diameters/flow as edge width.

    This function plots the network with one of parameters as edge width, as
    well as some histograms of key data.

    Parameters
    -------
    sid : SimInputData
        all config parameters of the simulation
        figsize - size of the plot (~resolution)
        ddrawconst - scaling parameter to improve visibility when drawing
        diameters
        qdrawconst - scaling parameter to improve visibility when drawing flow
        dirname - directory of the simulation

    graph : Graph class object
        network and all its properties
        in_nodes - list of inlet nodes
        out_nodes - list of outlet nodes

    edges : Edges class object
        all edges in network and their parameters
        diams - diameters of edges
        diams_initial - initial diameters of edges
        flow - flow in edges
        boundary_list - edges assuring PBC (to be excluded from drawing)

    cb : numpy ndarray
        vector of current B concentration

    cc : numpy ndarray
        vector of current C concentration

    name : str
        name of the saved file with the plot

    data : str
        parameter taken as edge width (diameter or flow)
    """
    cols = 4
    plt.figure(figsize=(sid.figsize * 1.25, sid.figsize))
    spec = gridspec.GridSpec(ncols = cols, nrows = 2, height_ratios = [5, 1])
    # draw first panel for the network
    plt.subplot(spec.new_subplotspec((0, 0), colspan = cols))
    plt.axis('equal')
    pos = nx.get_node_attributes(graph, 'pos')
    # draw inlet and outlet nodes
    x_in, y_in = [], []
    for node in graph.in_nodes:
        x_in.append(pos[node][0])
        y_in.append(pos[node][1])
    x_out, y_out = [], []
    for node in graph.out_nodes:
        x_out.append(pos[node][0])
        y_out.append(pos[node][1])
    plt.scatter(x_in, y_in, s = 60, facecolors = 'white', edgecolors = 'black')
    plt.scatter(x_out, y_out, s = 60, facecolors = 'black', \
        edgecolors = 'white')
    if data == 'd':
        qs1 = (1 - edges.boundary_list) * edges.diams \
            * (edges.diams < edges.diams_initial / 2)
        qs2 = (1 - edges.boundary_list) * edges.diams \
            * (edges.diams >= edges.diams_initial / 2)
        nx.draw_networkx_edges(graph, pos, edge_color = 'r', \
            width = sid.ddrawconst * np.array(qs1))
        nx.draw_networkx_edges(graph, pos, edge_color = 'k', \
            width = sid.ddrawconst * np.array(qs2))
    elif data == 'q':
        qs = (1 - edges.boundary_list) * np.abs(edges.flow)
        nx.draw_networkx_edges(graph, pos, edge_color = 'k', \
            width = sid.ddrawconst * np.array(qs))
    # draw histograms with data below the network
    plt.subplot(spec[cols]).set_title('Diameter')
    plt.hist(edges.diams, bins = 50)
    plt.yscale("log")
    plt.subplot(spec[cols + 1]).set_title('Flow')
    plt.hist(np.abs(edges.flow), bins = 50)
    plt.yscale("log")
    plt.subplot(spec[cols + 2]).set_title('cb')
    plt.hist(cb, bins = 50)
    plt.yscale("log")
    plt.subplot(spec[cols + 3]).set_title('cc')
    plt.hist(cc, bins = 50)
    plt.yscale("log")
    # save file in the directory
    plt.savefig(sid.dirname + "/" + name)
    plt.close()
