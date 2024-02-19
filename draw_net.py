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
from data import Data
from network import Edges, Graph
from incidence import Incidence


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
        nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'r', \
            width = sid.ddrawconst * np.array(qs1))
        nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'k', \
            width = sid.ddrawconst * np.array(qs2))
    elif data == 'q':
        qs = (1 - edges.boundary_list) * np.abs(edges.flow)
        nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'k', \
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


def draw(sid: SimInputData, graph: Graph, edges: Edges, \
    name: str, data: str) -> None:
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

    name : str
        name of the saved file with the plot

    data : str
        parameter taken as edge width (diameter or flow)
    """
    # draw first panel for the network
    plt.axis('equal')
    
    plt.figure(figsize=(sid.figsize, sid.figsize))
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
    x_zero, y_zero = [], []
    for node in graph.zero_nodes:
        x_zero.append(pos[node][0])
        y_zero.append(pos[node][1])
    #print (x_zero, y_zero)
    plt.scatter(x_in, y_in, s = 30, facecolors = 'white', edgecolors = 'black')
    plt.scatter(x_out, y_out, s = 30, facecolors = 'black', \
        edgecolors = 'white')
    
    if data == 'd':
        qs1 = (1 - edges.boundary_list) * edges.diams \
            * (edges.diams < edges.diams_initial / 2)
        qs2 = (1 - edges.boundary_list) * edges.diams \
            * (edges.diams >= edges.diams_initial / 2)
        nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'r', \
            width = sid.ddrawconst * np.array(qs1))
        nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'k', \
            width = sid.ddrawconst * np.array(qs2))
    elif data == 'q':
        qs = (1 - edges.boundary_list) * np.abs(edges.flow)
        nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'k', \
            width = sid.qdrawconst * np.array(qs))
    #nx.draw_networkx_edge_labels(graph, pos, edge_labels=dict(zip(edges.edge_list, np.arange(0, len(edges.edge_list)))), font_size = 5)
    #plt.scatter(x_zero, y_zero, s = 60, facecolors = 'blue', edgecolors = 'black')
    # save file in the directory
    plt.axis('off')
    plt.savefig(sid.dirname + "/" + name)
    plt.close()


def draw_nodes(sid: SimInputData, graph: Graph, edges: Edges, cb, \
    name: str, data: str) -> None:
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

    name : str
        name of the saved file with the plot

    data : str
        parameter taken as edge width (diameter or flow)
    """
    # draw first panel for the network
    plt.axis('equal')
    
    plt.figure(figsize=(sid.figsize, sid.figsize))
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
    x_zero, y_zero = [], []
    for node in graph.zero_nodes:
        x_zero.append(pos[node][0])
        y_zero.append(pos[node][1])
    #print (x_zero, y_zero)
    nx.draw_networkx_nodes(graph, pos, node_color = cb * (cb >= 1))
    nx.draw_networkx_labels(graph, pos, labels=dict(zip(graph.nodes(), graph.nodes())), font_size=5)
    plt.scatter(x_in, y_in, s = 30, facecolors = 'white', edgecolors = 'black')
    plt.scatter(x_out, y_out, s = 30, facecolors = 'black', \
        edgecolors = 'white')
    plt.scatter(x_zero, y_zero, s = 60, facecolors = 'blue', edgecolors = 'black')
    # save file in the directory
    plt.axis('off')
    plt.savefig(sid.dirname + "/" + name)
    plt.close()
    
def draw_labels(sid: SimInputData, graph: Graph, edges: Edges, \
    name: str, data: str) -> None:
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

    name : str
        name of the saved file with the plot

    data : str
        parameter taken as edge width (diameter or flow)
    """
    # draw first panel for the network
    plt.axis('equal')
    
    plt.figure(figsize=(sid.figsize, sid.figsize))
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
    x_zero, y_zero = [], []
    for node in graph.zero_nodes:
        x_zero.append(pos[node][0])
        y_zero.append(pos[node][1])
    #print (x_zero, y_zero)
    plt.scatter(x_in, y_in, s = 30, facecolors = 'white', edgecolors = 'black')
    plt.scatter(x_out, y_out, s = 30, facecolors = 'black', \
        edgecolors = 'white')
    
    if data == 'd':
        qs1 = (1 - edges.boundary_list) * edges.diams \
            * (edges.diams < edges.diams_initial / 2)
        qs2 = (1 - edges.boundary_list) * edges.diams \
            * (edges.diams >= edges.diams_initial / 2)
        nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'r', \
            width = sid.ddrawconst * np.array(qs1))
        nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'k', \
            width = sid.ddrawconst * np.array(qs2))
    elif data == 'q':
        qs = (1 - edges.boundary_list) * np.abs(edges.flow)
        nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'k', \
            width = sid.qdrawconst * np.array(qs))
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=dict(zip(edges.edge_list, np.arange(0, len(edges.edge_list)))), font_size = 5)
    #plt.scatter(x_zero, y_zero, s = 60, facecolors = 'blue', edgecolors = 'black')
    # save file in the directory
    plt.axis('off')
    plt.savefig(sid.dirname + "/" + name)
    plt.close()

def draw_data(sid: SimInputData, graph: Graph, edges: Edges, \
    data: Data, name: str, plot_data: str) -> None:
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
    plt.figure(figsize=(sid.figsize * 1.25, sid.figsize))
    plt.suptitle(f'G = {sid.G:.2f} Daeff = {sid.Da_eff:.2f} noise = {sid.noise_filename}')
    spec = gridspec.GridSpec(ncols = 2, nrows = 2, width_ratios = [3, 1])
    # draw first panel for the network
    plt.subplot(spec.new_subplotspec((0, 0), rowspan = 2)).set_title(f'time: {sid.old_t:.2f} dissolved: {data.dissolved_v:.2f}')
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
    if plot_data == 'd':
        qs1 = (1 - edges.boundary_list) * edges.diams \
            * (edges.diams < edges.diams_initial / 2)
        qs2 = (1 - edges.boundary_list) * edges.diams \
            * (edges.diams >= edges.diams_initial / 2)
        nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'r', \
            width = sid.ddrawconst * np.array(qs1))
        nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'k', \
            width = sid.ddrawconst * np.array(qs2))
    elif plot_data == 'q':
        qs = (1 - edges.boundary_list) * np.abs(edges.flow)
        nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'k', \
            width = sid.ddrawconst * np.array(qs))
    # draw histograms with data below the network
    plt.subplot(spec[1]).set_title(f'Slice t: {data.slice_times[-1]}')
    pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
    slices = np.linspace(np.min(pos_x), np.max(pos_x), 120)[10:-10]
    edge_number  = np.array(data.slices[0])
    plt.plot(slices, np.array(data.slices[1]) / edge_number, color = 'red')
    plt.plot(slices, np.array(data.slices[-1]) / edge_number)

    plt.ylim(0, 1)
    plt.xlabel('x')
    plt.ylabel('channeling [%]')
    plt.subplot(spec[3]).set_title('Participation ratio')
    plt.plot(data.dissolved_v_list/data.vol_init, data.participation_ratio)
    plt.ylim(0, 1)
    plt.xlim(0, sid.dissolved_v_max/data.vol_init)
    plt.xlabel('dissolved volume')
    # ax_p = plt.subplot(spec[3])
    # ax_p.set_title('Participation ratio')
    # ax_p.set_ylim(0, 1)
    # ax_p.set_xlim(0, sid.tmax)
    # ax_p.set_xlabel('time')
    # ax_p.set_ylabel('participation ratio')
    # ax_p2 = ax_p.twinx()
    # ax_p2.plot(data.t, data.participation_ratio_nom, label = "pi", color='green', linestyle='dashed')
    # ax_p2.plot(data.t, data.participation_ratio_denom, label = "pi'", color='red', linestyle='dashed')
    # ax_p.plot(data.t, data.participation_ratio)
    # save file in the directory
    plt.savefig(sid.dirname + "/" + name)
    plt.close()
    
    
def draw_focusing(sid: SimInputData, graph: Graph, inc: Incidence, edges: Edges, \
    data: Data, name: str, plot_data: str) -> None:
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
    plt.figure(figsize=(sid.figsize, sid.figsize * 1.5))
    plt.suptitle(f'G = {sid.G:.2f} Daeff = {sid.Da_eff:.2f} noise = {sid.noise_filename}')
    spec = gridspec.GridSpec(ncols = 1, nrows = 2, height_ratios = [2, 1])
    # draw first panel for the network
    #plt.subplot(spec.new_subplotspec((0, 0), rowspan = 2)).set_title(f'time: {sid.old_t:.2f} dissolved: {data.dissolved_v / data.vol_init:.2f}')
    ax1 = plt.subplot(spec[0])
    ax1.set_title(f'time: {sid.old_t:.2f} dissolved: {data.dissolved_v:.2f}')
    plt.axis('equal')
    plt.xlim(0, sid.n)
    plt.ylim(-1.1, sid.n + 1.1)
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
    plt.scatter(x_in, y_in, s = 1000 / sid.n, facecolors = 'white', edgecolors = 'black')
    plt.scatter(x_out, y_out, s = 1000 / sid.n, facecolors = 'black', \
        edgecolors = 'black')
    merged_number = np.asarray(inc.plot.sum(axis = 0)).flatten() * 2 / 3
    if np.sum(merged_number == 0) > 0:
        raise ValueError("Merge number zero")
    if np.sum((inc.plot @ edges.flow) * (edges.flow != 0) != edges.flow) > 0:
        raise ValueError("Merge number zero")
    merged_number = merged_number * (merged_number > 0) + 1 * (merged_number == 0)
    diams = inc.plot @ edges.diams / merged_number
    flow = inc.plot @ np.abs(edges.flow) / merged_number
    in_flow = np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2
    transverse_flow = np.max(in_flow[edges.edge_list], axis = 1) * edges.transversed
    flow2 = inc.plot @ np.abs(transverse_flow) / merged_number
    flow = np.max([flow, flow2], axis = 0)
    #print(np.max(merged_number))
    if plot_data == 'd':
        qs1 = (1 - edges.boundary_list) * diams \
            * (diams < edges.diams_initial / 2)
        qs2 = (1 - edges.boundary_list) * diams \
            * (diams >= edges.diams_initial / 2)
        nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'r', \
            width = sid.ddrawconst * np.array(qs1))
        nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'k', \
            width = sid.ddrawconst * np.array(qs2))
    elif plot_data == 'q':
        qs = (1 - edges.boundary_list) * np.abs(flow)
        nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'k', \
            width = sid.ddrawconst * np.array(qs))
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels=dict(zip(edges.edge_list, np.arange(0, len(edges.edge_list)))), font_size = 5)
    # draw histograms with data below the network
    plt.subplot(spec[1], sharex = ax1)
    pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
    slices = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
    edge_number  = np.array(data.slices[0])
    plt.plot(slices, np.array((edge_number - 2 * np.array(data.slices[1])) / edge_number), color = 'red')
    plt.plot(slices, np.array((edge_number - 2 * np.array(data.slices[-1])) / edge_number))
    plt.ylim(0, 1)
    plt.xlabel('x')
    plt.ylabel('flow focusing index')
    plt.subplots_adjust(wspace=0, hspace=0)
    # save file in the directory
    plt.savefig(sid.dirname + "/" + name)
    plt.close()


def draw_focusing2(sid: SimInputData, graph: Graph, inc: Incidence, edges: Edges, \
    data: Data, name: str, plot_data: str) -> None:
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
    plt.figure(figsize=(sid.figsize, sid.figsize * 1.5))
    plt.suptitle(f'G = {sid.G:.2f} Daeff = {sid.Da_eff:.2f} noise = {sid.noise_filename}')
    spec = gridspec.GridSpec(ncols = 1, nrows = 2, height_ratios = [2, 1])
    # draw first panel for the network
    #plt.subplot(spec.new_subplotspec((0, 0), rowspan = 2)).set_title(f'time: {sid.old_t:.2f} dissolved: {data.dissolved_v / data.vol_init:.2f}')
    plt.subplot(spec[0]).set_title(f'time: {sid.old_t:.2f} dissolved: {data.dissolved_v:.2f}')
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
    merged_number = np.asarray(inc.plot.sum(axis = 0)).flatten() * 2 / 3
    if np.sum(merged_number == 0) > 0:
        raise ValueError("Merge number zero")
    if np.sum((inc.plot @ edges.flow) * (edges.flow != 0) != edges.flow) > 0:
        raise ValueError("Merge number zero")
    merged_number = merged_number * (merged_number > 0) + 1 * (merged_number == 0)
    diams = inc.plot @ edges.diams / merged_number
    flow = inc.plot @ np.abs(edges.flow) / merged_number
    #print(np.max(merged_number))
    if plot_data == 'd':
        qs1 = (1 - edges.boundary_list) * diams \
            * (diams < edges.diams_initial / 2)
        qs2 = (1 - edges.boundary_list) * diams \
            * (diams >= edges.diams_initial / 2)
        nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'r', \
            width = sid.ddrawconst * np.array(qs1))
        nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'k', \
            width = sid.ddrawconst * np.array(qs2))
    elif plot_data == 'q':
        qs = (1 - edges.boundary_list) * np.abs(edges.flow)
        nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'k', \
            width = sid.ddrawconst * np.array(qs))
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels=dict(zip(edges.edge_list, np.arange(0, len(edges.edge_list)))), font_size = 5)
    # draw histograms with data below the network
    plt.subplot(spec[1]).set_title(f'Slice t: {data.slice_times[-1]}')
    pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
    slices = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
    edge_number  = np.array(data.slices[0])
    plt.plot(slices, np.array(data.slices[1]) / edge_number, color = 'red')
    plt.plot(slices, np.array(data.slices[-1]) / edge_number)
    plt.ylim(0, 1)
    plt.xlabel('x')
    plt.ylabel('flow focusing [%]')
    # save file in the directory
    plt.savefig(sid.dirname + "/" + name)
    plt.close()