#!/usr/bin/env python3
""" Start simulation based on parameters from config.

This module performs the whole simulation. It should be started after all
parameters in config file are set (most importantly n - network size,
iters/tmax - simulation length, Da_eff, G, K, Gamma - dissolution/precipitation
parameters, include_cc - turn on precipitation, load - build a new network
or load a previous one). After starting, directory consisting of
geometry + network size / G + Damkohler number / simulation index
will be created. Plots of the network and other data will be saved there.
"""

import dissolution as Di
import draw_net as Dr
import growth as Gr
import merging as Me
import precipitation as Pi
import pressure as Pr
import save as Sv

from build import build
from utils import initialize_iterators, update_iterators

import numpy as np

# initialize main classes
sid, inc, graph, edges, data = build()

iters, tmax, i, t, breakthrough = initialize_iterators(sid)
iterator_dissolved = 0

# initial merging
if sid.include_merging:
    for initial_i in range(sid.initial_merging):
        Me.new_solve_merging(sid, inc, graph, edges)

# main loop
# runs until we reach iteration limit or time limit or network is dissolved
#while t < tmax and i < iters and not breakthrough:
while t < tmax and i < iters and data.dissolved_v < sid.dissolved_v_max:
    print(f'Iter {i + 1}/{iters} Time {t:.2f}/{tmax:.2f} Dissolved {data.dissolved_v:.2f}/{sid.dissolved_v_max:.2f}')
    # initialize vectors
    pressure_b = Pr.create_vector(sid, graph)
    cb_b = Di.create_vector(sid, graph)
    # find pressure and update flow in edges
    print ('Solving pressure')
    pressure = Pr.solve_flow(sid, inc, graph, edges, pressure_b)
    # find B concentration
    print ('Solving concentration')
    cb = Di.solve_dissolution(sid, inc, graph, edges, cb_b)
    # find C concentration
    cc = Pi.solve_precipitation(sid, inc, graph, edges, cb)
    # update diameters and flows in graph, print physical parameters, save plot
    # if i % sid.plot_every == 0:  
    #     graph.update_network(edges)
    #     # Dr.uniform_hist(sid, graph, edges, cb, cc, \
    #     #     f'network_{sid.old_iters:.2f}.png', "d")
    #     Dr.draw(sid, graph, edges, \
    #         f'network_{sid.old_iters:.2f}.png', "q")
    #     data.check_data(edges)
    
    if t == 0:
        # graph.update_network(inc, edges)
        # Dr.draw(sid, graph, edges, \
        #     f'network_d_{sid.old_t:.2f}.png', "d")
        # Dr.draw_nodes(sid, graph, edges, \
        #     f'nodes_{sid.old_t:.2f}.png', "d")
        data.check_data(edges)
        data.check_init_slice_channelization(graph, inc, edges)
        data.check_slice_channelization(graph, inc, edges, t)
        Dr.draw_focusing(sid, graph, inc, edges, data, \
            f'network_d_{data.dissolved_v:.2f}.png', "d")
        Dr.draw_focusing(sid, graph, inc, edges, data, \
            f'network_q_{data.dissolved_v:.2f}.png', "q")
    else:
        if data.dissolved_v // sid.track_every > iterator_dissolved:
            print('Drawing')
            iterator_dissolved += 1
            data.check_data(edges)
            data.check_slice_channelization(graph, inc, edges, sid.old_t)
            # graph.update_network(inc, edges)
            # Dr.uniform_hist(sid, graph, edges, cb, cc, \
            #     f'hist_{sid.old_t:.2f}.png', "d")
            Dr.draw_focusing(sid, graph, inc, edges, data, \
                f'network_d_{data.dissolved_v:.2f}.png', "d")
            Dr.draw_focusing(sid, graph, inc, edges, data, \
                f'network_q_{data.dissolved_v:.2f}.png', "q")
            # Dr.draw_focusing2(sid, graph, inc, edges, data, \
            #     f'network_q2_{sid.old_t:.2f}.png', "q")
            # merged_number = np.asarray(inc.plot.sum(axis = 0)).flatten()
            # np.savetxt(sid.dirname + f'/q_{sid.old_t:.2f}.txt', edges.flow)
            # np.savetxt(sid.dirname + f'/d_{sid.old_t:.2f}.txt', edges.diams)
            # np.savetxt(sid.dirname + f'/merged_number_{sid.old_t:.2f}.txt', merged_number)
            # np.savetxt(sid.dirname + f'/merged_q_{sid.old_t:.2f}.txt', inc.plot @ edges.flow / merged_number)
            # np.savetxt(sid.dirname + f'/merged_d_{sid.old_t:.2f}.txt', inc.plot @ edges.diams / merged_number)
            data.check_data(edges)
            
    # if sid.old_iters % sid.plot_every == 0:
    #     graph.update_network(edges)
    #     Dr.uniform_hist(sid, graph, edges, cb, cc, \
    #         f'hist_{sid.old_t:.2f}.png', "q")   
    if np.max(cb) > 1.1:
        raise ValueError("cb...")

    # grow/shrink diameters and update them in edges, update volumes with
    # dissolved/precipitated values, check if network dissolved, find new
    # timestep
    print ('Growing')
    breakthrough, dt_next = Gr.update_diameters(sid, inc, edges, cb, cc)
    
    # merge edges
    if sid.include_merging:
        Me.new_solve_merging(sid, inc, graph, edges)
    
    # update physical parameters in data
    data.collect_data(sid, inc, edges, pressure, cb, cc)
    i, t = update_iterators(sid, i, t, dt_next)
    data.dissolved_v = (np.sum(edges.diams ** 2 * edges.lens) - data.vol_init) / data.vol_init
    
    


# save data from the last iteration of simulation, save the whole simulation
# to be able to continue it later
if i != 1 and sid.load != 1:
    data.check_data(edges)
    #graph.update_network(inc, edges)
    # Dr.uniform_hist(sid, graph, edges, cb, cc, \
    #     f'network_{sid.old_iters:.2f}.png', "d")
    # Dr.draw(sid, graph, edges, \
    #     f'network_{sid.old_iters:.2f}.png', "q")
    Sv.save('/save.dill', sid, graph, inc, edges)
    data.save_data()
    data.plot_data()
    data.plot_participation(sid)
    data.plot_slice_channelization_v2(sid, graph)
    

    # Dr.uniform_hist(sid, graph, edges, cb, cc, \
    #     f'hist_{sid.old_t:.2f}.png', "d")
    # Dr.draw(sid, graph, edges, \
    #     f'network_{sid.old_t:.2f}.png', "d")   
    # Dr.draw_nodes(sid, graph, edges, cb, \
    #     f'nodes_{sid.old_t:.2f}.png', "d")