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
import precipitation as Pi
import pressure as Pr
import save as Sv

from build import build
from utils import initialize_iterators, update_iterators


# initialize main classes
sid, inc, graph, edges, data = build()

# initialize constant vectors
pressure_b = Pr.create_vector(sid, graph)
cb_b = Di.create_vector(sid, graph)

iters, tmax, i, t, breakthrough = initialize_iterators(sid)

# main loop
# runs until we reach iteration limit or time limit or network is dissolved
#while t < tmax and i < iters and not breakthrough:
while t < tmax and i < iters:  
    print(f'Iter {i + 1}/{iters} Time {t:.2f}/{tmax:.2f}')
    # find pressure and update flow in edges
    pressure = Pr.solve_flow(sid, inc, graph, edges, pressure_b)
    # find B concentration
    cb = Di.solve_dissolution(sid, inc, graph, edges, cb_b)
    # find C concentration
    cc = Pi.solve_precipitation(sid, inc, graph, edges, cb)
    # update diameters and flows in graph, print physical parameters, save plot
    if i % sid.plot_every == 0:
        data.check_data(edges)
        graph.update_network(edges)
        # Dr.uniform_hist(sid, graph, edges, cb, cc, \
        #     f'network_{sid.old_iters:.2f}.png', "d")
        # Dr.draw(sid, graph, edges, \
        #     f'network_{sid.old_iters:.2f}.png', "d")
    if t == 0:
        data.check_init_slice_channelization(graph, inc, edges)
        data.check_slice_channelization(graph, inc, edges, t)
    elif sid.old_t // sid.track_every != (sid.old_t + sid.dt) \
        // sid.track_every:
        data.check_slice_channelization(graph, inc, edges, sid.old_t)
    # grow/shrink diameters and update them in edges, update volumes with
    # dissolved/precipitated values, check if network dissolved, find new
    # timestep
    breakthrough, dt_next = Gr.update_diameters(sid, inc, edges, cb, cc)
    # update physical parameters in data
    data.collect_data(sid, inc, edges, pressure, cb, cc)
    i, t = update_iterators(sid, i, t, dt_next)

# save data from the last iteration of simulation, save the whole simulation
# to be able to continue it later
if i != 1:
    data.check_data(edges)
    graph.update_network(edges)
    # Dr.uniform_hist(sid, graph, edges, cb, cc, \
    #     f'network_{sid.old_iters:.2f}.png', "d")
    Dr.draw(sid, graph, edges, \
    f'network_{sid.old_iters:.2f}.png', "d")
    Sv.save('/save.dill', sid, graph)
    data.save_data()
    data.plot_data()
    data.plot_slice_channelization(graph)