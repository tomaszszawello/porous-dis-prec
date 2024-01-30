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

import scipy.sparse as spr
# inc.inlet = spr.diags(edges.inlet) @ inc.incidence
# inc.middle = spr.diags(1 - graph.in_vec - graph.out_vec) @ ((inc.incidence.T @ inc.incidence) != 0)
# inc.boundary = spr.diags(graph.in_vec + graph.out_vec)

# #print (np.abs(inc.inlet) @ graph.in_vec)
# in_edges = 1 * (np.abs(inc.inlet) @ graph.in_vec == 2)
# out_edges = 1 * (np.abs(spr.diags(edges.outlet) @ inc.incidence) @ graph.out_vec == 2)
# print (np.sum(in_edges), np.sum(out_edges))
# inc.merge = spr.diags(1 - in_edges - out_edges) @ inc.merge @ spr.diags(1 - in_edges - out_edges)

iters, tmax, i, t, breakthrough = initialize_iterators(sid)

# main loop
# runs until we reach iteration limit or time limit or network is dissolved
#while t < tmax and i < iters and not breakthrough:
while t < tmax and i < iters:
    print(f'Iter {i + 1}/{iters} Time {t:.2f}/{tmax:.2f}')
    # initialize vectors
    pressure_b = Pr.create_vector(sid, graph)
    cb_b = Di.create_vector(sid, graph)
    # find pressure and update flow in edges
    print ('Solving pressure')
    pressure = Pr.solve_flow(sid, inc, graph, edges, pressure_b)
    #data.check_data(edges)
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
        graph.update_network(edges)
        Dr.draw(sid, graph, edges, \
            f'network_{sid.old_t:.2f}.png', "q")
        # Dr.draw(sid, graph, edges, \
        #     f'network_d_{sid.old_t:.2f}.png', "d")
        # Dr.draw_nodes(sid, graph, edges, \
        #     f'nodes_{sid.old_t:.2f}.png', "d")
        data.check_data(edges)
        data.check_init_slice_channelization(graph, inc, edges)
        data.check_slice_channelization(graph, inc, edges, t)
    else:
        if sid.old_t // sid.track_every != (sid.old_t + sid.dt) \
            // sid.track_every:
            data.check_data(edges)
            data.check_slice_channelization(graph, inc, edges, sid.old_t)
        if sid.old_t // sid.plot_every != (sid.old_t + sid.dt) \
            // sid.plot_every:
            graph.update_network(edges)
            # Dr.uniform_hist(sid, graph, edges, cb, cc, \
            #     f'hist_{sid.old_t:.2f}.png', "d")
            Dr.draw(sid, graph, edges, \
                f'network_{sid.old_t:.2f}.png', "d")   
            data.check_data(edges)
            
    # if sid.old_iters % sid.plot_every == 0:
    #     graph.update_network(edges)
    #     Dr.uniform_hist(sid, graph, edges, cb, cc, \
    #         f'hist_{sid.old_t:.2f}.png', "q")   

    # for node in graph.zero_nodes:
    #     if len(inc.middle[node].nonzero()[1]):
    #         print (inc.middle[node].nonzero()[1])
    #     if len(inc.middle[:, node].nonzero()[0]):
    #         print (inc.middle[node].nonzero()[0])
    #     if len(inc.incidence[:,node].nonzero()[0]):
    #         print (inc.incidence[:,node].nonzero()[0])
    
    # Q_in = np.sum(edges.inlet * np.abs(edges.flow))
    # Q_out = np.sum(edges.outlet * np.abs(edges.flow))
    # if np.abs(Q_in - Q_out) > 0.001:
    #     Dr.uniform_hist(sid, graph, edges, cb, cc, \
    #         f'hist_{sid.old_t:.2f}.png', "q")
    #     Dr.draw_labels(sid, graph, edges, \
    #          f'network_{sid.old_t:.2f}.png', "q")
    #     # Dr.draw_nodes(sid, graph, edges, cb, \
    #     #     f'nodes_{sid.old_t:.2f}.png', "d")
    #     #np.savetxt('outlet.txt', edges.outlet)
    #     #np.savetxt('q.txt', edges.flow)
    #     np.savetxt('inc.txt', inc.incidence.toarray())
    #     np.savetxt('inc_dif.txt', (inc.incidence - inc_prev).toarray())
    #     np.savetxt('merge.txt', (inc.merge - merge_prev).toarray())
    #     print('Q_in =', Q_in, 'Q_out =', Q_out)
    #     raise ValueError('Flow not matching!')
    
    if np.max(cb) > 1.1:
        print (np.max(cb))
        nmax = np.where(cb == np.max(cb))[0][0]
        print (nmax)
        for index in inc.incidence[:,nmax].nonzero()[0]:
            nodes = inc.incidence[index].nonzero()[1]
            print (index, nodes)
            print(edges.flow[index])
            for node in nodes:
                print(cb[node])

        # Dr.uniform_hist(sid, graph, edges, cb, cc, \
        #     f'hist_{sid.old_t:.2f}.png', "d")
        # Dr.draw_labels(sid, graph, edges, \
        #     f'network_{sid.old_t:.2f}.png', "d")
        # Dr.draw_nodes(sid, graph, edges, cb, \
        #     f'nodes_{sid.old_t:.2f}.png', "d")
        #print(inc.incidence.T[)[0][0]])
        cb_inc = np.abs(inc.incidence.T @ (spr.diags(edges.flow) \
            @ inc.incidence > 0))
        print (np.sum(cb_inc > 1) + np.sum(cb_inc < -1))
        np.savetxt("cb_inc.txt", cb_inc.toarray())
        # find vector with non-diagonal coefficients
        qc = edges.flow * np.exp(-np.abs(sid.Da / (1 + sid.G * edges.diams) \
            * edges.diams * edges.lens / edges.flow))
        qc = np.array(np.ma.fix_invalid(qc, fill_value = 0))
        qc_matrix = np.abs(inc.incidence.T @ spr.diags(qc) @ inc.incidence)
        cb_matrix = cb_inc.multiply(qc_matrix)
        diag = -np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2
        print(diag[nmax])
        diag = np.array(-np.abs(inc.incidence.T @ spr.diags(edges.flow) @ inc.incidence).sum(axis = 1).reshape((1, sid.nsq)) / 2)[0]
        diag = graph.in_vec + 1 * (diag == 0) + diag * (1 + graph.out_vec - graph.in_vec)
        cb_matrix.setdiag(diag)
        print((inc.incidence.T @ spr.diags(edges.flow) @ inc.incidence)[nmax])
        print(np.abs(inc.incidence.T @ spr.diags(edges.flow) @ inc.incidence).sum(axis = 1)[nmax])
        print(diag[nmax])
        print(cb_matrix[nmax])
        raise ValueError("cb...")

    #     # find diagonal coefficients (inlet flow for each node)
    #     diag = -np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2
    #     # set diagonal for input nodes to 1
    #     for node in graph.in_nodes:
    #         diag[node] = 1
    #     # multiply diagonal for output nodes (they have no outlet, so inlet flow
    #     # is equal to whole flow); also fix for nodes which are connected only to
    #     # other out_nodes - without it we get a singular matrix (whole row of
    #     # zeros)
    #     for node in graph.out_nodes:
    #         if diag[node] != 0:
    #             diag[node] *= 2
    #         else:
    #             diag[node] = 1
    #     # fix for nodes with no connections
    #     for i, node in enumerate(diag):
    #         if node == 0:
    #             diag[i] = 1
    #     # replace diagonal
    #     cb_matrix.setdiag(diag)
    #     np.savetxt('cb_m.txt', cb_matrix.toarray()[nmax])
    #     np.savetxt('cb.txt', cb)
    #     np.savetxt('incmerge.txt', ((inc.incidence.T @ inc.incidence) != 0).sum(axis = 1))

    #     break
    # grow/shrink diameters and update them in edges, update volumes with
    # dissolved/precipitated values, check if network dissolved, find new
    # timestep
    print ('Growing')
    breakthrough, dt_next = Gr.update_diameters(sid, inc, edges, cb, cc)
    #inc_prev = inc.incidence.copy()
    #merge_prev = inc.merge.copy()
    
    # merge edges
    Me.new_solve_merging(sid, inc, graph, edges)
    
    # if i > 200:
    #     import numpy as np
    #     np.savetxt('incmerge.txt', (inc.incidence != 0).sum(axis = 1))
    #     np.savetxt('merged.txt', edges.merged)
    #     #np.savetxt('inlet.txt', inc.inlet.toarray())
    #     np.savetxt('middle.txt', (inc.middle != 0).sum(axis = 1))
    #     break
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
    # Dr.draw(sid, graph, edges, \
    #     f'network_{sid.old_iters:.2f}.png', "q")
    # Sv.save('/save.dill', sid, graph, inc, edges)
    data.save_data()
    data.plot_data()
    data.plot_slice_channelization_v2(sid, graph)

    # Dr.uniform_hist(sid, graph, edges, cb, cc, \
    #     f'hist_{sid.old_t:.2f}.png', "d")
    # Dr.draw(sid, graph, edges, \
    #     f'network_{sid.old_t:.2f}.png', "d")   
    # Dr.draw_nodes(sid, graph, edges, cb, \
    #     f'nodes_{sid.old_t:.2f}.png', "d")