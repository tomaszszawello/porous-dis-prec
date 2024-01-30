""" Merge pores in the network.

This module merges pores in the network when the sum of their diameters exceed
a threshold fraction of the distance between them. Nodes from which the edges
originate are merged into one with new position being a diameter-weighted
average of the positions of original nodes. However lenghts of other edges
connected to that new effective node remain unchanged, despite the new
position. After merging entries in the incidence matrices are changed...

Conserve pore space? What with the 3rd edge (connecting merged nodes)?

"""
import numpy as np
import scipy.sparse as spr

from config import SimInputData
from network import Edges, Graph
from incidence import Incidence

import draw_net as Dr



def new_solve_merging(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges):
    """ Find edges which should be merged.

    """
    merge_edges = ((inc.merge @ spr.diags(1 - edges.merged) > 0) \
        @ spr.diags(edges.diams) + ((inc.merge.T @ spr.diags(1 - edges.merged) \
            > 0) @ spr.diags(edges.diams)).T) / 2 > inc.merge
    merged, zeroed, transversed = [], [], []
    merged_diams, zeroed_diams = [], []
    merged_nodes = []
    # take coordinates of nonzero matrix elements - these are indices of edges
    # that we want to merge
    for edge_pair in list(zip(merge_edges.nonzero()[0], \
        merge_edges.nonzero()[1])):
        # choose which edge will remain - the one with larger diameter
        if (edges.inlet[edge_pair[1]] and edges.inlet[edge_pair[0]]) or (edges.outlet[edge_pair[1]] and edges.outlet[edge_pair[0]]):
            if edges.diams[edge_pair[1]] > edges.diams[edge_pair[0]]:
                merge_i = edge_pair[1]
                zero_i = edge_pair[0]
            else:
                merge_i = edge_pair[0]
                zero_i = edge_pair[1]
        elif edges.inlet[edge_pair[1]] or edges.outlet[edge_pair[1]]:
            merge_i = edge_pair[1]
            zero_i = edge_pair[0]
        elif edges.inlet[edge_pair[0]] or edges.outlet[edge_pair[0]]:
            merge_i = edge_pair[0]
            zero_i = edge_pair[1]
        elif edges.diams[edge_pair[1]] > edges.diams[edge_pair[0]]:
            merge_i = edge_pair[1]
            zero_i = edge_pair[0]
        else:
            merge_i = edge_pair[0]
            zero_i = edge_pair[1]
        if merge_i in merged + zeroed + transversed or zero_i in merged + zeroed + transversed:
            continue
        n1, n2 = inc.incidence[merge_i].nonzero()[1]
        n3, n4 = inc.incidence[zero_i].nonzero()[1]
        # try:
        #     n1, n2 = inc.incidence[merge_i].nonzero()[1]
        # except ValueError:
        #     print(merge_i, zero_i)
        #     print(merged)
        #     print(zeroed)
        #     print(inc.incidence[merge_i].nonzero())
        #     print(inc.incidence[zero_i].nonzero())
        #     print(inc.merge[merge_i].nonzero())
        #     print(inc.merge[zero_i].nonzero())
        #     raise ValueError('n1, n2...')
        # try:
        #     n3, n4 = inc.incidence[zero_i].nonzero()[1]
        # except ValueError:
        #     print(merge_i, zero_i)
        #     print(merged)
        #     print(zeroed)
        #     print(inc.incidence[merge_i].nonzero())
        #     print(inc.incidence[zero_i].nonzero())
        #     print(inc.merge[merge_i].nonzero())
        #     print(inc.merge[zero_i].nonzero())
        #     raise ValueError('n3, n4...')
        if (n1 in graph.in_nodes and n2 in graph.in_nodes) or (n1 in graph.out_nodes and n2 in graph.out_nodes):
            continue
        if (n3 in graph.in_nodes and n4 in graph.in_nodes) or (n3 in graph.out_nodes and n4 in graph.out_nodes):
            continue
        if n1 == n3:
            merge_node = n2
            zero_node = n4
        elif n1 == n4:
            merge_node = n2
            zero_node = n3
        elif n2 == n3:
            merge_node = n1
            zero_node = n4
        elif n2 == n4:
            merge_node = n1
            zero_node = n3
        else:
            raise ValueError("Wrong merging!")
        merge_nodes_flat = np.reshape(merged_nodes, 2 * len(merged_nodes))
        if merge_node in merge_nodes_flat or zero_node in merge_nodes_flat:
            continue
        
        # we need the third edge taking part in merging
        transverse_i_list = []
        if merge_node != zero_node:
            for i in inc.incidence.T[merge_node].nonzero()[1]:
                if inc.incidence[i, zero_node] != 0:
                    transverse_i_list.append(i)
        flag = 0
        for transverse_i in transverse_i_list:
            if transverse_i in merged + zeroed + transversed:
                flag = 1
        if flag:
            continue
        # if len(transverse_i_list) == 0:
        #     continue
        print (f"Merging {merge_i} {zero_i} {transverse_i_list} Nodes {merge_node} {zero_node}")
        merged.append(merge_i)
        zeroed.append(zero_i)
        for transverse_i in transverse_i_list:
            transversed.append(transverse_i)
        # TO DO: consider how merged diameter should be calculated
        d1, d2, l1, l2 = edges.diams[merge_i], edges.diams[zero_i], \
            edges.lens[merge_i], edges.lens[zero_i]
        edges.diams[merge_i] = d1 + d2
        edges.lens[merge_i] = d1 * l1 + d2 * l2
        edges.diams[zero_i] = 0
        for transverse_i in transverse_i_list:
            di, li = edges.diams[transverse_i], edges.lens[transverse_i]
            edges.diams[merge_i] += di
            edges.diams[transverse_i] = 0
            edges.lens[merge_i] += di * li
            edges.lens[transverse_i] = 1
            if edges.inlet[transverse_i]:
                edges.inlet[merge_i] = 1
            if edges.outlet[transverse_i]:
                edges.outlet[merge_i] = 1
            edges.inlet[transverse_i] = 0
            edges.outlet[transverse_i] = 0
            if edges.boundary_list[transverse_i]:
                edges.boundary_list[merge_i] = 1
        merged_diams.append((edges.diams[merge_i] - d1) / 2)
        zeroed_diams.append((edges.diams[merge_i] - d2) / 2)
        # TO DO: does it matter how we set lenghts of merged edges?
        edges.lens[merge_i] /= edges.diams[merge_i]
        edges.lens[zero_i] = 1
        edges.inlet[zero_i] = 0
        edges.outlet[zero_i] = 0
        if edges.boundary_list[zero_i]:
            edges.boundary_list[merge_i] = 1
        if merge_node != zero_node:
            merged_nodes.append((zero_node, merge_node))
            if edges.boundary_list[merge_i] == 0:
                graph.nodes[zero_node]['pos'] = graph.nodes[merge_node]['pos']
            graph.zero_nodes.append(zero_node)
            # check if one of inlet/outlet nodes was merged
            if merge_node in graph.in_nodes and zero_node in graph.in_nodes:
                graph.in_nodes = np.delete(graph.in_nodes, \
                    np.where(graph.in_nodes == zero_node)[0])
                graph.in_vec[zero_node] = 0
            elif merge_node in graph.out_nodes and zero_node in graph.out_nodes:
                graph.out_nodes = np.delete(graph.out_nodes, \
                    np.where(graph.out_nodes == zero_node)[0])
                graph.out_vec[zero_node] = 0
            else:
                if zero_node in graph.out_nodes:
                    print (zero_node)
                    
            # elif zero_node in graph.in_nodes:
            #     graph.in_nodes[np.where(graph.in_nodes == zero_node)] = merge_node
            # elif zero_node in graph.out_nodes:
            #     graph.out_nodes[np.where(graph.out_nodes == zero_node)] = merge_node
        # add edges that should be omitted to merged list
        if edges.merged[zero_i]:
            raise ValueError('Merging zeroed edge')
        for transverse_i in transverse_i_list:
            if edges.merged[transverse_i]:
                raise ValueError('Merging zeroed edge (transverse)')
        edges.merged[zero_i] = 1
        for transverse_i in transverse_i_list:
            edges.merged[transverse_i] = 1
    if len(merged) > 0:
        # fix merge matrix
        merge_fix_edges = np.ones(sid.ne)
        merge_fix_diams = np.zeros(sid.ne)
        for i, edge in enumerate(merged):
             merge_fix_edges[edge] = 0
             merge_fix_diams[edge] = merged_diams[i]
        for i, edge in enumerate(zeroed):
            merge_fix_edges[edge] = 0
            merge_fix_diams[edge] = zeroed_diams[i]
        for edge in transversed:
            merge_fix_edges[edge] = 0
        merge_fix = spr.diags(merge_fix_edges) @ (inc.merge > 0) @ spr.diags(merge_fix_diams)
        inc.merge += merge_fix + merge_fix.T
        diag_nodes = spr.csr_matrix(spr.diags(np.ones(sid.nsq)))
        for n1, n2 in merged_nodes:
            diag_nodes[n1, n1] = 0
            diag_nodes[n1, n2] = 1
        merged_edges = np.ones(sid.ne)
        for edge in zeroed + transversed:
            merged_edges[edge] = 0
        diag_edges = spr.diags(merged_edges)
        #diag_edges = spr.diags(1 - edges.merged)
        incidence3 = inc.incidence.copy()
        inc.incidence = diag_edges @ (inc.incidence @ diag_nodes)
        incidence2 = inc.incidence.copy()
        sign_fix2 = 2 * inc.incidence @ graph.in_vec + edges.inlet + 2 * inc.incidence @ graph.out_vec + edges.outlet
        sign_fix = 1 * (sign_fix2 == -1) - 1 * (sign_fix2 == 3) + 1 * (sign_fix2 == 0)
        diag_edges *= spr.diags(1 * (sign_fix != 0))
        edges.inlet *= (sign_fix != 0)
        edges.outlet *= (sign_fix != 0)
        inc.incidence = spr.diags(sign_fix) @ inc.incidence
        # if np.sum(np.abs(inc.incidence @ graph.in_vec) - edges.inlet):
        #     print('inlet: ', (np.abs(inc.incidence @ graph.in_vec) - edges.inlet).nonzero())
        #     raise ValueError("inlet problem")
        # if np.sum(np.abs(inc.incidence @ graph.out_vec) - edges.outlet):
        #     print('outlet: ', (np.abs(inc.incidence @ graph.out_vec) - edges.outlet).nonzero())
        #     raise ValueError("outlet problem")     
        # if np.sum(np.abs(inc.incidence) - np.abs(incidence2)):
        #     print('inlet: ', (np.abs(inc.incidence @ graph.in_vec) - edges.inlet).nonzero())
        #     print('outlet: ', (np.abs(inc.incidence @ graph.out_vec) - edges.outlet).nonzero())
        #     np.savetxt('out_vec.txt', inc.incidence @ graph.out_vec)
        #     np.savetxt('out_nodes.txt', graph.out_nodes)
        #     np.savetxt('outlet.txt', edges.outlet)
        #     np.savetxt('sign_fix.txt', sign_fix)
        #     np.savetxt('sign_fix2.txt', sign_fix2)
        #     print(incidence3[4581].nonzero(), inc.incidence[1357].nonzero(), inc.incidence[1358].nonzero(), inc.incidence[1354].nonzero())
        #     print(graph.out_nodes)
        #     print((inc.incidence[4581] - incidence2[4581]).nonzero())
        #     raise ValueError("incidence problem")
        inc.inlet = spr.diags(edges.inlet) @ inc.incidence
        inc.middle = spr.diags(1 - graph.in_vec - graph.out_vec) @ ((inc.incidence.T @ inc.incidence) != 0)
        inc.boundary = spr.diags(graph.in_vec + graph.out_vec)
        inc.merge = diag_edges @ inc.merge @ diag_edges


# def solve_merging(sid: SimInputData, inc: Incidence, graph: Graph, \
#     edges: Edges):
#     """ Find edges which should be merged.

#     """
#     merge_edges = ((inc.merge @ spr.diags(1 - edges.merged) > 0) \
#         @ spr.diags(edges.diams) + ((inc.merge.T @ spr.diags(1 - edges.merged) \
#             > 0) @ spr.diags(edges.diams)).T) / 2 > inc.merge * sid.merge_length
#     merged = []
#     # take coordinates of nonzero matrix elements - these are indices of edges
#     # that we want to merge
#     # np.savetxt(f'incmerge{sid.old_iters}.txt', inc.merge.toarray())
#     # np.savetxt('lens.txt', edges.lens)
#     for edge_pair in list(zip(merge_edges.nonzero()[0], \
#         merge_edges.nonzero()[1])):
#         # we don't want to merge same edges twice
#         #if edge_pair[0] >= edge_pair[1]:
#             #print ("SKIP")
#             #continue
#         # if edges.inlet[edge_pair[0]] == 1 or edges.inlet[edge_pair[1]] == 1:
#         #     print ("SKIP2")
#         #     continue
#         # if edges.diams[edge_pair[1]] == 0 or edges.diams[edge_pair[0]] == 0:
#         #     continue
        
#         # choose which edge will remain - the one with larger diameter
#         if edges.inlet[edge_pair[0]]:
#             merge_i = edge_pair[0]
#             zero_i = edge_pair[1]
#         elif edges.inlet[edge_pair[1]]:
#             merge_i = edge_pair[1]
#             zero_i = edge_pair[0]
#         elif edges.diams[edge_pair[1]] > edges.diams[edge_pair[0]]:
#             merge_i = edge_pair[1]
#             zero_i = edge_pair[0]
#         else:
#             merge_i = edge_pair[0]
#             zero_i = edge_pair[1]
#         # we need the third edge taking part in merging
#         transverse_i_list = []
#         for i in inc.merge[edge_pair[0]].nonzero()[1]:
#             if inc.merge[edge_pair[1], i] != 0:
#                 if i != edge_pair[0] and i != edge_pair[1]:
#                     transverse_i_list.append(i)
#         for transverse_i in transverse_i_list:
#             n1, n2 = inc.incidence[transverse_i].nonzero()[1]
#             for index in inc.incidence[:, n1].nonzero()[0]:
#                 if inc.incidence[index, n2] != 0:
#                     if index not in transverse_i_list:
#                         transverse_i_list.append(index)
#         if merge_i in merged or zero_i in merged:
#             continue
#         flag = 0
#         for transverse_i in transverse_i_list:
#             if transverse_i in merged:
#                 flag = 1
#         if flag:
#             continue
#         print (f"Merging {merge_i} {zero_i}")
#         print (transverse_i_list)
#         merged.append(merge_i)
#         merged.append(zero_i)
#         for transverse_i in transverse_i_list:
#             merged.append(transverse_i)
#         # find node that should be merged - what if n3 == n4?
#         n1, n2 = inc.incidence[merge_i].nonzero()[1]
#         n3, n4 = inc.incidence[zero_i].nonzero()[1]
#         if n1 == n3:
#             merge_node = n2
#             zero_node = n4
#         elif n1 == n4:
#             merge_node = n2
#             zero_node = n3
#         elif n2 == n3:
#             merge_node = n1
#             zero_node = n4
#         elif n2 == n4:
#             merge_node = n1
#             zero_node = n3
#         else:
#             raise ValueError("Wrong merging!")
#         # if n1 in graph.zero_nodes or n2 in graph.zero_nodes or n3 in graph.zero_nodes or n4 in graph.zero_nodes:
#         #     print ('in zero nodes')
#         #     continue
#         #merge_node, zero_node = zero_node, merge_node
#         # if merge_node == zero_node:
#         #     continue
#         print (edges.edge_list[merge_i], edges.edge_list[zero_i])
#         print (merge_node, zero_node, n1, n2, n3, n4)


#         # zero diameters of merged edges
#         # TO DO: consider how merged diameter should be calculated
#         d1, d2, l1, l2 = edges.diams[merge_i], edges.diams[zero_i], \
#             edges.lens[merge_i], edges.lens[zero_i]
#         edges.diams[merge_i] = d1 + d2
#         edges.lens[merge_i] = d1 * l1 + d2 * l2
#         edges.diams[zero_i] = 0
#         for transverse_i in transverse_i_list:
#             di, li = edges.diams[transverse_i], edges.lens[transverse_i]
#             edges.diams[merge_i] += di
#             edges.diams[transverse_i] = 0
#             edges.lens[merge_i] += di * li
#             edges.lens[transverse_i] = 1
#         # TO DO: does it matter how we set lenghts of merged edges?
#         edges.lens[merge_i] /= edges.diams[merge_i]
#         edges.lens[zero_i] = 1
#         # if one of the edges was inlet or outlet, set that to the new edge
#         # TO DO: avoid situation when inlet is outlet
#         edges.inlet[merge_i] = np.max((edges.inlet[merge_i], \
#             edges.inlet[zero_i]))
#         edges.outlet[merge_i] = np.max((edges.outlet[merge_i], \
#             edges.outlet[zero_i]))
#         edges.inlet[zero_i] = 0
#         edges.outlet[zero_i] = 0
#         for transverse_i in transverse_i_list:
#             if edges.inlet[transverse_i]:
#                 edges.inlet[transverse_i] = 0
#                 edges.inlet[merge_i] = 1
#             if edges.outlet[transverse_i]:
#                 edges.outlet[transverse_i] = 0
#                 edges.outlet[merge_i] = 1
#         # node merging
#         # print (inc.incidence[merge_i].nonzero()[1], inc.incidence[zero_i].nonzero()[1])
#         # for transverse_i in transverse_i_list:
#         #     print(inc.incidence[transverse_i].nonzero()[1])
#         # print (inc.incidence[2253].nonzero()[1])
#         if merge_node != zero_node:
#             graph.zero_nodes.append(zero_node)
#             # check if one of inlet/outlet nodes was merged
#             if merge_node in graph.in_nodes and zero_node in graph.in_nodes:
#                 graph.in_nodes = np.delete(graph.in_nodes, \
#                     np.where(graph.in_nodes == zero_node)[0])
#             elif merge_node in graph.out_nodes and zero_node in graph.out_nodes:
#                 graph.out_nodes = np.delete(graph.out_nodes, \
#                     np.where(graph.out_nodes == zero_node)[0])
#             elif zero_node in graph.in_nodes:
#                 graph.in_nodes[np.where(graph.in_nodes == zero_node)] = merge_node
#             elif zero_node in graph.out_nodes:
#                 graph.out_nodes[np.where(graph.out_nodes == zero_node)] = merge_node
#             # replace connections to the zeroed node with connections to merged node
#             for index in inc.incidence[:, zero_node].nonzero()[0]:
#                 if inc.incidence[index, merge_node] == 0:
#                     inc.incidence[index, merge_node] \
#                         = inc.incidence[index, zero_node]
#                 inc.incidence[index, zero_node] = 0
#             if merge_node in graph.in_nodes:
#                 for index in inc.middle[merge_node].nonzero()[1]:
#                     inc.middle[merge_node, index] = 0
#                 for index in inc.middle[zero_node].nonzero()[1]:
#                     inc.middle[zero_node, index] = 0
#                 inc.boundary[merge_node, merge_node] = 1
#                 for index in inc.inlet[:, zero_node].nonzero()[0]:
#                     if inc.inlet[index, merge_node] == 0:
#                         inc.inlet[index, merge_node] = inc.inlet[index, zero_node]
#                     inc.inlet[index, zero_node] = 0
#             elif merge_node in graph.out_nodes:
#                 for index in inc.middle[merge_node].nonzero()[1]:
#                     inc.middle[merge_node, index] = 0
#                 for index in inc.middle[zero_node].nonzero()[1]:
#                     inc.middle[zero_node, index] = 0
#                 inc.boundary[merge_node, merge_node] = 1
#             else:
#                 for index in inc.middle[zero_node].nonzero()[1]:
#                     if inc.middle[merge_node, index] == 0:
#                         inc.middle[merge_node, index] = inc.middle[zero_node, index]
#                     inc.middle[zero_node, index] = 0
#             inc.boundary[zero_node, zero_node] = 0
#             for index in inc.middle[:, zero_node].nonzero()[0]:
#                 if inc.middle[index, merge_node] == 0:
#                     inc.middle[index, merge_node] = inc.middle[index, zero_node]
#                 inc.middle[index, zero_node] = 0

#         else:
#             print ('zero = merge')

#         # edge merging

#         # zero rows in incidence matrix for merged edges
#         for index in inc.incidence[zero_i].nonzero()[1]:
#             inc.incidence[zero_i, index] = 0
#         for transverse_i in transverse_i_list:
#             for index in inc.incidence[transverse_i].nonzero()[1]:
#                 inc.incidence[transverse_i, index] = 0

#         # add edges that should be omitted to merged list
#         edges.merged[zero_i] = 1
#         for transverse_i in transverse_i_list:
#             edges.merged[transverse_i] = 1        
#         # fix merge matrix
#         for index in inc.merge[merge_i, :].nonzero()[1]:
#             inc.merge[merge_i, index] += (edges.diams[merge_i] - d1) / 2
#         for index in inc.merge[:, merge_i].nonzero()[0]:
#             inc.merge[index, merge_i] += (edges.diams[merge_i] - d1) / 2
#         # shouldn't contain if != 0?
#         for index in inc.merge[zero_i, :].nonzero()[1]:
#             inc.merge[merge_i, index] = inc.merge[zero_i, index] + (edges.diams[merge_i] - d2) / 2
#         for index in inc.merge[:, zero_i].nonzero()[0]:
#             inc.merge[index, merge_i] = inc.merge[index, zero_i] + (edges.diams[merge_i] - d2) / 2
#         for index in inc.merge[zero_i, :].nonzero()[1]:
#             inc.merge[zero_i, index] = 0
#         for index in inc.merge[:, zero_i].nonzero()[0]:
#             inc.merge[index, zero_i] = 0
#         inc.merge[merge_i, merge_i] = 0
#         for transverse_i in transverse_i_list:
#             if transverse_i != zero_i:
#                 for index in inc.merge[transverse_i].nonzero()[1]:
#                     inc.merge[transverse_i, index] = 0
#                 for index in inc.merge[:, transverse_i].nonzero()[0]:
#                     inc.merge[index, transverse_i] = 0

#         if edges.inlet[merge_i]:
#             for index in inc.inlet[zero_i].nonzero()[1]:
#                 inc.inlet[zero_i, index] = 0
#             for transverse_i in transverse_i_list:
#                 for index in inc.inlet[transverse_i].nonzero()[1]:
#                     inc.inlet[transverse_i, index] = 0

#         # for i, (n1, n2) in enumerate(edges.edge_list):
#         #     if n1 == zero_node:
#         #         edges.edge_list[i] = (merge_node, n2)
#         #     elif n2 == zero_node:
#         #         edges.edge_list[i] = (n1, merge_node)

        
# def new_solve_merging(sid: SimInputData, inc: Incidence, graph: Graph, \
#     edges: Edges):
#     """ Find edges which should be merged.

#     """
#     merge_edges = ((inc.merge @ spr.diags(1 - edges.merged) > 0) \
#         @ spr.diags(edges.diams) + ((inc.merge.T @ spr.diags(1 - edges.merged) \
#             > 0) @ spr.diags(edges.diams)).T) / 2 > inc.merge
#     merged = []
#     zeroed = []
#     trasversed = []
#     merged_nodes = []
#     # take coordinates of nonzero matrix elements - these are indices of edges
#     # that we want to merge
#     for edge_pair in list(zip(merge_edges.nonzero()[0], \
#         merge_edges.nonzero()[1])):
#         # choose which edge will remain - the one with larger diameter
#         if (edges.inlet[edge_pair[1]] and edges.inlet[edge_pair[0]]) or (edges.outlet[edge_pair[1]] and edges.outlet[edge_pair[0]]):
#             if edges.diams[edge_pair[1]] > edges.diams[edge_pair[0]]:
#                 merge_i = edge_pair[1]
#                 zero_i = edge_pair[0]
#             else:
#                 merge_i = edge_pair[0]
#                 zero_i = edge_pair[1]
#         elif edges.inlet[edge_pair[1]] or edges.outlet[edge_pair[1]]:
#             merge_i = edge_pair[1]
#             zero_i = edge_pair[0]
#         elif edges.inlet[edge_pair[0]] or edges.outlet[edge_pair[0]]:
#             merge_i = edge_pair[0]
#             zero_i = edge_pair[1]
#         elif edges.diams[edge_pair[1]] > edges.diams[edge_pair[0]]:
#             merge_i = edge_pair[1]
#             zero_i = edge_pair[0]
#         else:
#             merge_i = edge_pair[0]
#             zero_i = edge_pair[1]
#         if merge_i in merged + zeroed or zero_i in merged + zeroed:
#             continue

#         # # we need the third edge taking part in merging
#         # transverse_i_list = []
#         # for i in inc.merge[edge_pair[0]].nonzero()[1]:
#         #     if inc.merge[edge_pair[1], i] != 0:
#         #         if i != edge_pair[0] and i != edge_pair[1]:
#         #             transverse_i_list.append(i)
#         # for transverse_i in transverse_i_list.copy():
#         #     n1, n2 = inc.incidence[transverse_i].nonzero()[1]
#         #     for index in inc.incidence[:, n1].nonzero()[0]:
#         #         if inc.incidence[index, n2] != 0:
#         #             if index not in transverse_i_list:
#         #                 transverse_i_list.append(index)

#         # flag = 0
#         # for transverse_i in transverse_i_list:
#         #     if transverse_i in merged:
#         #         flag = 1
#         # if flag:
#         #     continue

#         #print (transverse_i_list)
#         # find node that should be merged - what if n3 == n4?

#         # not enough values to unpack?!
#         try:
#             n1, n2 = inc.incidence[merge_i].nonzero()[1]
#         except ValueError:
#             print(merge_i, zero_i)
#             print(merged)
#             print(zeroed)
#             print(inc.incidence[merge_i].nonzero())
#             print(inc.incidence[zero_i].nonzero())
#             print(inc.merge[merge_i].nonzero())
#             print(inc.merge[zero_i].nonzero())
#             #np.savetxt('inc.txt', inc.incidence.toarray())
#             #np.savetxt('merge.txt', inc.merge.toarray())
#             raise ValueError('...')
#         n3, n4 = inc.incidence[zero_i].nonzero()[1]
#         if (n1 in graph.in_nodes and n2 in graph.in_nodes) or (n1 in graph.out_nodes and n2 in graph.out_nodes):
#             continue
#         if (n3 in graph.in_nodes and n4 in graph.in_nodes) or (n3 in graph.out_nodes and n4 in graph.out_nodes):
#             continue
#         if n1 == n3:
#             merge_node = n2
#             zero_node = n4
#         elif n1 == n4:
#             merge_node = n2
#             zero_node = n3
#         elif n2 == n3:
#             merge_node = n1
#             zero_node = n4
#         elif n2 == n4:
#             merge_node = n1
#             zero_node = n3
#         else:
#             raise ValueError("Wrong merging!")
#         merge_nodes_flat = np.reshape(merged_nodes, 2 * len(merged_nodes))
#         if merge_node in merge_nodes_flat or zero_node in merge_nodes_flat:
#             continue
        
#         # we need the third edge taking part in merging
#         transverse_i_list = []
#         if merge_node != zero_node:
#             for i in inc.incidence.T[merge_node].nonzero()[1]:
#                 if inc.incidence[i, zero_node] != 0:
#                     transverse_i_list.append(i)
                    
#         # for i in inc.merge[edge_pair[0]].nonzero()[1]:
#         #     if inc.merge[edge_pair[1], i] != 0:
#         #         if i != edge_pair[0] and i != edge_pair[1]:
#         #             transverse_i_list.append(i)
#         # for transverse_i in transverse_i_list.copy():
#         #     n1, n2 = inc.incidence[transverse_i].nonzero()[1]
#         #     for index in inc.incidence[:, n1].nonzero()[0]:
#         #         if inc.incidence[index, n2] != 0:
#         #             if index not in transverse_i_list:
#         #                 transverse_i_list.append(index)

#         flag = 0
#         for transverse_i in transverse_i_list:
#             if transverse_i in merged + zeroed:
#                 flag = 1
#         if flag:
#             continue
#         # if len(transverse_i_list) == 0:
#         #     continue
#         print (f"Merging {merge_i} {zero_i} {transverse_i_list}")
#         merged_nodes.append((zero_node, merge_node))
#         merged.append(merge_i)
#         zeroed.append(zero_i)
#         for transverse_i in transverse_i_list:
#             zeroed.append(transverse_i)
#         # TO DO: consider how merged diameter should be calculated
#         d1, d2, l1, l2 = edges.diams[merge_i], edges.diams[zero_i], \
#             edges.lens[merge_i], edges.lens[zero_i]
#         edges.diams[merge_i] = d1 + d2
#         edges.lens[merge_i] = d1 * l1 + d2 * l2
#         edges.diams[zero_i] = 0
#         for transverse_i in transverse_i_list:
#             di, li = edges.diams[transverse_i], edges.lens[transverse_i]
#             edges.diams[merge_i] += di
#             edges.diams[transverse_i] = 0
#             edges.lens[merge_i] += di * li
#             edges.lens[transverse_i] = 1
#             if edges.inlet[transverse_i]:
#                 edges.inlet[merge_i] = 1
#             if edges.outlet[transverse_i]:
#                 edges.outlet[merge_i] = 1
#             edges.inlet[transverse_i] = 0
#             edges.outlet[transverse_i] = 0
#             if edges.boundary_list[transverse_i]:
#                 edges.boundary_list[merge_i] = 1
#         # TO DO: does it matter how we set lenghts of merged edges?
#         edges.lens[merge_i] /= edges.diams[merge_i]
#         edges.lens[zero_i] = 1
#         edges.inlet[zero_i] = 0
#         edges.outlet[zero_i] = 0
#         if edges.boundary_list[zero_i]:
#             edges.boundary_list[merge_i] = 1
#         if merge_node != zero_node:
#             if edges.boundary_list[merge_i] == 0:
#                 graph.nodes[zero_node]['pos'] = graph.nodes[merge_node]['pos']
#             graph.zero_nodes.append(zero_node)
#             # check if one of inlet/outlet nodes was merged
#             if merge_node in graph.in_nodes and zero_node in graph.in_nodes:
#                 graph.in_nodes = np.delete(graph.in_nodes, \
#                     np.where(graph.in_nodes == zero_node)[0])
#                 graph.in_vec[zero_node] = 0
#             elif merge_node in graph.out_nodes and zero_node in graph.out_nodes:
#                 graph.out_nodes = np.delete(graph.out_nodes, \
#                     np.where(graph.out_nodes == zero_node)[0])
#                 graph.out_vec[zero_node] = 0
#             # elif zero_node in graph.in_nodes:
#             #     graph.in_nodes[np.where(graph.in_nodes == zero_node)] = merge_node
#             # elif zero_node in graph.out_nodes:
#             #     graph.out_nodes[np.where(graph.out_nodes == zero_node)] = merge_node
#         # add edges that should be omitted to merged list
#         if edges.merged[zero_i]:
#             raise ValueError('Merging zeroed edge')
#         for transverse_i in transverse_i_list:
#             if edges.merged[transverse_i]:
#                 raise ValueError('Merging zeroed edge (transverse)')
#         edges.merged[zero_i] = 1
#         for transverse_i in transverse_i_list:
#             edges.merged[transverse_i] = 1
#         # fix merge matrix
#         for index in inc.merge[merge_i].nonzero()[1]:
#             inc.merge[merge_i, index] += (edges.diams[merge_i] - d1) / 2
#         for index in inc.merge[:, merge_i].nonzero()[0]:
#             inc.merge[index, merge_i] += (edges.diams[merge_i] - d1) / 2
#         # shouldn't contain if != 0?
#         for index in inc.merge[zero_i].nonzero()[1]:
#             inc.merge[merge_i, index] = inc.merge[zero_i, index] + (edges.diams[merge_i] - d2) / 2
#         for index in inc.merge[:, zero_i].nonzero()[0]:
#             inc.merge[index, merge_i] = inc.merge[index, zero_i] + (edges.diams[merge_i] - d2) / 2
#         for index in inc.merge[zero_i].nonzero()[1]:
#             inc.merge[zero_i, index] = 0
#         for index in inc.merge[:, zero_i].nonzero()[0]:
#             inc.merge[index, zero_i] = 0
#         inc.merge[merge_i, merge_i] = 0
#         for transverse_i in transverse_i_list:
#             if transverse_i != zero_i:
#                 for index in inc.merge[transverse_i].nonzero()[1]:
#                     inc.merge[transverse_i, index] = 0
#                 for index in inc.merge[:, transverse_i].nonzero()[0]:
#                     inc.merge[index, transverse_i] = 0
#     if len(merged) > 0:
#         # fix merge matrix
#         # 
        
#         diag_nodes = spr.csr_matrix(spr.diags(np.ones(sid.nsq)))
#         for n1, n2 in merged_nodes:
#             diag_nodes[n1, n1] = 0
#             diag_nodes[n1, n2] = 1
#         merged_edges = np.ones(sid.ne)
#         for edge in zeroed:
#             merged_edges[edge] = 0
#         diag_edges = spr.diags(merged_edges)
#         #diag_edges = spr.diags(1 - edges.merged)
#         inc.incidence = diag_edges @ (inc.incidence @ diag_nodes)
#         sign_fix = 2 * inc.incidence @ graph.in_vec + edges.inlet + 2 * inc.incidence @ graph.out_vec + edges.outlet
#         inc.incidence = spr.diags(1 * (sign_fix == -1)) @ inc.incidence - spr.diags(1 * (sign_fix == 3)) @ inc.incidence + spr.diags(1 * (sign_fix == 0)) @ inc.incidence
#         inc.inlet = spr.diags(edges.inlet) @ inc.incidence
#         inc.middle = spr.diags(1 - graph.in_vec - graph.out_vec) @ ((inc.incidence.T @ inc.incidence) != 0)
#         inc.boundary = spr.diags(graph.in_vec + graph.out_vec)
#         inc.merge = diag_edges @ inc.merge @ diag_edges
