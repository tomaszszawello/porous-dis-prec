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

def create_dth_matrix():
    """ Create matrix with threshold merging diameter for each edge neighbours.
    
    Build by triangles? (dth[edge1, edge2] = dth[edge2, edge1] = edge3_l / 2, )
    ne x ne
    """
    print ('create merging matrix')

def solve_merging(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges):
    """ Find edges which should be merged.
    
    ((dth_matrix > 0) @ spr.diags(edges.diams) +
    ((dth_matrix > 0).T @ spr.diags(edges.diams).T) / 2
    """
    merge_edges = ((inc.merge @ spr.diags(1 - edges.merged) > 0) \
        @ spr.diags(edges.diams) + ((inc.merge.T @ spr.diags(1 - edges.merged) \
            > 0) @ spr.diags(edges.diams)).T) / 2 > inc.merge * sid.merge_length
    # if sid.old_t == 0:
    #     np.savetxt('merge.txt', merge_edges.toarray())
    #     np.savetxt('lens.txt', edges.lens)
    #     np.savetxt('diams.txt', edges.diams)
    #np.savetxt('merge.txt', merge_edges.toarray())
    # take coordinates of nonzero matrix elements - these are indices of edges
    # that we want to merge
    merged = []
    merge_index = 0
    for edge_pair in list(zip(merge_edges.nonzero()[0], \
        merge_edges.nonzero()[1])):
        # we don't want to merge same edges twice
        if edge_pair[0] >= edge_pair[1]:
            #print ("SKIP")
            continue
        if edges.inlet[edge_pair[0]] or edges.inlet[edge_pair[1]] or edges.outlet[edge_pair[0]] or edges.outlet[edge_pair[1]]:
            print ("SKIP2")
            continue
        # if edges.diams[edge_pair[1]] == 0 or edges.diams[edge_pair[0]] == 0:
        #     continue
        
        # choose which edge will remain
        # if inc.incidence[edge_pair[0]].sum() \
        #     < inc.incidence[edge_pair[1]].sum():
        if edges.diams[edge_pair[1]] > edges.diams[edge_pair[0]]:
            merge_i = edge_pair[1]
            zero_i = edge_pair[0]
        else:
            merge_i = edge_pair[0]
            zero_i = edge_pair[1]
        # we need the third edge taking part in merging
        transverse_i_list = []
        for i in inc.merge[edge_pair[0]].nonzero()[1]:
            if inc.merge[edge_pair[1], i] != 0:
                if i != edge_pair[0] and i != edge_pair[1]:
                    transverse_i_list.append(i)
        for transverse_i in transverse_i_list:
            n1, n2 = inc.incidence[transverse_i].nonzero()[1]
            for index in inc.incidence[:, n1].nonzero()[0]:
                if inc.incidence[index, n2] != 0:
                    if index not in transverse_i_list:
                        transverse_i_list.append(index)
        if merge_i in merged or zero_i in merged:
            continue
        flag = 0
        for transverse_i in transverse_i_list:
            if transverse_i in merged:
                flag = 1
            if edges.inlet[transverse_i] or edges.outlet[transverse_i]:
                flag = 1
        if flag:
            continue
        print (f"Merging {merge_i} {zero_i}")
        merged.append(merge_i)
        merged.append(zero_i)
        for transverse_i in transverse_i_list:
            merged.append(transverse_i)
        # find node that should be merged - what if n3 == n4?
        n1, n2 = inc.incidence[merge_i].nonzero()[1]
        n3, n4 = inc.incidence[zero_i].nonzero()[1]
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
            np.savetxt('incmerge.txt', (inc.merge != 0).sum(axis = 1))
            np.savetxt('merged.txt', edges.merged)
            #np.savetxt('inlet.txt', inc.inlet.toarray())
            np.savetxt('middle.txt', (inc.middle != 0).sum(axis = 1))
            np.savetxt('zero.txt', graph.zero_nodes)
            print (n1, n2, n3, n4, merge_i, zero_i, transverse_i_list)
            print (edges.edge_list[merge_i], edges.edge_list[zero_i], edges.edge_list[transverse_i_list])
            print (graph.nodes[n1]['pos'], graph.nodes[n2]['pos'], graph.nodes[n3]['pos'], graph.nodes[n4]['pos'])
            print (merge_node, zero_node)
            raise ValueError("Wrong merging!")

        graph.zero_nodes[zero_node] = 1

        # zero diameters of merged edges
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
        # TO DO: does it matter how we set lenghts of merged edges?
        edges.lens[merge_i] /= edges.diams[merge_i]
        edges.lens[zero_i] = 1
        # if one of the edges was inlet or outlet, set that to the new edge
        # TO DO: avoid situation when inlet is outlet
        edges.inlet[merge_i] = np.max((edges.inlet[merge_i], \
            edges.inlet[zero_i]))
        edges.outlet[merge_i] = np.max((edges.outlet[merge_i], \
            edges.outlet[zero_i]))
        edges.inlet[zero_i] = 0
        edges.outlet[zero_i] = 0
        for transverse_i in transverse_i_list:
            if edges.inlet[transverse_i]:
                edges.inlet[transverse_i] = 0
                edges.inlet[merge_i] = 1
            if edges.outlet[transverse_i]:
                edges.outlet[transverse_i] = 0
                edges.outlet[merge_i] = 1
        # node merging
        # print (inc.incidence[merge_i].nonzero()[1], inc.incidence[zero_i].nonzero()[1])
        # for transverse_i in transverse_i_list:
        #     print(inc.incidence[transverse_i].nonzero()[1])
        # print (inc.incidence[2253].nonzero()[1])
        if merge_node != zero_node:
            # check if one of inlet/outlet nodes was merged
            if merge_node in graph.in_nodes and zero_node in graph.in_nodes:
                graph.in_nodes = np.delete(graph.in_nodes, \
                    np.where(graph.in_nodes == zero_node)[0])
            elif merge_node in graph.out_nodes and zero_node in graph.out_nodes:
                graph.out_nodes = np.delete(graph.out_nodes, \
                    np.where(graph.out_nodes == zero_node)[0])
            elif zero_node in graph.in_nodes:
                graph.in_nodes[np.where(graph.in_nodes == zero_node)] = merge_node
            elif zero_node in graph.out_nodes:
                graph.out_nodes[np.where(graph.out_nodes == zero_node)] = merge_node
            # replace connections to the zeroed node with connections to merged node
            for index in inc.incidence[:, zero_node].nonzero()[0]:
                if inc.incidence[index, merge_node] == 0:
                    inc.incidence[index, merge_node] \
                        = inc.incidence[index, zero_node]
                inc.incidence[index, zero_node] = 0
            if merge_node in graph.in_nodes:
                for index in inc.middle[merge_node].nonzero()[1]:
                    inc.middle[merge_node, index] = 0
                for index in inc.middle[zero_node].nonzero()[1]:
                    inc.middle[zero_node, index] = 0
                inc.boundary[merge_node, merge_node] = 1
                for index in inc.inlet[:, zero_node].nonzero()[0]:
                    if inc.inlet[index, merge_node] == 0:
                        inc.inlet[index, merge_node] = inc.inlet[index, zero_node]
                    inc.inlet[index, zero_node] = 0
            else:
                for index in inc.middle[zero_node].nonzero()[1]:
                    if inc.middle[merge_node, index] == 0:
                        inc.middle[merge_node, index] = inc.middle[zero_node, index]
                    inc.middle[zero_node, index] = 0
            inc.boundary[zero_node, zero_node] = 0
            for index in inc.middle[:, zero_node].nonzero()[0]:
                if inc.middle[index, merge_node] == 0:
                    inc.middle[index, merge_node] = inc.middle[index, zero_node]
                inc.middle[index, zero_node] = 0

        # edge merging


        # zero rows in incidence matrix for merged edges
        for index in inc.incidence[zero_i].nonzero()[1]:
            inc.incidence[zero_i, index] = 0
        for transverse_i in transverse_i_list:
            for index in inc.incidence[transverse_i].nonzero()[1]:
                inc.incidence[transverse_i, index] = 0
        if 1.0 in (inc.incidence != 0).sum(axis = 1):
            print (n1, n2, n3, n4, merge_i, zero_i, transverse_i_list)
            print (inc.incidence[merge_i].nonzero()[1], inc.incidence[zero_i].nonzero()[1], inc.incidence[transverse_i_list].nonzero()[1])
            print (graph.nodes[n1]['pos'], graph.nodes[n2]['pos'], graph.nodes[n3]['pos'], graph.nodes[n4]['pos'])
            print (merge_node, zero_node)
            np.savetxt('zero.txt', graph.zero_nodes)
            np.savetxt('merged.txt', edges.merged)
            np.savetxt('incmerge.txt', (inc.incidence != 0).sum(axis = 1))
            print (inc.incidence[2253].nonzero()[1])
            raise ValueError('after')
        # add edges that should be omitted to merged list
        edges.merged[zero_i] = 1
        for transverse_i in transverse_i_list:
            edges.merged[transverse_i] = 1        
        # fix merge matrix
        for index in inc.merge[merge_i, :].nonzero()[1]:
            inc.merge[merge_i, index] += (edges.diams[merge_i] - d1) / 2
        for index in inc.merge[:, merge_i].nonzero()[0]:
            inc.merge[index, merge_i] += (edges.diams[merge_i] - d1) / 2
        for index in inc.merge[zero_i, :].nonzero()[1]:
            inc.merge[merge_i, index] = inc.merge[zero_i, index] + (edges.diams[merge_i] - d2) / 2
        for index in inc.merge[:, zero_i].nonzero()[0]:
            inc.merge[index, merge_i] = inc.merge[index, zero_i] + (edges.diams[merge_i] - d2) / 2
        for index in inc.merge[zero_i, :].nonzero()[1]:
            inc.merge[zero_i, index] = 0
        for index in inc.merge[:, zero_i].nonzero()[0]:
            inc.merge[index, zero_i] = 0
        inc.merge[merge_i, merge_i] = 0
        for transverse_i in transverse_i_list:
            if transverse_i != zero_i:
                for index in inc.merge[transverse_i].nonzero()[1]:
                    inc.merge[transverse_i, index] = 0
                for index in inc.merge[:, transverse_i].nonzero()[0]:
                    inc.merge[index, transverse_i] = 0
                
        if edges.inlet[merge_i]:
            for index in inc.inlet[zero_i].nonzero()[1]:
                inc.inlet[zero_i, index] = 0
            for transverse_i in transverse_i_list:
                for index in inc.inlet[transverse_i].nonzero()[1]:
                    inc.inlet[transverse_i, index] = 0


        merge_index += 1
        if merge_index > 50 and sid.old_t > 10:
            #np.savetxt('merged_in.txt', edges.merged * edges.inlet)
            #np.savetxt('merged_out.txt', edges.merged * edges.outlet)
            np.savetxt('inc.txt', (inc.incidence != 0).sum(axis = 1))
            np.savetxt('merge.txt', (inc.merge != 0).sum(axis = 1))
            np.savetxt('middle.txt', (inc.middle != 0).sum(axis = 1))
            raise ValueError("Too much merging")





def merge():
    """ Restructure all matrices due to merging.
    
    """
    print ('restructure arrays')
