""" Initial parameters of the simulation.

This module contains all parameters set before the simulation. Class
SimInputData is used in nearly all functions. Most of the parameters (apart
from VARIOUS section) are set by the user before starting the simulation.
Most notable parameters are: n - network size, iters/tmax - simulation length,
Da_eff, G, K, Gamma - dissolution/precipitation parameters, include_cc - turn
on precipitation, load - build a new network or load a previous one.

TO DO: fix own geometry
"""

import numpy as np


class SimInputData:
    ''' Configuration class for the whole simulation.
    '''
    # GENERAL
    n: int = 100
    "network size"
    iters: int = 100000
    "maximum number of iterations"
    tmax: float = 5000.
    "maximum time"
    plot_every: int = 100
    "frequency of plotting the results"
    track_every: int = 100000

    # DISSOLUTION & PRECIPITATION
    Da_eff: float = 1
    "effective Damkohler number"
    G: float = 1
    "diffusion to reaction ratio"
    Da: float = Da_eff * (1 + G)
    "Damkohler number"
    K: float = 0.5
    "precipitation to dissolution reaction rate"
    Gamma: float = 2.
    "precipitation to dissolution acid capacity number"
    merge_length: float = 100.
    "diameter scale to length scale ratio for merging"

    # INCLUDE
    include_adt: bool = True
    "include adaptive timestep"
    include_cc: bool = False
    "include precipitation"

    # INITIAL CONDITIONS
    qin: float = 1.
    "characteristic flow for inlet edge"
    cb_in: float = 1.
    "inlet B concentration"
    cc_in: float = 0.
    "inlet C concentration"

    # TIME
    dt: float = 0.01
    "initial timestep (if no adaptive timestep, timestep for whole simulation)"
    growth_rate: float = 0.02
    ("maximum percentage growth of an edges (used for finding adaptive \
     timestep)")
    dt_max: float = 5.
    "maximum timestep (for adaptive)"

    # DIAMETERS
    d0: float = 1.
    "initial dimensionless mean diameter"
    sigma_d0: float = 0.5
    "initial diameter standard deviation"
    dmin: float = 0.01
    "minimum diameter"
    dmax: float = 1000.
    "maximum diameter"
    d_break: float = 4.
    "minimal diameter of outlet edge for network to be dissolved"

    # DRAWING
    figsize: float = 10.
    "figure size"
    qdrawconst: float = 30.
    "constant for improving flow drawing"
    ddrawconst: float = 1 / merge_length
    "constant for improving diameter drawing"

    # INITIALIZATION
    load: int = 2
    ("type of loading: 0 - build new network based on config and start new \
     simulation, 1 - load previous network from load_name and continue \
     simulation, 2 - load template network from load_name and start new \
     simulation")
    load_name: str = 'rect100/G1.00Daeff1.00/39'
    "name of loaded network"

    # GEOMETRY
    geo: str = "rect" # WARNING - own is deprecated
    ("type of geometry: 'rect' - rectangular, 'own' - custom inlet and outlet \
     nodes, set in in/out_nodes_own")
    periodic: str = 'top'
    ("periodic boundary condition: 'none' - no PBC, 'top' - up and down, \
     'side' - left and right, 'all' - PBC everywhere")
    in_nodes_own: np.ndarray = np.array([[20, 50]]) / 100 * n
    "custom outlet for 'own' geometry"
    out_nodes_own: np.ndarray = np.array([[80, 50], [70, 25], [70, 75]]) \
        / 100 * n
    "custom outlet for 'own' geometry"

    # VARIOUS
    ne: int = 0
    "number of edges (updated later)"
    ntr: int = 0
    "number of triangles (updated later)"
    nsq: int = n ** 2
    "number of nodes"
    old_iters: int = 0
    "total iterations of simulation"
    old_t: float = 0.
    "total time of simulation"
    Q_in = 1.
    "total inlet flow (updated later)"
    dirname: str = geo + str(n) + '/' + f'G{G:.2f}Daeff{Da_eff:.2f}'
    "directory of simulation"
    