# Make division default to floating-point, saving confusion
from __future__ import division
from __future__ import print_function

# Allowed libraries
import numpy as np
import pandas as pd
import scipy as sp
import heapq as pq
import matplotlib as mp
import math
import datetime
from itertools import product, combinations
from collections import OrderedDict as odict
from graphviz import Digraph
from tabulate import tabulate

from helper import *
from model import *

"""
Load csvs into factors. Used for prediction.
Assumes csvs are saved in the same directory as this file.
"""

#for loading more than is used
Sensitive_Sig = {
    'reliable_sensor1': ['r16', 'r17', 'c3b'],
    'reliable_sensor2': ['r5', 'r6', 'r9', 'c3a'],
    'reliable_sensor3': ['r25', 'r26', 'c1'],
    'reliable_sensor4': ['r31', 'r32'],
    'unreliable_sensor1': ['c4', 'o1'],
    'unreliable_sensor2': ['r5', 'r10', 'r15', 'c3a', 'c3b'],
    'unreliable_sensor3': ['r1', 'r2'], #r4???
    'unreliable_sensor4': ['r23', 'r24'],
    'door_sensor1': ['r8', 'r9'],
    'door_sensor2': ['c1', 'c2'],
    'door_sensor3': ['r26', 'r27'],
    'door_sensor4': ['r35', 'c4']
}

# Priors: dictionary of (area) variables to their prior distribution
Priors = {}
for R in Areas:
    Priors[R] = csv_to_factor(R + '_.csv')

# Belief: dictionary of variables to their belief. This is updated in-place and initialised as our Priors.
Belief = Priors.copy()

# Neighbours: conditional probability of a room, given its neighbours (at the previous timestep)
Neighbours = {}
for R in Areas:
    neighs = Adj[R]
    filename = '_'.join([v+'^t' for v in neighs])
    filename += '_' + R + '^t+1_.csv'
    Neighbours[R] = csv_to_factor(filename)

# EvConds: conditional probabilities of evidence, given an adjacent room.
EvConds = {}
for E in Sensors:
    EvConds[E] = {}
    for R in Sensitive_Sig[E]:
        filename = R + '_' + E + '_.csv'
        EvConds[E][R] = csv_to_factor(filename)

# Robots: probability of robot being correct
Robots = {}
for W in Wandering_Robots:
    Robots[W] = {}
    f = csv_to_factor(W + '_.csv')
    for R in Areas:
        new_f = {}
        new_f['dom'] = (R,W)
        new_f['table'] = odict([
            ((0,0), f['table'][(1,)]),
            ((1,0), f['table'][(0,)]),
            ((0,1), f['table'][(0,)]),
            ((1,1), f['table'][(1,)]),
        ])
        Robots[W][R] = new_f