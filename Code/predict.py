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
from load_parameters import *
from model import *

def passage_of_time(room, Neighbours, Belief, Adj, outcomeSpace, *, verbose=False):
    """
    argument
        room: variable Room, to update our probability of
        Neighbours, Belief, Adj, outcomeSpace: as previously defined.
        verbose: True if wanting to print factors at each step.
    returns
        passage: factor P(room | current_evidence)
    note that current evidence is not passed in, as it does not affect the calculation, since
      this probability is only based on current knowledge.
    """
    f = Neighbours[room]
    if verbose:
        printFactor(f)
        print()
    vars_to_elim = Adj[room]
    for V in vars_to_elim:
        #Join prior belief, then eliminate it
        old_belief = Belief[V].copy()
        old_belief['dom'] = (V + '^t',)
        f = join(old_belief, f, outcomeSpace)
        f = marginalize_room(f, V + '^t', outcomeSpace)
        if verbose:
            printFactor(f)
            print()
    f = normalize(f, outcomeSpace)
    f['dom'] = (room,)
    return f

def observation(room, obs, Sig, Passage, EvConds, Robots, outcomeSpace, *, verbose=False):
    """
    arguments
        room: variable to update belief of
        evid: dictionary mapping sensors to their observation
        Passage: dictionary mapping room variables to their passage probabilities (as above)
        Sig, EvPriors, EvConds, outcomeSpace
    returns
        new factor, defining updated belief of room
    """
    newOutcomeSpace = outcomeSpace.copy()
    for E, ev in obs.items():
        if E == 'time' or E == 'electricity_price':
            continue
        elif ev == None:
            continue
        elif E not in ['robot1', 'robot2']:
            newOutcomeSpace = evidence(E, ev, newOutcomeSpace)
        else:
            newOutcomeSpace = evidence(E, ev[1], newOutcomeSpace)
    adj_sensors = [E for E in obs.keys() if E != 'robot1' and E != 'robot2' and E != 'time' \
                   and E != 'electricity_price' and room in Sig[E]]
    oth_sensors = [E for E in obs.keys() if E != 'robot1' and E != 'robot2' and E != 'time' \
                   and E != 'electricity_price' and room not in Sig[E]]
    new_f = Passage[room].copy()
    if verbose:
        printFactor(new_f)
        print()
    for E in adj_sensors:
        if obs[E] == None:
            continue
        new_f = join(new_f, EvConds[E][room], newOutcomeSpace)
        new_f = marginalize_sensor(new_f, E, newOutcomeSpace)
        new_f = normalize(new_f, newOutcomeSpace)
        if verbose:
            print(E, obs[E])
            print()
            printFactor(EvConds[E][room])
            print()
            printFactor(new_f)
            print()
    #finally, we also update with robot ONLY IF the robot is in this room
    for W in ['robot1', 'robot2']:
        if obs[W] == None:
            continue
        if room == obs[W][0]:
            new_f = join(new_f, Robots[W][room], newOutcomeSpace)
            new_f = marginalize_sensor(new_f, W, newOutcomeSpace)
            new_f = normalize(new_f, newOutcomeSpace)
            if verbose:
                print(W, obs[W][1])
                print()
                printFactor(Robots[W][room])
                print()
                printFactor(new_f)
                print()
    return new_f

