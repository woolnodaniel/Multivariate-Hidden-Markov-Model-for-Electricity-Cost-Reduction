'''
To optimize the threshold for how confident we should be in turning a light on.
'''
# Make division default to floating-point, saving confusion
from __future__ import division
from __future__ import print_function

# Allowed libraries
import numpy as np
import pandas as pd
import scipy as sp
import scipy.special
import heapq as pq
import matplotlib as mp
import matplotlib.pyplot as plt
import math
from itertools import product, combinations
from collections import OrderedDict as odict
import collections
from graphviz import Digraph, Graph
from tabulate import tabulate
import copy
import sys
import os
import datetime
import sklearn
import re
import ast
import random
import time
# import the function written by the student
from solution import *
from example_test import *

#reset global belief, for multiple iterations
def reset_belief():
    global Belief
    Belief = Priors.copy()

def action(sensor_data, alpha):
    global Belief, Neighbours, Adj, Sig, EvConds, Robots, outcomeSpace, Rooms, Areas
    # first, call passage of time
    Passage = {}
    for R in Areas:
        Passage[R] = passage_of_time(R, Neighbours, Belief, Adj, outcomeSpace)

    # next, call observation
    obs = convert_observation(sensor_data)
    for R in Areas:
        Belief[R] = observation(R, obs, Sig, Passage, EvConds, Robots, outcomeSpace, verbose=False)

    #create dictionary of actions
    action = {'lights' + room[1:]:None for room in Rooms}
    price = obs['electricity_price']
    proba = alpha
    for R in Rooms:
        if Belief[R]['table'][(1,)] > proba:
            action['lights' + R[1:]] = 'on'
        else:
            action['lights' + R[1:]] = 'off'
    return action

def run(num_sims=1):
    seeds = [np.random.randint(1,100_000) for _ in range(num_sims)]
    costs = np.zeros((16,num_sims))
    print('Running...')

    for k, alpha in enumerate(np.linspace(0.2,0.35,num=16)):

        print(f'Alpha = {alpha}')

        for j in range(num_sims):

            print(f' -- Simulation {j}:')

            #get_action
            global Belief
            reset_belief()

            np.random.seed(seeds[j])
            simulator = SmartBuildingSimulatorExample()

            start_time = time.time()
            sensor_data = simulator.timestep()
            for i in range(len(simulator.data)-1):
                actions_dict = action(sensor_data, alpha)
                sensor_data = simulator.timestep(actions_dict)
            #costs.append(simulator.cost)
            costs[k,j] = simulator.cost

    mean_costs = np.mean(costs, axis=1)
    print(mean_costs)
    plt.plot(np.linspace(0,1,num=16), mean_costs, '.-b')
    plt.show()

run(50)