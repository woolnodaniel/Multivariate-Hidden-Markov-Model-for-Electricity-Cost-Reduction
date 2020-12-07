'''
To test solution against the trivial solutions of always on/off.
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

#simplified model - only updates belief for rooms.
def get_action_prev(sensor_data):
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
    proba = price / (price + 4)
    for R in Rooms:
        if Belief[R]['table'][(1,)] > proba:
            action['lights' + R[1:]] = 'on'
        else:
            action['lights' + R[1:]] = 'off'
    return action

# more complex model: supposes sensors can see beyond their current room
def get_action_sensitive(sensor_data):
    global Belief, Neighbours, Adj, Sensitive_Sig, EvConds, Robots, outcomeSpace, Rooms, Areas
    # first, call passage of time
    Passage = {}
    for R in Areas:
        Passage[R] = passage_of_time(R, Neighbours, Belief, Adj, outcomeSpace)

    # next, call observation
    obs = convert_observation(sensor_data)
    for R in Areas:
        Belief[R] = observation(R, obs, Sensitive_Sig, Passage, EvConds, Robots, outcomeSpace, verbose=False)

    #create dictionary of actions
    action = {'lights' + room[1:]:None for room in Rooms}
    x = obs['electricity_price']
    if x > 1.2:
        thresh = x / (x + 4) * 1.5 - (1.5 * 1.2)/(1.2 + 4) + 1.2/(1.2 + 4)
    else:
        thresh = x / (x + 4)
    for R in Rooms:
        if Belief[R]['table'][(1,)] > thresh:
            action['lights' + R[1:]] = 'on'
        else:
            action['lights' + R[1:]] = 'off'
    return action

#reset global belief, for multiple iterations
def reset_belief():
    global Belief
    Belief = Priors.copy()

def run(num_sims=1):

    bad1 = 0
    bad2 = 0

    sens_costs = []
    curr_costs = []
    prev_costs = []
    on_costs = []
    print('Running...')

    for j in range(num_sims):

        print(f'Simulation {j}:')

        t = int(time.time())

        #get_action
        global Belief
        reset_belief()

        np.random.seed(t)
        simulator = SmartBuildingSimulatorExample()

        start_time = time.time()
        sensor_data = simulator.timestep()
        for i in range(len(simulator.data)-1):
            actions_dict = get_action(sensor_data)
            sensor_data = simulator.timestep(actions_dict)
        print(f" -- get_action : Total cost for the day: {simulator.cost} cents")
        print(f" -- -- Time taken: {time.time() - start_time}")
        curr_costs.append(simulator.cost)

        #get_action_prev
        np.random.seed(t)
        simulator = SmartBuildingSimulatorExample()

        start_time = time.time()
        sensor_data = simulator.timestep()
        for i in range(len(simulator.data)-1):
            actions_dict = get_action_prev(sensor_data)
            sensor_data = simulator.timestep(actions_dict)
        print(f" -- get_action_prev : Total cost for the day: {simulator.cost} cents")
        print(f" -- -- Time taken: {time.time() - start_time}")
        prev_costs.append(simulator.cost)

        #get_action_sensitive
        global Belief
        reset_belief()

        np.random.seed(t)
        simulator = SmartBuildingSimulatorExample()

        start_time = time.time()
        sensor_data = simulator.timestep()
        for i in range(len(simulator.data)-1):
            actions_dict = get_action_sensitive(sensor_data)
            sensor_data = simulator.timestep(actions_dict)
        print(f" -- get_action_sensitive : Total cost for the day: {simulator.cost} cents")
        print(f" -- -- Time taken: {time.time() - start_time}")
        sens_costs.append(simulator.cost)

        # always on
        np.random.seed(t)
        simulator = SmartBuildingSimulatorExample()

        start_time = time.time()
        sensor_data = simulator.timestep()
        for i in range(len(simulator.data)-1):
            actions_dict = always_on(sensor_data)
            sensor_data = simulator.timestep(actions_dict)
        print(f" -- always_on : Total cost for the day: {simulator.cost} cents")
        print(f" -- -- Time taken: {time.time() - start_time}")
        on_costs.append(simulator.cost)

        if curr_costs[-1] > on_costs[-1]:
            bad1 += 1
        if sens_costs[-1] < curr_costs[-1]:
            bad2 += 1

    print(f'Method performed worse than trivial {bad1} times')
    print(f'Method performed worse than sensitive {bad2} times')
    print(f'Average costs: get_action: {np.mean(curr_costs)}')
    print(f'Average costs: get_action_prev: {np.mean(prev_costs)}')
    print(f'Average costs: get_action_sensitive: {np.mean(sens_costs)}')
    print(f'Average costs: always_on: {np.mean(on_costs)}')

run(100)