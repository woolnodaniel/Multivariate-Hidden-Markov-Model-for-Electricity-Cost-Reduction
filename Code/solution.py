'''
COMP9418 Assignment 2
This file is the example code to show how the assignment will be tested.

Name: Daniel Woolnough     zID: 5116128
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


from load_parameters import *
from helper import *
from predict import *
from model import *


def get_action(sensor_data):
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


"""
Trivial Solutions: for testing/comparison purposes
"""


def always_off(sensor_data):
    return {'lights' + str(num) : 'off' for num in range(1,36)}



def always_on(sensor_data):
    return {'lights' + str(num) : 'on' for num in range(1,36)}



