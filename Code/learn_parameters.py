"""
The code in this file was used to generate all of the included .csv files in the submission.
This file is not called or loaded at any point in the prediction, and is only included
in the submission for completeness.
"""


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
from itertools import product, combinations
from collections import OrderedDict as odict
from graphviz import Digraph
from tabulate import tabulate

from helper import *
from model import *
from random import choice

df = pd.read_csv('data.csv')

# convert data to easier to use format (0s and 1s, mostly)
for col in df.columns:
    if col == 'Unnamed: 0':
        continue
    if col[:5] == 'relia' or col[:5] == 'unrel':
        df[col] = df[col].map(lambda x : 1 if x == 'motion' else 0)
    if col[:5] == 'door_':
        df[col] = df[col].map(lambda x : 0 if x == 0 else 1)
    if col[:5] == 'robot':
        df[col] = df[col].map(lambda x : eval(x))
        df[col] = df[col].map(lambda x: (x[0], 1) if x[1] > 0 else (x[0], 0))
    if col[1].isnumeric() or col=='outside':
        df[col] = df[col].map(lambda x : 1 if x > 0 else 0)

#corridor 3 has been split in two: we will turn col c3 into two identical c3a and c3b.
new_columns = {col : col for col in df.columns}
new_columns['c3'] = 'c3a'
df = df.rename(columns=new_columns)
df['c3b'] = df['c3a']
column_order = list(df.columns)
index = column_order.index('c3a')
column_order.remove('c3b')
column_order.insert(index + 1, 'c3b')
df = df[column_order]

#to handle the case of a robot in corridor three, we will choose from both rooms with equal probability
def func(x):
    if x[0] == 'c3':
        return (choice(['c3a', 'c3b']), x[1])
    else:
        return x
df['robot1'] = df['robot1'].map(lambda x : func(x))
df['robot2'] = df['robot2'].map(lambda x : func(x))

# Prior Probabilities for rooms:
A = np.sum(df.iloc[0][Areas] > 0) / len(Areas)
for R in Areas:
    #create a factor table and save to csv
    f = {
        'dom': (R,),
        'table': odict([
            ((1,), A),
            ((0,), 1-A)
        ])
    }
    factor_to_csv(f)

# Transition Probabilities between rooms:
for R in Areas:
    f = {}
    f['dom'] = [v + '^t' for v in Adj[R]]
    f['dom'].append(R + '^t+1')
    f['table'] = odict([])
    f_ev = {}
    for entries in product(*[outcomeSpace for node in f['dom']]):
        f['table'][entries] = 0
    learn_next_proba_from_data(f, df[Adj[R]])
    factor_to_csv(f)

# Conditional Probabilities of Evidence given Rooms
for E in Sig.keys():
    for R in Sig[E]:
        f = {}
        f['dom'] = (R, E)
        f['table'] = odict([])
        for entries in product(*[outcomeSpace for node in f['dom']]):
            f['table'][entries] = 0
        learn_cond_ev_from_data(f, df[[R,E]])
        factor_to_csv(f)

for W in ['robot1', 'robot2']:
    f = {}
    learn_robot_data(f, W, df)
    factor_to_csv(f)
