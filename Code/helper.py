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

"""
Self-defined Helper Files
"""

def smoothen(f):
    """
    argument
        f, factor to be smoothened (has 0 probabilities)
    returns
        new, smoothed factor
    assumes that factor has only one variable.
    """
    table = list()
    for row in f['table'].items():
        if row[1] == 0:
            table.append((row[0]), 0.0001)
        elif row[1] == 1:
            table.append((row[0]), 0.9999)
        else:
            raise Exception("Failed.")
    return {'dom': f['dom'], 'table':odict(table)}

def factor_to_csv(f):
    """
    argument
        `f`, a factor to save to a csv file.
    returns
        None (saves a file)
    The first row of the csv file will be the variables; followed by the factor
    """
    cols = list(f['dom'])
    cols.append('')
    O = f['table']
    A = np.zeros((len(O), len(cols)))
    for i, (key, val) in enumerate(O.items()):
        A[i] = [*key, val]
    df = pd.DataFrame(A, columns=cols)
    df.to_csv('_'.join(cols)+'.csv', index=False, header=True)

def csv_to_factor(filename):
    """
    argument
        `filename`, a csv file to convert to a factor
    returns
        new factor defined by csv
    The first row of the csv file contains the columns headers, followed by the data
    """
    f = {'dom':None, 'table':odict([])}
    with open(filename, "r") as file:
        for i, line in enumerate(file):
            line = line.strip()
            L = line.split(',')
            L = [v for v in L if v]
            if i == 0: #header
                f['dom'] = tuple(L)
            else: #data
                L = [float(v) for v in L]
                f['table'][tuple(L[:-1])] = L[-1]
    return f

def convert_observation(obs):
    """
    arguments
        obs: dictionary of sensors to their oberserved value
    returns
        new dictionary, with observed values mapped to 0/1
    """
    new_obs = obs.copy()
    for E in obs.keys():
        if E[:5] == 'relia' or E[:5] == 'unrel':
            if obs[E] == 'motion':
                new_obs[E] = 1
            elif obs[E] == 'no motion':
                new_obs[E] = 0
            else:
                new_obs[E] = None
        elif E[:5] == 'door_':
            if obs[E] != 0:
                new_obs[E] = 1
            elif obs[E] == 0:
                new_obs[E] = 0
            else:
                new_obs[E] = None
        elif E[:5] == 'robot':
            if obs[E] == None:
                new_obs[E] = None
            else:
                robot_obs = eval(obs[E])
                new_obs[E] = (robot_obs[0], 1) if robot_obs[1] > 0 else (robot_obs[0], 0)
        else:
            new_obs[E] = obs[E]
    return new_obs

def learn_next_proba_from_data(f, data):
    """
    Learns probabilities in a factor from the data. Update is done in-place.
    Note that we are learning
            P(X_t | Adj(X,t-1)) = P(X_t, Adj(X,t-1)) / P(Adj(X,t-1))
    """
    # variables
    all_vars = f['dom']
    prev_vars = f['dom'][:-1]
    R = f['dom'][-1]
    # create a dataframe with columns [X_t, Adj(X_t-1)]
    columns = f['dom']
    df = data.copy()
    df.columns = columns[:-1]
    df[R] = df[R.replace('+1', '')]
    df[R] = df[R].shift(-1)
    df = df.iloc[:-1,:]
    total = df.shape[0]

    all_entries = list(product(*[outcomeSpace for node in f['dom']]))
    prev_entries = list(product(*[outcomeSpace for node in f['dom'][:-1]]))

    #for each (pair of) rows, increment count as appropriate
    for i in range(df.shape[0]):
        row = df.iloc[i,:]
        f['table'][tuple(row[all_vars])] += 1

    #smooth
    for entries in all_entries:
        c = f['table'][entries]
        f['table'][entries] = (c + 1)/(total + len(f['table']))

    #final probability
    for entries in prev_entries:
        entries0, entries1 = tuple(list(entries) + [0]), tuple(list(entries) + [1])
        N = f['table'][entries0] + f['table'][entries1]
        f['table'][entries0] /= N
        f['table'][entries1] /= N

def learn_cond_ev_from_data(f, data):
    """
    Learn the conditional P(E|R), where R is in in Sig[E]
    """
    # variables
    all_vars = f['dom']
    R = f['dom'][-1]
    total = data.shape[0]

    all_entries = list(product(*[outcomeSpace for node in f['dom']]))

    #for each (pair of) rows, increment count as appropriate
    for i in range(data.shape[0]):
        row = data.iloc[i,:]
        f['table'][tuple(row)] += 1

    #smooth
    for entries in all_entries:
        c = f['table'][entries]
        f['table'][entries] = (c + 1)/(total + len(f['table']))

    #final probability
    for entries in product(*[outcomeSpace]):
        entries0, entries1 = tuple(list(entries) + [0]), tuple(list(entries) + [1])
        N = f['table'][entries0] + f['table'][entries1]
        f['table'][entries0] /= N
        f['table'][entries1] /= N

def learn_robot_data(f, W, data):
    total = data.shape[0]
    f['dom'] = (W,)
    f['table'] = odict([
        ((0,),0),
        ((1,),0)
    ])
    for i in range(total):
        row = data.iloc[i,:]
        R = row[W][0]
        if R != 'c3':
            if row[R] == row[W][1]:
                f['table'][(1,)] += 1
            else:
                f['table'][(0,)] += 1
        else:
            if row['c3a'] == row[W][1]:
                f['table'][(1,)] += 1
            else:
                f['table'][(0,)] += 1
    f['table'][(0,)] = (f['table'][(0,)] + 1) / (total + 2)
    f['table'][(1,)] = (f['table'][(1,)] + 1) / (total + 2)


"""
Helper Files: taken from tutorials
"""

def printFactor(f):
    """
    argument
        `f`, a factor to print on screen
    returns
        None
    """
    table = list()
    for key, item in f['table'].items():
        k = list(key)
        k.append(item)
        table.append(k)
    dom = list(f['dom'])
    dom.append('Pr')
    print(tabulate(table,headers=dom,tablefmt='orgtbl'))

def prob(factor, *entry):
    """
    argument
        `factor`, a dictionary of domain and probability table,
        `entry`, a list of values, one for each variable, in the same order as specified in the factor domain.
    returns
        p(entry)
    """
    return factor['table'][entry]

def evidence(E, obs, outcomeSpace):
    """
    argument
        `E`, variable.
        `e`, the observed value for E.
        `outcomeSpace`, dictionary with the domain of each variable
    returns
        newOutcomeSpace: new outcomeSpace with update made to E.
    """
    newOutcomeSpace = outcomeSpace.copy()
    newOutcomeSpace[E] = (obs,)
    return newOutcomeSpace

def marginalize_room(f, var, outcomeSpace):
    """
    argument
        `f`, factor to be marginalized.
        `var`, variable to be summed out.
        `outcomeSpace`, dictionary with the domain of each variable
    returns
        new factor f' with dom(f') = dom(f) - {var}
    assumes that the variable to be eliminated is at the front of the domain (asserted), and is binary
    """

    new_dom = tuple(f['dom'][1:])
    vars_to_keep = 2**(len(f['dom']) - 1)
    items = list(f['table'].items())
    table = [(items[i][0][1:], items[i][1] + items[i + vars_to_keep][1]) for i in range(vars_to_keep)]
    return {'dom': new_dom, 'table': odict(table)}

def marginalize_sensor(f, var, outcomeSpace):
    """
    argument
        `f`, factor to be marginalized.
        `var`, variable to be summed out.
        `outcomeSpace`, dictionary with the domain of each variable
    returns
        new factor f' with dom(f') = dom(f) - {var}
    assumes that the variable to be eliminated is at the front of the domain (asserted), and is binary
    """
    new_dom = list(f['dom'])
    new_dom.remove(var)
    table = list()
    for entries in product(*[outcomeSpace[node] for node in new_dom]):
        s = 0;
        S = []
        for val in outcomeSpace[var]:
            entriesList = list(entries)
            entriesList.insert(f['dom'].index(var), val)
            p = prob(f, *entriesList)
            s = s + p
        table.append((entries, s))
    return {'dom': tuple(new_dom), 'table': odict(table)}


def normalize(f, outcomeSpace):
    """
    argument
        `f`, factor to be normalized.
    returns
        new factor as a copy of f with entries that sum to 1, over the dependent variable
    """
    new_f = {}
    new_f['dom'] = f['dom']
    new_f['table'] = odict([])
    vars = f['dom'][:-1]
    for entries in product(*[outcomeSpace[V] for V in vars]):
        entries0, entries1 = tuple(list(entries) + [0]), tuple(list(entries) + [1])
        N = f['table'][entries0] + f['table'][entries1]
        #if N == 0:
        #    continue
        new_f['table'][entries0] = f['table'][entries0] / N
        new_f['table'][entries1] = f['table'][entries1] / N
    return new_f


def join(f1, f2, outcomeSpace):
    """
    argument
        `f1`, first factor to be joined.
        `f2`, second factor to be joined.
        `outcomeSpace`, dictionary with the domain of each variable
    returns
        new factor with a join of f1 and f2
    """

    Vars = f2['dom']

    table = [(entries, f2['table'][entries] * f1['table'][(entries[0],)]) for entries in product(*[[0,1] for V in Vars])]

    return {'dom': Vars, 'table': odict(table)}