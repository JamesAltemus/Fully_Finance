# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:30:12 2019

@author: James Altemus
"""
import warnings

import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as st

from copy import deepcopy

def unfeeder(feed_dict):
    return (feed_dict['objective'], feed_dict['features'], feed_dict['constraints'],
            feed_dict['simulations'], feed_dict['methods'], feed_dict['tol'],
            feed_dict['max_iter'], feed_dict['disp'])


def constrainer(constraints):
    constraint = []
    if 'total_margin_limit' in constraints:
        constraint.append({'type': 'ineq',
                           'fun': lambda x: constraints['total_margin_limit']
                           - np.sum(np.abs(x))-1})
    
    if 'min_cap' in constraints:
        constraint.append({'type': 'ineq', 'fun': lambda x: np.sum(x)-constraints['min_cap']})
        constraint.append({'type': 'ineq', 'fun': lambda x: constraints['max_cap']-np.sum(x)})
    
    if 'numb_long_limit' in constraints:
        constraint.append({'type': 'ineq', 'fun': lambda x: constraints['numb_long_limit']-len(x[x>0])})
        constraint.append({'type': 'ineq', 'fun': lambda x: constraints['numb_short_limit']-len(x[x<0])})
    return constraint


def get_bins(item):
    bins = len(item)
    if len(str(bins)) < 3:
        return round(bins/10)
    else:
        return 10 ** (len(str(bins))-2)


def unpack_dist_params(params):
    return params[-2], params[-1], params[:-2]


def get_distribution(data, bins, d_list):
    # get histogram values and edges
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0
    
    # set placeholders and find best distribution
    best = st.uniform
    best_params = (0.0, 1.0)
    best_err = np.inf
    for dist in d_list:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                params = dist.fit(data)
                loc, scale, args = unpack_dist_params(params)
                
                density = dist.pdf(x, loc=loc, scale=scale, *args)
                err = np.sum(np.power(y - density, 2.0))
                
                if best_err > err:
                    best = dist
                    best_params = params
                    best_err = err
        except:
            pass
    return tuple([best, best_params, best_err])