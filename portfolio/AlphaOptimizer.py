# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:44:55 2019

@author: James Altemus
"""
import pandas as pd
import numpy as np

from copy import deepcopy
from scipy.optimize import minimize

import optimizercore as oc
from core import _embed_periodicity, _geomean

class Alpha:
    def __init__(self, feed_dict):
        # initiate common vatiables
        self.returns = feed_dict['input']
        self.covar = self.returns.cov()
        del feed_dict['input']
        columns = []
        for col in self.returns.columns:
            columns.append(col.split('_')[0])
        
        # save parameters in case replication is desired
        self.Parameters = deepcopy(feed_dict)
        objective,features,constraints,simulations,methods,tol,maxiter,disp = oc.unfeeder(feed_dict)
        
        # features
        if 'benchmark' in features:
            self.benchmark = features['benchmark']
        else:
            raise AttributeError('Alpha requires a benchmark.')
        
        self.periodicity = _embed_periodicity(features['periodicity'])
        if methods == 'arithmetic':
            self.mean_returns = self.returns.mean()*self.periodicity
            self.benchmark_mean = self.benchmark.mean()*self.periodicity
        if methods == 'geometric':
            self.mean_returns = _geomean(self.returns, self.periodicity)
            self.benchmark_mean = _geomean(self.benchmark, self.periodicity)
        
        if 'margin_int_rate' in features:
            self.margin_rate = features['margin_int_rate']
        else:
            self.margin_rate = 0
        
        # constraints
        if 'min_weights' in constraints:
            self.weight_bounds = tuple(zip(constraints['min_weights'], constraints['max_weights']))
        else:
            self.weight_bounds = None
        
        constraints = self.get_constraints(constraints)
        
        # optimal portfolio
        self.guess = np.random.normal(0, 0.5, len(self.mean_returns))/10
        self.OptiParam = minimize(self.optimize_alpha, self.guess, method = 'SLSQP',
                                  bounds = self.weight_bounds, tol = tol,
                                  constraints = constraints,
                                  options={'maxiter': maxiter, 'disp': disp})
        if not self.OptiParam['success']:
            print('Warning!', self.OptiParam['message'])
        tanw = self.OptiParam['x']
        tanww = pd.DataFrame(tanw).T
        tanww.columns = columns
        tanr = self.weighted_annual_return(tanw)
        margin = np.sum(np.abs(tanw))-1
        tanra = tanr-self.margin_rate*margin
        tans = self.weighted_annual_stddev(tanw)
        self.Optimal = {'Weights': tanww, 'Return': tanr, 'AdjReturn': tanra, 
                        'StdDev': tans, 'Alpha': tanra - self.benchmark_mean,
                        'Series': self.weighted_return(tanw)}
        
        # Constituents
        self.ConstituentAlpha = self.get_alphas(columns)
    
    
    def optimize_alpha(self, weights):
        margin = (np.sum(np.abs(weights))-1)*self.margin_rate
        return -(self.weighted_annual_return(weights) - margin - self.benchmark_mean)
    
    
    def get_alphas(self, final_cols):
        alpha = pd.DataFrame()
        alpha['Return'] = self.mean_returns
        alpha['StdDev'] = self.returns.std()*np.sqrt(self.periodicity)
        alpha['Alpha'] = self.mean_returns - self.benchmark_mean
        alpha.index = final_cols
        return alpha
    
    
    def get_constraints(self, constraints):
        constraint = []
        if 'required_return' in constraints:
            constraint.append({'type': 'ineq',
                               'fun': lambda x: self.weighted_annual_return(x)
                               -constraints['required_return']})
        
        if 'track_err' in constraints:
            constraint.append({'type': 'ineq', 'fun': lambda x: constraints['track_err']-self.tracking_error(x)})
        constraint.extend(oc.constrainer(constraints))
        return tuple(constraint)
    
    
    def weighted_return(self, weights):
        return (weights*self.returns).sum(axis = 1)
    
    
    def weighted_annual_stddev(self, weights):
        return np.sqrt(np.dot(weights.T, np.dot(self.covar, weights)))*np.sqrt(self.periodicity)
    
    
    def weighted_annual_return(self, weights):
        return (weights*self.mean_returns).sum()
    
    
    def tracking_error(self, weights):
        return np.std(self.weighted_return-self.benchmark)