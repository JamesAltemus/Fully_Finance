# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:44:55 2019

@author: James Altemus
"""
import warnings

import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as st

from copy import deepcopy
from scipy.optimize import minimize

import optimizercore as oc
from core import _embed_periodicity, _geomean

class Beta:
    def __init__(self, feed_dict, full_analysis):
        # initiate common vatiables
        self.returns = feed_dict['input']
        self.covar = self.returns.cov()
        del feed_dict['input']
        columns = []
        for col in self.returns.columns:
            columns.append(col.split('_')[0])
        
        self.bins = oc.get_bins(self.returns)
        
        # save parameters in case replication is desired
        self.Parameters = deepcopy(feed_dict)
        self.Parameters.update({'num_bins': deepcopy(self.bins)})
        objective,features,constraints,simulations,methods,tol,maxiter,disp = oc.unfeeder(feed_dict)
        
        if objective['target']:
            self.target = objective['target']
        else:
            self.target = 0.0
        
        # features
        if 'benchmark' in features:
            self.benchmark = features['benchmark']
        else:
            raise AttributeError('Beta requires a benchmark.')
        
        self.periodicity = _embed_periodicity(features['periodicity'])
        if methods == 'arithmetic':
            self.mean_returns = self.returns.mean()*self.periodicity
        if methods == 'geometric':
            self.mean_returns = _geomean(self.returns, self.periodicity)
        
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
        self.OptiParam = minimize(self.optimize_beta, self.guess, method = 'SLSQP',
                                  bounds = self.weight_bounds, tol = tol,
                                  constraints = constraints,
                                  options = {'maxiter':maxiter, 'disp':disp})
        if not self.OptiParam['success']:
            print('Warning!', self.OptiParam['message'])
        tanw = self.OptiParam['x']
        tanww = pd.DataFrame(tanw).T
        tanww.columns = columns
        tanr = self.weighted_annual_return(tanw)
        margin = np.sum(np.abs(tanw))-1
        tanra = tanr-self.margin_rate*margin
        tans = self.weighted_annual_stddev(tanw)
        tanss = self.weighted_return(tanw)
        beta = self.benchmark.cov(tanss)/self.benchmark.std()
        self.Optimal = {'Weights': tanww, 'Return': tanr, 'AdjReturn': tanra, 
                        'StdDev': tans, 'Beta': beta, 'Series': tanss}
        
        # Beta frontier
        if full_analysis == True:
            analyze = np.linspace(max(self.Optimal['Return']/4,0.0001),
                                  min(self.Optimal['Return']*4,0.5),simulations)
            efficent = []
            eff = []
            for ret in analyze:
                self.ret = ret
                e = self.optimize_betas(ret, constraints, tol, maxiter, disp)
                we = e['x']
                wew = pd.DataFrame(we).T
                wew.columns = columns
                margin = np.sum(np.abs(we))-1
                re = self.weighted_annual_return(we)-self.margin_rate*margin
                se = self.weighted_annual_stddev(we)
                be = self.benchmark.cov(self.weighted_return(we))/self.benchmark.std()
                efficent.append({'Weights': wew, 'Returns': re, 'StdDevs': se, 'Beta': be})
                eff.append(e)
            self.EfficentParam = eff
            self.EfficentFrontier = efficent
        
        # Constituents
        self.ConstituentBeta = self.get_betas(columns)
    
    
    def optimize_beta(self, weights):
        margin = np.sum(np.abs(weights))-1
        returns = self.weighted_return(weights) - (margin*self.margin_rate)/self.periodicity
        return (self.target - (self.benchmark.cov(returns)/(self.benchmark.std()**2)))**2
    
    
    def optimize_betas(self, ret, constraints, tol, maxiter, disp):
        constraint = [c for c in constraints]
        constraint.append({'type': 'eq', 'fun': lambda x: self.weighted_annual_return(x) - ret})
        constraint = tuple(constraint)
        efficent = minimize(self.optimize_beta, self.guess, method = 'SLSQP',
                            bounds = self.weight_bounds, tol = tol,
                            constraints = constraint,
                            options = {'maxiter':maxiter, 'disp':disp})
        return efficent
    
    
    def get_betas(self, final_cols):
        Beta = pd.DataFrame()
        Beta['Return'] = self.mean_returns
        Beta['StdDev'] = (self.returns.std()*np.sqrt(self.periodicity))
        
        cols = self.returns.columns
        beta = []
        for col in cols:
            beta.append(self.benchmark.cov(self.returns[col])/self.benchmark.std())
        Beta['Beta'] = beta
        Beta.index = final_cols
        return Beta
    
    
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
