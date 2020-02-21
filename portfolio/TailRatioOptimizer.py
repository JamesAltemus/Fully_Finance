# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:31:15 2019

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

class TailRatio:
    '''The expected exraordinary gain/risk of excessive loss. target is the one tail probability.
    If target is None, the probability will be 5%. Maximized'''
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
            self.confidence = objective['target'][0]
            self.dist_type = objective['target'][1]
            self.type = objective['target'][2]
        else:
            self.confidence = 0.05
            self.dist_type = False
            self.type = False
        
        # features
        self.periodicity = _embed_periodicity(features['periodicity'])
        if methods == 'arithmetic':
            self.mean_returns = self.returns.mean()*self.periodicity
        if methods == 'geometric':
            self.mean_returns = _geomean(self.returns, self.periodicity)
        
        if 'benchmark' in features:
            self.benchmark = features['benchmark']
        
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
        self.OptiParam = minimize(self.optimize_tr, self.guess, method = 'SLSQP',
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
        if self.dist_type == False:
            vtr, etr, dist = self.optimize_tr(tanw, True)
            self.Optimal = {'Weights': tanww, 'Return': tanr, 'AdjReturn': tanra, 
                            'StdDev': tans, 'TailRatio': vtr,
                            'ExprectedTailRatio': etr, 'Distribution': dist,
                            'Series': self.weighted_return(tanw)}
        else:
            vtr, etr, = self.optimize_tr(tanw, True)
            self.Optimal = {'Weights': tanww, 'Return': tanr, 'AdjReturn': tanra, 
                            'StdDev': tans, 'TailRatio': vtr,
                            'ExprectedTailRatio': etr,
                            'Series': self.weighted_return(tanw)}
        
        # TR frontier
        if full_analysis == True:
            analyze = np.linspace(max(self.Optimal['Return']/4,0.0001),
                                  min(self.Optimal['Return']*4,0.5),simulations)
            efficent = []
            eff = []
            for ret in analyze:
                self.ret = ret
                e = self.optimize_trs(ret, constraints, tol, maxiter, disp)
                we = e['x']
                wew = pd.DataFrame(we).T
                wew.columns = columns
                margin = np.sum(np.abs(we))-1
                re = self.weighted_annual_return(we)-self.margin_rate*margin
                se = self.weighted_annual_stddev(we)
                efficent.append({'Weights': wew, 'Returns': re, 'StdDevs': se, 'TRs': -self.optimize_tr(we)})
                eff.append(e)
            self.EfficentParam = eff
            self.EfficentFrontier = efficent
        
        # close
        self.ConstituentTailRatios = self.get_trs(columns)
    
    
    def optimize_tr(self, weights, ret_all = False):
        margin = np.sum(np.abs(weights))-1
        periodic_margin_cost = (margin*self.margin_rate)/self.periodicity
        returns = self.weighted_return(weights) - periodic_margin_cost
        
        if self.dist_type == False:
            if self.bins < 100:
                dist_list = [st.triang, st.dgamma, st.levy_stable, st.gennorm]
            else:
                dist_list = [st.norm, st.johnsonsu, st.t]
            dist = oc.get_distribution(returns, self.bins, dist_list)
            loc, scale, args = oc.unpack_dist_params(dist[1])
            VaR = dist[0].ppf(self.confidence, loc=loc, scale=scale, *args)
            ES = dist[0].ppf(np.linspace(0.0,self.confidence,self.bins*10), loc=loc, scale=scale, *args)[1:].mean()
            VaE = dist[0].ppf(1-self.confidence, loc=loc, scale=scale, *args)
            EP = dist[0].ppf(np.linspace(1-self.confidence,1.0,self.bins*10), loc=loc, scale=scale, *args)[:-1].mean()
            if ret_all:
                return VaE/VaR, EP/ES, (dist[0].name, dist[1], dist[2])
            elif self.type:
                return ES/EP
            else:
                return VaR/VaE
        else:
            VaR = np.percentile(returns, self.confidence*100)
            ES = returns[returns<=VaR].mean()
            VaE = np.percentile(returns, (1-self.confidence)*100)
            EP = returns[returns>=VaE].mean()
            if ret_all:
                return VaE/VaR, EP/ES
            elif self.type:
                return ES/EP
            else:
                return VaR/VaE
    
    
    def optimize_trs(self, ret, constraints, tol, maxiter, disp):
        constraint = [c for c in constraints]
        constraint.append({'type': 'eq', 'fun': lambda x: self.weighted_annual_return(x) - ret})
        constraint = tuple(constraint)
        efficent = minimize(self.optimize_tr, self.guess, method = 'SLSQP',
                            bounds = self.weight_bounds, tol = tol,
                            constraints = constraint,
                            options = {'maxiter':maxiter, 'disp':disp})
        return efficent
    
    
    def get_trs(self, final_cols):
        cvar = pd.DataFrame()
        cvar['Return'] = self.mean_returns
        cvar['StdDev'] = (self.returns.std()*np.sqrt(self.periodicity))
        
        cols = self.returns.columns
        tr = []
        etr = []
        if self.dist_type:
            distribution = []
            locs = []
            scales = []
            argss = []
            errs = []
            for col in cols:
                ret = self.returns[col]
                if self.bins < 100:
                    dist_list = [st.triang, st.dgamma, st.levy_stable, st.gennorm]
                else:
                    dist_list = [st.norm, st.johnsonsu, st.t]
                dist = oc.get_distribution(ret, self.bins, dist_list)
                loc, scale, args = oc.unpack_dist_params(dist[1])
                VaR = dist[0].ppf(self.confidence, loc=loc, scale=scale, *args)
                ES = dist[0].ppf(np.linspace(0.0,self.confidence,self.bins*10), loc=loc, scale=scale, *args)[1:].mean()
                VaE = dist[0].ppf(1-self.confidence, loc=loc, scale=scale, *args)
                EP = dist[0].ppf(np.linspace(1-self.confidence,1.0,self.bins*10), loc=loc, scale=scale, *args)[:-1].mean()
                tr.append(VaE/VaR)
                etr.append(EP/ES)
                distribution.append(dist[0].name)
                locs.append(loc)
                scales.append(scale)
                argss.append(args)
                errs.append(dist[2])
            cvar['Tail_Ratio'] = tr
            cvar['Expected_Tail_Ratio'] = etr
            cvar['Distribution'] = distribution
            cvar['Loc'] = locs
            cvar['Scale'] = scales
            cvar['*args'] = argss
            cvar['Fitted_Error'] = errs
        else:
            for col in cols:
                ret = self.returns[col]
                VaR = np.percentile(ret, self.confidence*100)
                ES = ret[ret<=VaR].mean()
                VaE = np.percentile(ret, (1-self.confidence)*100)
                EP = ret[ret>=VaE].mean()
                tr.append(VaE/VaR)
                etr.append(EP/ES)
            cvar['Tail_Ratio'] = tr
            cvar['Expected_Tail_Ratio'] = etr
        cvar.index = final_cols
        return cvar
    
    
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