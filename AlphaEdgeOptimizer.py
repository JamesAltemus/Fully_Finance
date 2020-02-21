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

class AlphaEdge:
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
            self.dist_type = objective['target']
        else:
            self.dist_type = False
        
        # features
        if 'benchmark' in features:
            self.benchmark = features['benchmark']
        else:
            raise AttributeError('AlphaEdge requires a benchmark.')
        
        self.periodicity = _embed_periodicity(features['periodicity'])
        if methods == 'arithmetic':
            self.mean_returns = self.returns.mean()*self.periodicity
            self.benchmark_mean = self.benchmark.mean()*self.periodicity
        if methods == 'geometric':
            self.mean_returns = _geomean(self.returns, self.periodicity)
            self.benchmark_mean = _geomean(self.benchmark, self.periodicity)
        
        ## Benchmark AE param
        if self.dist_type == False:
            if self.bins < 100:
                dist_list = [st.triang, st.dgamma, st.levy_stable, st.gennorm]
            else:
                dist_list = [st.norm, st.johnsonsu, st.t]
            dist = oc.get_distribution(self.benchmark, self.bins, dist_list)
            loc, scale, args = oc.unpack_dist_params(dist[1])
            self.benchmark_p_loss = dist[0].cdf(0, loc = loc, scale = scale, *args)
            self.benchmark_p_profit = 1-self.benchmark_p_loss
            self.benchmark_alphaedge = self.benchmark_mean * self.benchmark_p_profit/self.benchmark_p_loss
            self.benchmark_dist = (dist[0].name, dist[1], dist[2])
        else:
            self.benchmark_p_loss = st.percentileofscore(self.benchmark, 0)/100
            self.benchmark_p_profit = 1-self.benchmark_p_loss
            self.benchmark_alphaedge = self.benchmark_mean * self.benchmark_p_profit/self.benchmark_p_loss
        
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
        self.OptiParam = minimize(self.optimize_ae, self.guess, method = 'SLSQP',
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
            ae, a, pp, pl, dist = self.optimize_ae(tanw, True)
            self.Optimal = {'Weights': tanww, 'Return': tanr, 'AdjReturn': tanra, 
                            'StdDev': tans, 'AlphaEdge': ae, 'Alpha': a,
                            'Prob_Profit': pp, 'Prob_Loss': pl, 'Distribution': dist,
                            'Series': self.weighted_return(tanw)}
        else:
            ae, a, pp, pl = self.optimize_var(tanw, True)
            self.Optimal = {'Weights': tanww, 'Return': tanr, 'AdjReturn': tanra, 
                            'StdDev': tans, 'AlphaEdge': ae, 'Alpha': a,
                            'Prob_Profit': pp, 'Prob_Loss': pl,
                            'Series': self.weighted_return(tanw)}
        
        # AlphaEdge frontier
        if full_analysis == True:
            analyze = np.linspace(max(self.Optimal['Return']/4,0.0001),
                                  min(self.Optimal['Return']*4,0.5),simulations)
            efficent = []
            eff = []
            for ret in analyze:
                self.ret = ret
                e = self.optimize_vars(ret, constraints, tol, maxiter, disp)
                we = e['x']
                wew = pd.DataFrame(we).T
                wew.columns = columns
                margin = np.sum(np.abs(we))-1
                re = self.weighted_annual_return(we)-self.margin_rate*margin
                se = self.weighted_annual_stddev(we)
                efficent.append({'Weights': wew, 'Returns': re, 'StdDevs': se, 'AlphaEdge': -self.optimize_ae(we)})
                eff.append(e)
            self.EfficentParam = eff
            self.EfficentFrontier = efficent
        
        # Constituents
        self.ConstituentAlphaEdge = self.get_aes(columns)
    
    
    def optimize_ae(self, weights, ret_all = False):
        margin = (np.sum(np.abs(weights))-1)*self.margin_rate
        year_ret = self.weighted_annual_return(weights) - margin
        alpha = year_ret - self.benchmark_mean
        returns = self.weighted_return(weights) - margin/self.periodicity
        if self.dist_type == False:
            if self.bins < 100:
                dist_list = [st.triang, st.dgamma, st.levy_stable, st.gennorm]
            else:
                dist_list = [st.norm, st.johnsonsu, st.t]
            dist = oc.get_distribution(returns, self.bins, dist_list)
            loc, scale, args = oc.unpack_dist_params(dist[1])
            p_loss = dist[0].cdf(0, loc = loc, scale = scale, *args)
            p_profit = 1-p_loss
            alphaedge = year_ret * p_profit/p_loss - self.benchmark_alphaedge
            if ret_all:
                return alphaedge, alpha, p_profit, p_loss, (dist[0].name, dist[1], dist[2])
            else:
                return -alphaedge
        else:
            p_loss = st.percentileofscore(returns, 0)/100
            p_profit = 1-p_loss
            alphaedge = year_ret * p_profit/p_loss - self.benchmark_alphaedge
            if ret_all:
                return alphaedge, alpha, p_profit, p_loss
            else:
                return -alphaedge
    
    
    def optimize_aes(self, ret, constraints, tol, maxiter, disp):
        constraint = [c for c in constraints]
        constraint.append({'type': 'eq', 'fun': lambda x: self.weighted_annual_return(x) - ret})
        constraint = tuple(constraint)
        efficent = minimize(self.optimize_var, self.guess, method = 'SLSQP',
                            bounds = self.weight_bounds, tol = tol,
                            constraints = constraint,
                            options = {'maxiter':maxiter, 'disp':disp})
        return efficent
    
    
    def get_aes(self, final_cols):
        alphaedges = pd.DataFrame()
        alphaedges['Return'] = self.mean_returns
        alphaedges['StdDev'] = self.returns.std()*np.sqrt(self.periodicity)
        
        cols = self.returns.columns
        ae = []
        pp = []
        pl = []
        if self.dist_type == False:
            distribution = []
            locs = []
            scales = []
            argss = []
            errs = []
            for col in cols:
                ret = self.returns[col]
                year_ret = self.mean_returns[col]
                if self.bins < 100:
                    dist_list = [st.triang, st.dgamma, st.levy_stable, st.gennorm]
                else:
                    dist_list = [st.norm, st.johnsonsu, st.t]
                dist = oc.get_distribution(ret, self.bins, dist_list)
                loc, scale, args = oc.unpack_dist_params(dist[1])
                p_loss = dist[0].cdf(0, loc = loc, scale = scale, *args)
                p_profit = 1-p_loss
                alphaedge = year_ret * p_profit/p_loss - self.benchmark_alphaedge
                
                ae.append(alphaedge)
                pp.append(p_profit)
                pl.append(p_loss)
                distribution.append(dist[0].name)
                locs.append(loc)
                scales.append(scale)
                argss.append(args)
                errs.append(dist[2])
            alphaedges['AlphaEdge'] = ae
            alphaedges['Alpha'] = self.mean_returns - self.benchmark_mean
            alphaedges['Prob_Profit'] = pp
            alphaedges['Prob_Loss'] = pl
            alphaedges['Loc'] = locs
            alphaedges['Scale'] = scales
            alphaedges['*args'] = argss
            alphaedges['Fitted_Error'] = errs
        else:
            for col in cols:
                ret = self.returns[col]
                year_ret = self.mean_returns[col]
                p_loss = st.percentileofscore(ret, 0)/100
                p_profit = 1-p_loss
                alphaedge = year_ret * p_profit/p_loss - self.benchmark_alphaedge
                
                ae.append(alphaedge)
                pp.append(p_profit)
                pl.append(p_loss)
            alphaedges['AlphaEdge'] = ae
            alphaedges['Alpha'] = self.mean_returns - self.benchmark_mean
            alphaedges['Prob_Profit'] = pp
            alphaedges['Prob_Loss'] = pl
        alphaedges.index = final_cols
        return alphaedges
    
    
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
