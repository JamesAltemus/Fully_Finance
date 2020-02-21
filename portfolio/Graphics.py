# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:17:08 2019

@author: James Altemus
"""

import pandas as pd
import numpy as np
import matplotlib as matplt
import matplotlib.pyplot as plt
import seaborn as sea

from core import _accumulate_returns

def plot_efficent_frontier_weights(self, size = (12,8), marker = 's', linestyle = '--',
                 title = 'Weights of the Efficent Frontier'):
    weights = self.Optimal['Weights']
    weights = weights.append(self.EfficentFrontier[0]['Weights'])
    weights = weights.append(self.EfficentFrontier[-1]['Weights'])
    weights = weights.T
    weights.columns = ['Tangent','Min','Max']
    
    plt.figure(figsize=size)
    plt.plot(weights, marker = marker, linestyle = linestyle)
    plt.title(title)
    plt.ylabel('Weights')
    plt.legend(['Tangent Portfolio','Min Return','Max Return'])
    plt.show()


def plot_efficent_frontier(self, size = (12,8), title = 'Efficent Frontier'):
    data = [(point['StdDevs'],point['Returns']) for point in self.EfficentFrontier]
    std, ret = list(zip(*data))
    
    plt.figure(figsize=size)
    plt.scatter(x = std, y = ret, label = None)
    plt.scatter(x=self.Optimal['StdDev'], y=self.Optimal['AdjReturn'], label = 'Tangent Portfolio')
    plt.title(title)
    plt.xlabel('Standard Deviation')
    plt.ylabel('Return')
    plt.legend()
    plt.show()


def plot_tangency(self, size = (12,8), title = 'Tangent Portfolio Returns'):
    cum_rets = _accumulate_returns(self.Optimal['Series'])
    
    plt.figure(figsize=size)
    plt.plot(cum_rets, label = None)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.show()


def plot_tangency_distribution(self, size = (12,8), title = 'Tangent Portfolio Returns', color = 'c'):
    bins = self.Parameters['num_bins']
    if self.Parameters['objective']['target'][1]:
        self.Optimal['Series'].hist(bins = bins, figsize = size)
    else:
        import scipy.stats as stat
        
        dist = getattr(stat, self.Optimal['Distribution'][0])
        loc = self.Optimal['Distribution'][1][-2]
        scale = self.Optimal['Distribution'][1][-1]
        args = self.Optimal['Distribution'][1][:-2]
        
        st = dist.ppf(0.01, loc=loc, scale=scale, *args) if args else dist.ppf(0.01, loc=loc, scale=scale)
        end = dist.ppf(0.99, loc=loc, scale=scale, *args) if args else dist.ppf(0.99, loc=loc, scale=scale)
        
        x = np.linspace(st, end, 10000)
        pdf = dist.pdf(x, loc=loc, scale=scale, *args)
        
        plt.figure(figsize = size)
        ax = pd.Series(pdf, x).plot(label = 'Probability Density', legend = True)
        self.Optimal['Series'].plot(kind = 'hist', bins = bins, alpha = 0.5,
                    color = color, label = 'Tangent Portfolio', legend = True, ax = ax)
        ax.set_title(title)
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')


def plot_portfolio(self, size = (12,8), title = 'Portfolio Returns'):
    cum_rets = _accumulate_returns(self.Portfolio)
    
    plt.figure(figsize=size)
    plt.plot(cum_rets, label = None)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.show()


def plot_all(self, size = (12,8), title = 'Returns'):
    cum_rets = _accumulate_returns(self.Portfolio)
    for col in self.returns.columns:
        s_ret = _accumulate_returns(self.returns[col])
        cum_rets = pd.concat((cum_rets, s_ret),axis = 1)
    idx = ['Portfolio_Returns']
    idx.extend(self.returns.columns)
    cum_rets.columns = idx
    
    plt.figure(figsize=size)
    plt.plot(cum_rets, label = None)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend(idx)
    plt.show()