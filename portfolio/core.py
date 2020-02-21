# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:02:09 2019

@author: James Altemus
"""
import numpy as np
import pandas as pd


def _accumulate_returns(returns):
    returns = returns + 1
    cum_rets = [1]
    for ret in returns:
        cum_rets.append(cum_rets[-1]*ret)
    cum_rets = pd.DataFrame(cum_rets[1:])
    cum_rets.index = returns.index
    return cum_rets


def _embed_periodicity(periodicity):
    if periodicity == 'daily':
        return 252
    if periodicity == 'weekly':
        return 52
    if periodicity == 'monthly':
        return 12
    if periodicity == 'yearly':
        return 1


def _geomean(returns, periodicity):
    return ((returns+1).prod(axis = 0)**(1/len(returns))-1)*periodicity


class PortfolioBuilder:
    def __init__(self, portfolio, periodicity = 'daily'):
        '''periodicity should be "daily", "weekly", "monthly" or "yearly"'''
        self.periodicity = periodicity
        self.prices = portfolio
        self.tickers = portfolio.columns
    
    
    def calculate_returns_arith(self):
        '''Calculates the arithmetic returns for a series of data.'''
        self.returns = self.prices.pct_change()
        self.returns = self.returns.dropna()
        self.returns.columns = self.prices.columns + '_Return'
    
    
    def calculate_returns_log(self):
        '''Calculates the logarithmic returns for a series of data.'''
        self.log_returns = np.log(self.prices) - np.log(self.prices.shift(1))
        self.log_returns = self.log_returns.dropna()
        self.log_returns.columns = self.prices.columns + '_Log_Return'
    
    
    def build_portfolio(self, weights, method = 'arithmetic'):
        self.weights = weights
        self.portfolio = (weights*self.returns).sum(axis = 1)
        
        period = _embed_periodicity(self.periodicity)
        
        if method.lower() == 'arithmetic':
            mean_returns = self.returns.mean()*period
        if method.lower() == 'geometric':
            mean_returns = _geomean(self.returns, period)
        
        self.portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov(), weights)))*np.sqrt(period)
        self.portfolio_mean = (weights*mean_returns).sum()
        import Graphics
        setattr(PortfolioBuilder, 'plot_portfolio', Graphics.plot_portfolio)
        setattr(PortfolioBuilder, 'plot_all', Graphics.plot_all)
    
    
    def add_analytics(self):
        '''Allows indicators to be calculated for the portfolio.'''
        import Analytics
        setattr(PortfolioBuilder, 'calculate_bollinger', Analytics.bollinger)
        setattr(PortfolioBuilder, 'calculate_chande_mo', Analytics.chande_mo)
        setattr(PortfolioBuilder, 'calculate_ichimoku', Analytics.ichimoku)
        setattr(PortfolioBuilder, 'calculate_macd', Analytics.macd)
        setattr(PortfolioBuilder, 'calculate_rsi', Analytics.rsi)
        setattr(PortfolioBuilder, 'calculate_stochastic', Analytics.stochastic)
        setattr(PortfolioBuilder, 'calculate_wilder_smoothing', Analytics.wilder_smoothing)
    
    
    def add_optimization(self, method = 'logarithmic'):
        '''Allows portfolio optimization to be perfomed on the portfolio.'''
        from Optimization import OptimizePortfolio
        
        if method.lower() in ['a','arithmetic']:
            try:
                self.returns
            except:
                self.calculate_returns_arith()
            
            self.OptimizePortfolio = OptimizePortfolio(self.returns, 'arithmetic', self.periodicity)
        
        if method.lower() in ['l','logarithmic']:
            try:
                self.log_returns
            except:
                self.calculate_returns_log()
            
            self.OptimizePortfolio = OptimizePortfolio(self.log_returns, 'logarithmic', self.periodicity)