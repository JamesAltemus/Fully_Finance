# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 09:58:06 2019

@author: James Altemus
"""
class OptimizePortfolio:
    def __init__(self, returns, return_type, periodicity):
        '''return type should be "logarithmic" or "arithmetic"'''
        self.input = returns
        self.objective_parameters = {}
        self.features = {'return_type':return_type, 'periodicity': periodicity}
        self.constraints = {}
    
    
    def objective(self, objective_type = 'Sharpe_Ratio', target = None):
        '''Sets the objective for the portfolio.
        
        No benchmark:
            Sharpe_Ratio -- Maximizes the return per unit of standard deviation.
            
            Utility -- Maximizes the utility score - risk free rate. Target is the Risk Aversion parameter, default 2.5.
            
            VaR -- Minimizes the expected excessive loss. Target is a tuple of (the probability for excessive loss, True for true distribution or false for fit distribution, True for the Expected Shortfall or False for VaR), default (0.5, False, False).
            
            Tail_Ratio -- Maximizes the ratio of tail_gain/tail_loss. Target is the same format as VaR and has the same defaults.
        
        
        Benchmark required:
            Alpha -- Maximizes the portfolio return - benchmark return.
            
            Alpha_Edge -- Maximizes E[r]port * p(port>0)/p(port<0) - E[r]benchmark * p(benchmark>0)/p(benchmark<0). Target is True for true distribution or false for fit distribution.
            
            Beta -- Portfolio beta to the benchamrk. Target is the target beta, default 0.
            
            Black-Litterman -- Performs Black-Litterman optimization. Target should be the forward views held by the investor in the format [view1[abs % view, 0 (ignored) 1(positive change) or -1(negative change) for each ticker comma separeted], view2[...], etc...]. The benchmark is used to calculate the market price of risk.
            
            Sharpe_Edge -- (Maximizes E[r]port * p(port>0))/(sigma[port] * p(port<0)) - (E[r]benchmark * p(benchmark>0))/(sigma[benchmark] * p(benchmark<0)). Target is True for true distribution or false for fit distribution.
        '''
        self.objective_parameters.update({'objective': objective_type,
                                          'target': target})
    
    
    ## Features
    def feature_benchmark(self, benchmark = None, style = 'match'):
        '''Specifies a benchmark for the portfolio.
        Benchmark must be a list of returns of the same length and dates as the input returns.'''
        self.features.update({'benchmark': benchmark})
    
    
    def feature_margin_rate(self, margin_rate = 0.06):
        '''Specifies the interest rate charged on margin.'''
        self.features.update({'margin_int_rate': margin_rate})
    
    
    def feature_risk_free_rate(self, rf = 0.01):
        '''Specifies the risk free rate for computing ratios.'''
        self.features.update({'rf': rf})
    
    
    def remove_feature(self, remove = None):
        '''Removes the specified feature. The current list of constraints can be found in OptimalPortfolio.features.
        Features can be overridden instead of deleted by calling the function again'''
        del self.features[remove]
    
    
    ## Constraints
    def constraint_box_constraints(self, min_weights, max_weights):
        '''Specifies the minimum and maximum weights for each security.
        Arguments must be a list of the same length as the number of securities
        in the portfolio.'''
        self.constraints.update({'min_weights': min_weights,
                                 'max_weights': max_weights})
    
    
    def constraint_margin_amount_limit(self, margin_total = 0.5):
        '''Specifies the absolute amount of margin that may be used at a time.'''
        self.constraints.update({'total_margin_limit': margin_total})
    
    
    def constraint_net_invested_capital(self, max_cap = 1, min_cap = 1):
        '''Specifies the maximum amount to invest for the entire portfolio.
        The default is 1 for full investment. A number higher than 1 indicates
        using margin, lower than1 indicates holding cash. 0 indicates dollar neutral'''
        self.constraints.update({'min_cap': min_cap,
                                 'max_cap': max_cap})
    
    
    def constraint_position_limits(self, long_positions = None, short_positions = None):
        '''Specifies the number of positions that can be taken.'''
        self.constraints.update({'numb_long_limit': long_positions,
                                 'numb_short_limit': short_positions})
    
    
    def constraint_minimum_return(self, required_return = 0.1):
        '''Specifies the minimum target return for the portfolio.'''
        self.constraints.update({'required_return': required_return})
    
    
    def constraint_tracking_error(self, track_err = None):
        '''Sets tracking error constraints vs a benchmark. Set the benchmark with "add_benchmark".'''
        self.constraints.update({'track_err':track_err})
    
    
    def remove_constraint(self, remove = None):
        '''Removes the specified constraint. The current list of constraints can be found in OptimalPortfolio.constraints.
        Constraints can be overridden instead of deleted by calling the function again.'''
        del self.constraints[remove]
    
    
    ## Optimize
    def optimize(self, numb_portfolios = 200, efficent_frontier = False,
                 methods = 'arithmetic', tolerance = 1E-6, maxiter = 10000, display = False):
        '''Creates the efficent frontier based on the optimal portfolio.
        
        Args:
            numb_portfolios: the number of simulations to attempt (specifically for the efficent frontier), default 200
            
            efficent_frontier: whether to return the efficent frontier of the optimization, default False
            
            methods: method to use for calculating mean, 'arithmetic' or 'geometric' default 'arithmetic'
            
            tolerance: the tolerance for constraints and objectives, default 1E-6
        '''
        feed_dict = {'input': self.input, 'objective': self.objective_parameters,
                     'features': self.features, 'constraints': self.constraints,
                     'simulations': numb_portfolios, 'methods': methods.lower(),
                     'tol': tolerance, 'max_iter': maxiter, 'disp': display}
        
        obj = self.objective_parameters['objective'].lower()
        if obj == 'black-litterman':
            import BlackLittermanOptimizer as BLO
            self.BlackLitterman = BLO.BlackLitterman(feed_dict, efficent_frontier)
        
        if obj == 'sharpe_ratio':
            import SharpeRatioOptimizer as SRO
            self.SharpeRatio = SRO.SharpeRatio(feed_dict, efficent_frontier)
        
        if obj == 'utility':
            import UtilityOptimizer as UO
            self.Utility = UO.Utility(feed_dict, efficent_frontier)
        
        if obj == 'tail_ratio':
            import TailRatioOptimizer as TRO
            self.TailRatio = TRO.TailRatio(feed_dict, efficent_frontier)
        
        if obj == 'var':
            import VaROptimizer as VO
            self.VaR = VO.VaR(feed_dict, efficent_frontier)
        
        if obj == 'alpha':
            import AlphaOptimizer as AO
            self.Alpha = AO.Alpha(feed_dict)
        
        if obj == 'alpha_edge':
            import AlphaEdgeOptimizer as AEO
            self.AlphaEdge = AEO.AlphaEdge(feed_dict, efficent_frontier)
        
        if obj == 'sharpe_edge':
            import SharpeEdgeOptimizer as SEO
            self.SharpeEdge = SEO.SharpeEdge(feed_dict, efficent_frontier)
        
        if obj == 'beta':
            import BetaOptimizer as BO
            self.Beta = BO.Beta(feed_dict, efficent_frontier)