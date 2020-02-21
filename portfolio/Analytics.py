# -*- coding: utf-8 -*-
"""
This is a class file for calculating certain
trading indicators

@Author: James Altemus
"""

import pandas as pd


def bollinger(self, period = 14, ran = 2, analyze = False):
    '''Calculates bolinger bands based off a pandas prices frame or series
    
    Args:
        period: default 14, the amount of prices to use in the calcuation
        ran: default 2, the number of standard deviaitons to be used for 
             calculating the bollinger bands
        calc: default 'Close', the name of the column used for calculations
        analyze: default False, whether to perform differential analysis
        unique: default False, assigns the name calc + 'name' if set to True
                only useful if amend is True
    
    Returns:
        avg: The average bollinger band line
        up: upper bollinger band line
        down: lower bollinger band line
        
        if analyze:
            above_top: positive if trade prices is above the top bollinger band
            below_bot: positive if trade prices is below the bottom bollinger band
    '''
    stdev = self.prices.rolling(window = period).std()
    avg = self.prices.rolling(window = period).mean()
    up = avg + stdev * ran
    down = avg - stdev * ran
    
    avg.columns = self.tickers + '_Boll_EMA'
    up.columns = self.tickers + '_Boll_High'
    down.columns = self.tickers + '_Boll_Low'
    self.BollingerBands = pd.concat([avg,up,down],axis=1)
    if analyze:
        above_top = self.prices - up
        below_bot = down - self.prices
        
        above_top.columns = self.tickers + '_Above_Boll_Top'
        below_bot.columns = self.tickers + '_Below_Boll_Bot'
        self.BollingerBands = pd.concat([self.BollingerBands,above_top,below_bot],axis=1)


def macd(self, slow_span = 26, fast_span = 12, signal_span = 9, analyze = False):
    '''Calculates MACD from prices from pandas prices freme or series
    
    Args:
        slow_run: the higher period used to calculate MACD line, default 26
        fast_span: the lower period used to calculate MACD line, default 12
        signal_span: the period used to calculate the signal like, default 9
        analyze: default False, whether to perform differential analysis
    
    Returns:
        signal: the pandas series of the signal line
        MACD: the pandas series of the MACD like
        
        if analyze:
            MACD_signal: the difference between the MACD and signal line
    '''
    slow = self.prices.ewm(span = slow_span, min_periods = slow_span).mean()
    fast = self.prices.ewm(span = fast_span, min_periods = slow_span).mean()
    MACD = fast - slow
    
    signal = MACD.ewm(span = signal_span, min_periods = signal_span)
    
    MACD.columns = self.tickers + '_MACD'
    signal.columns = self.tickers + '_Signal'
    self.MACD = pd.concat([MACD,signal],axis=1)
    if analyze:
        MACD_signal = MACD - signal
        
        MACD_signal.columns = self.tickers + '_MACD_Signal'
        self.MACD = pd.concat([self.MACD,MACD_signal],axis=1)


def stochastic(self, k_window = 14, d_window = 3, analyze = False):
    '''Calculates the stochstic oscillators from a pandas prices frame or series
    and returns them and their differnces
    
    Args:
        prices: the pandas prices frame or pandas series
        k_window: the number of periods used to calculate %K, default 14
        d_window: the number of periods used to calculate %D, default 3
        analyze: default False, whether to perform differential analysis
    
    Returns:
        K: the %K line
        D: the %D line
        DS: the %DS line
        DSS: the %DSS line
    '''
    l = self.prices.rolling(window = k_window).min()
    h = self.prices.rolling(window = k_window).max()
    k = ((self.prices - l)/(h - l)) * 100
    
    d = k.rolling(window = d_window).mean()
    ds = d.rolling(window = d_window).mean()
    dss = ds.rolling(window = d_window).mean()
    
    k.columns = self.tickers + '_K'
    d.columns = self.tickers + '_D'
    ds.columns = self.tickers + '_DS'
    dss.columns = self.tickers + '_DSS'
    self.StochasticOscillator = pd.concat([k,d,ds,dss],axis=1)
    if analyze:
        kd = k - d
        dds = d - ds
        dsdss = ds - dss
        
        kd.columns = self.tickers + '_K_D'
        dds.columns = self.tickers + '_D_DS'
        dsdss.columns = self.tickers + '_DS_DSS'
        self.MACD = pd.concat([self.Stochastic,kd,dds,dsdss],axis=1)


def rsi(self, period = 14, calc = 'Returns'):
    '''Calculates RSI from a pandas prices frame or series
    
    Args:
        prices: the pandas prices frame or series
        period: the number of prices points used to calculate RSI, default 14
        dtype: the type of prices, either 'trade' or 'return'.
            'trade' specifies price prices, 'return' specifies return prices
        calc: default 'Returns', accepts either 'Returns' or 'Log_Returns'
    
    Returns:
        rsi: the relative strength index
    '''
    if calc.lower() == 'returns':
        trade = self.returns
    if calc.lower() == 'log_returns':
        trade = self.log_returns
    
    mx = trade[trade > 0].fillna(0)[:-1]
    mn = trade[trade < 0].fillna(0)[:-1]
    
    rsi_mn = -mn.rolling(window = period-1).mean()
    rsi_mx = mx.rolling(window = period-1).mean()
    
    rsi_mx = rsi_mx.shift(1)
    rsi_mn = rsi_mn.shift(1)
    
    rs = (rsi_mx + mx) / (rsi_mn + mn)
    
    rsi = 100 - (100 / (1 + rs))
    
    rsi.columns = self.tickers + 'RSI'
    self.RSI = rsi


def ichimoku(self, conv_per = 9, base_per = 26, lead_per = 56, analyze = False):
    '''Calculates prices necessary to construct an ichimoku cloud
    
    Args:
        prices: the pandas prices frame or series
        conv_per: the period for calculating the conversion line, defualt 9
        base_per: the period for calculating the base line, default 26
        lead_per: the period used for calculating the leading line, default 56
        analyze: default False, whether to perform differential analysis on
                the base prices vs the computed prices
    
    Returns:
        tenken_sen: the conversion line calculation
        kijun_sen: the base line
        senkou_a: the leading span a
        senkou_b: the leading span b
        chikou: the lagging span
        
        All are formatted as pandas prices frames or series
    '''
    mx = self.prices.rolling(window = conv_per).max()
    mn = self.prices.rolling(window = conv_per).min()
    
    tenken_sen = (mx + mn)/2
    
    mx = self.prices.rolling(window = base_per).max()
    mn = self.prices.rolling(window = base_per).min()
    
    kijun_sen = (mx + mn)/2
    
    senkou_a = tenken_sen + kijun_sen
    
    mx = self.prices.rolling(window = lead_per).max()
    mn = self.prices.rolling(window = lead_per).min()
    
    senkou_b = (mx + mn)/2
    
    senkou_a = senkou_a.shift(base_per)
    senkou_b = senkou_b.shift(base_per)
    
    chikou_span = self.prices.shift(-base_per)
    
    senkou_a = pd.DataFrame(senkou_a)
    senkou_b = pd.DataFrame(senkou_b)
    chikou_span = pd.DataFrame(chikou_span)
    
    tenken_sen.columns = self.tickers + '_Tenken_Sen'
    kijun_sen.columns = self.tickers + '_Kijun_Sen'
    senkou_a.columns = self.tickers + '_Senkou_A'
    senkou_b.columns = self.tickers + '_Senkou_B'
    chikou_span.columns = self.tickers + '_Chikou_Span'
    self.Ichimoku = pd.concat([tenken_sen,kijun_sen,senkou_a,senkou_b,chikou_span],axis=1)
    if analyze:
        cloud = senkou_a - senkou_b
        
        cloud.columns = self.tickers + '_Cloud'
        self.MACD = pd.concat([self.Ichimoku,cloud],axis=1)


def chande_mo(self, period = 14, calc = 0):
    '''Calculates the Chande Momentum Oscillator from a pandas prices frame or series
    
    Args:
        period: the number of prices points used to calcualte the oscillator,
            default is 14
        calc: default 'Close', the name of the column to be used for calculations
        
        amend: default True, whether to add the computed columns to the pricesframe
        unique: default False, assigns the name calc + 'name' if set to True
                only useful if amend is True
    
    Returns:
        chande: the chande oscillator values
    '''
    high = self.prices[self.prices > 0].fillna(0)
    low = self.prices[self.prices < 0].fillna(0)
    
    high = high.rolling(window = period).sum()
    low = low.rolling(window = period).sum()
    
    chande = ((high - low) / (high + low)) * 100
    
    chande.columns = self.tickers + '_Chande'
    
    self.ChandeMomentumOscillator = chande


def wilder_smoothing(self, period = 14, calc = 'Close', unique = False):
    '''Calculates the Wilder's Smoothing of given prices
    
    Args:
        period: the number of prices points for calculation, default is 14
        calc: default 'Close', the name of the column to be used for calculations
        unique: default False, assigns the name calc + 'name' if set to True
                only useful if amend is True
    
    Returns:
        chande: the wilder's smoothing values
    '''
    prev_trade = self.prices.shift(period)
    
    trade2 = self.prices.rolling(window = period).sum()
    prev_trade = prev_trade.rolling(window = period).sum()
    
    wild_smoo = trade2 - prev_trade/period + self.prices
    
    wild_smoo.columns = self.tickers + '_Wilder_Smooth'
    self.WilderSmoothing = wild_smoo