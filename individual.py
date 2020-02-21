# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 18:44:35 2019

@author: James Altemus
"""

import numpy as np
import pandas as pd


def bollinger(data, period = 14, ran = 2, analyze = False):
    '''Calculates bolinger bands based off a pandas data frame or series
    
    Args:
        data: the base data as a pandas series
        period: default 14, the amount of data to use in the calcuation
        ran: default 2, the number of standard deviaitons to be used for 
             calculating the bollinger bands
        calc: default 'Close', the name of the column used for calculations
        analyze: default False, whether to perform differential analysis
    
    Returns:
        avg: The average bollinger band line
        up: upper bollinger band line
        down: lower bollinger band line
        
        if analyze:
            above_top: positive if trade data is above the top bollinger band
            below_bot: positive if trade data is below the bottom bollinger band
    '''
    trade = data
    stdev = trade.rolling(window = period).std()
    
    avg = trade.rolling(window = period).mean()
    up = avg + stdev * ran
    down = avg - stdev * ran
    if analyze:
        return avg, up, down
    else:
        above_top = trade - up
        below_bot = down - trade
        return avg, up, down, above_top, below_bot


def macd(data, slow_span = 26, fast_span = 12, signal_span = 9, analyze = False):
    '''Calculates MACD from data from pandas data freme or series
    
    Args:
        data: the base data as a pandas series
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
    trade = data
    slow = trade.ewm(span = slow_span, min_periods = slow_span).mean()
    fast = trade.close.ewm(span = fast_span, min_periods = slow_span).mean()
    MACD = fast - slow
    
    signal = MACD.ewm(span = signal_span, min_periods = signal_span)
    
    if analyze:
        MACD_signal = MACD - signal
        return signal, MACD, MACD_signal
    else:
        return signal, MACD


def stochastic(data, high = None, low = None, k_window = 14, d_window = 3,
               absolute = False, analyze = False):
    '''Calculates the stochstic oscillators from a pandas data frame or series
    and returns them and their differnces
    
    Args:
        data: the base data as a pandas series
        hgih: the high data as a pandas series
        low: the low data as a pandas series
        k_window: the number of periods used to calculate %K, default 14
        d_window: the number of periods used to calculate %D, default 3
        absolute
        
        amend: default True, whether to add the computed columns to the dataframe
        analyze: default False, whether to perform differential analysis
        unique: default False, assigns the name calc + 'name' if set to True
                only useful if amend is True
    
    Returns:
        K: the %K line
        D: the %D line
        DS: the %DS line
        DSS: the %DSS line
    '''
    trade = data
    if absolute:
        l = low
        h = high
        l = l.rolling(window = k_window).min()
        h = h.rolling(window = k_window).max()
    else:
        l = trade.rolling(window = k_window).min()
        h = trade.rolling(window = k_window).max()
    k = ((trade - l)/(h - l)) * 100
    
    d = k.rolling(window = d_window).mean()
    ds = d.rolling(window = d_window).mean()
    dss = ds.rolling(window = d_window).mean()
    
    if analyze:
        kd = k - d
        dds = d - ds
        dsdss = ds - dss
        return k, d, ds, dss, kd, dds, dsdss
    else:
        return k, d, ds, dss


def rsi(data, period = 14):
    '''Calculates RSI from a pandas data frame or series
    
    Args:
        data: the base data as a pandas series
        period: the number of data points used to calculate RSI, default 14
    
    Returns:
        rsi: the relative strength index
    
    Raises:
        IndexError: if the indexed column is not found
    '''
    try:
        trade = data
    except:
        raise IndexError('The return column not found. Use the .returns function before calculating RSI')
    
    mx = trade[trade > 0].fillna(0)[:-1]
    mn = trade[trade < 0].fillna(0)[:-1]
    
    rsi_mn = -mn.rolling(window = period-1).mean()
    rsi_mx = mx.rolling(window = period-1).mean()
    
    rsi_mx = rsi_mx.shift(1)
    rsi_mn = rsi_mn.shift(1)
    
    rs = (rsi_mx + mx) / (rsi_mn + mn)
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def ichimoku(data, conv_per = 9, base_per = 26, lead_per = 56, analyze = False):
    '''Calculates data necessary to construct an ichimoku cloud
    
    Args:
        data: the base data as a pandas series
        conv_per: the period for calculating the conversion line, defualt 9
        base_per: the period for calculating the base line, default 26
        lead_per: the period used for calculating the leading line, default 56
        analyze: default False, whether to perform differential analysis on
                the base data vs the computed data
    
    Returns:
        tenken_sen: the conversion line calculation
        kijun_sen: the base line
        senkou_a: the leading span a
        senkou_b: the leading span b
        chikou: the lagging span
        
        All are formatted as pandas data frames or series
    '''
    trade = data
    mx = trade.rolling(window = conv_per).max()
    mn = trade.rolling(window = conv_per).min()
    
    tenken_sen = (mx + mn)/2
    
    mx = trade.rolling(window = base_per).max()
    mn = trade.rolling(window = base_per).min()
    
    kijun_sen = (mx + mn)/2
    
    senkou_a = tenken_sen + kijun_sen
    
    mx = trade.rolling(window = lead_per).max()
    mn = trade.rolling(window = lead_per).min()
    
    senkou_b = (mx + mn)/2
    
    senkou_a = senkou_a.shift(base_per)
    senkou_b = senkou_b.shift(base_per)
    
    chikou_span = trade.shift(-base_per)
    
    senkou_a = pd.DataFrame(senkou_a)
    senkou_b = pd.DataFrame(senkou_b)
    chikou_span = pd.DataFrame(chikou_span)
    
    if analyze:
        cloud = senkou_a - senkou_b
        return tenken_sen, kijun_sen, senkou_a, senkou_b, chikou_span, cloud
    else:
        return tenken_sen, kijun_sen, senkou_a, senkou_b, chikou_span


def chande_mo(data, period = 14):
    '''Calculates the Chande Momentum Oscillator from a pandas data frame or series
    
    Args:
        data: the base data as a pandas series
        period: the number of data points used to calcualte the oscillator,
            default is 14
    
    Returns:
        chande: the chande oscillator values
    '''
    trade = data
    high = trade[trade > 0].fillna(0)
    low = trade[trade < 0].fillna(0)
    
    high = high.rolling(window = period).sum()
    low = low.rolling(window = period).sum()
    
    chande = ((high - low) / (high + low)) * 100
    
    return chande


def accel_bands(close_data, high_data, low_data, period = 20,
                analyze = False):
    '''Calculates the acceleration bands for a given period from a
    pandas data frame or series
    
    Args:
        close_data: the pandas series with closing prices
        high_data: the pandas series with high prices
        low_data: the pandas series with low prices
        period: the number of data points used to calcualte the bands, 
            default is 20
        analyze: default False, whether to perform differential analysis on
                the base data vs the computed data
    
    Returns:
        mid: The middle acceleration band line
        up: upper acceleration band line
        down: lower acceleration band line
        above_top: positive if trade data is above the top acceleration band
        below_bot: positive if trade data is below the bottom acceleration band
    '''
    mid = close_data.rolling(window = period).mean()
    
    up = high_data * (1 + 4 * (high_data + low_data) / (high_data + low_data))
    down = low_data * (1 - 4 * (high_data - low_data) / (high_data + low_data))
    
    up = up.rolling(window = period).mean()
    down = down.rolling(window = period).mean()
    
    if analyze:
        above_top = close_data - up
        below_bot = down - close_data
        return mid, up, down, above_top, below_bot
    else:
        return mid, up, down


def cci(close_data, high_data, low_data, period = 20):
    '''Calculates the Commodity Channel Index from a pandas data frame or series
    
    Args:
        close_data: the pandas data frame or series with closing prices
        high_data: the pandas data frame or series with high prices
        low_data: the pandas data frame or series with low prices
        period: the number of data points to use for calculation, default is 20
    
    Returns:
        cci: the commodity channel index of the data set
    '''
    typical = (high_data + low_data + close_data) / 3
    
    average = typical.rolling(window = period).mean()
    
    deviation = (typical - average) / period
    
    cci = (typical - average) / (0.015 * deviation)
    
    return cci


def wilder_smoothing(data, period = 14):
    '''Calculates the Wilder's Smoothing of given data
    
    Args:
        data: the base data as a pandas series
        period: the number of data points for calculation, default is 14
        calc: default 'Close', the name of the column to be used for calculations
        
        amend: default True, whether to add the computed columns to the dataframe
        unique: default False, assigns the name calc + 'name' if set to True
                only useful if amend is True
    
    Returns:
        chande: the wilder's smoothing values
    '''
    trade = data
    prev_trade = trade.shift(period)
    
    trade2 = trade.rolling(window = period).sum()
    prev_trade = prev_trade.rolling(window = period).sum()
    
    wild_smoo = trade2 - prev_trade/period + trade
    return wild_smoo

    
def direct_movement(close_data, high_data, low_data, period = 14, analyze = False):
    '''Calculates the directional movement oscillators given high, low and close
    
    Args:
        close_data: the pandas series with closing prices
        high_data: the pandas series with high prices
        low_data: the pandas series with low prices
        period: the number of data points for calculation, default is 14
        analyze: default False, whether to perform differential analysis on
                the base data vs the computed data
    
    Returns:
        di_plus: the +DI line
        di_minus: the -DI line
        dx: the difference between the +DI and -DI lines
        adx: the moving average of the DX line
    
    Raises:
        Variable Length if the data sets are not the same length
    '''
    data = pd.DataFrame([close_data, high_data, low_data]).T
    
    prev_close = close_data.shift(1) 
    prev_high = high_data.shift(1)
    prev_low = low_data.shift(1)
    
    plus = high_data - prev_high
    minus = prev_low - low_data
    dm_plus = np.where(plus > minus, np.maximum(plus, 0), 0)
    dm_minus = np.where(minus > plus, np.maximum(minus, 0), 0)
    
    tr = pd.DataFrame()
    tr['1'] = high_data - low_data
    tr['2'] = list(map(abs, high_data - prev_close))
    tr['3'] = list(map(abs, low_data - prev_close))
    tr = tr.max(axis = 1)
    
    data['tr'] = tr
    data['dmp'] = dm_plus
    data['dmm'] = dm_minus
    atr = wilder_smoothing(period=period, calc = 'tr', amend=False)
    dm_plus = wilder_smoothing(period=period, calc = 'dmp', amend=False)
    dm_minus = wilder_smoothing(period=period, calc = 'dmm', amend=False)
    data.drop(['tr','dmp','dmm'], axis = 1)
    
    di_plus = dm_plus / atr * 100
    di_minus = dm_minus / atr * 100
    
    di_plus_abs = pd.DataFrame(map(abs, di_plus - di_minus))
    di_minus_abs = pd.DataFrame(map(abs, di_plus + di_minus))
    
    dx = di_plus_abs / di_minus_abs * 100
    
    data['dx'] = dx
    adx = wilder_smoothing(period=period, calc = 'dx', amend= False)
    data.drop('dx', axis = 1)
    
    if analyze:
        d_comp = np.where(di_plus > di_minus, 1, -1)
        return di_plus, di_minus, dx, adx, d_comp
    else:
        return di_plus, di_minus, dx, adx


def ultimate(close_data, high_data, low_data, period_1 = 7, period_2 = 14,
             period_3 = 28, weight_1 = 4, weight_2 = 2, weight_3 = 1):
    '''Computes the Ultimate Oscillator from a pandas data frame or series
    given short, medium, and long periods
    
    Args:
        close_data: the pandas series with closing prices
        high_data: the pandas series with high prices
        low_data: the pandas series with low prices
        period_1: the number of data points for calcualtion, default is 7
        period_2: the number of data points for calcualtion, default is 14
        period_3: the number of data points for calcualtion, default is 28
        weight_1: the weight for period 1
        weight_2: the weight for period 2
        weight_3: the weight for period 3
    
    Returns:
        ultimate: the ultimate oscillator values as a pandas data frame
    '''
    
    prev = list(np.repeat(None, 1)) + list(close_data)
    prev = prev[:len(prev)-1]
    prev = pd.DataFrame(prev)
    
    buying = pd.DataFrame()
    buying['prev'] = prev
    buying['low'] = low_data
    buying = buying.min(axis = 1)
    
    trange = pd.DataFrame()
    trange['prev'] = prev
    trange['high'] = high_data
    trange = trange.max(axis = 1)
    
    buying = close_data - buying
    trange = trange - buying
    
    avg_1 = buying.rolling(window = period_1).sum() / trange.rolling(window = period_1).sum()
    avg_2 = buying.rolling(window = period_2).sum() / trange.rolling(window = period_2).sum()
    avg_3 = buying.rolling(window = period_3).sum() / trange.rolling(window = period_3).sum()
    
    ultimate = 100 * ((weight_1 * avg_1) + (weight_2 * avg_2) + (weight_3 * avg_3)) / (weight_1 + weight_2 + weight_3)
    
    return ultimate