# -*- coding: utf-8 -*-
"""
This is a class file for calculating certain
trading indicators

@Author: James Altemus
"""

import numpy as np
import pandas as pd

class TechnicalAnalysis:
    def __init__(self, price_data, col_names = None, order = 'increasing'):
        '''The file must have a date column followed by at least one column
        of trading data. If there is only one column, it will be named Close.
        The default for multiple columns is High, Low, Open, Close.
        
        Args:
            data: the pandas data frame to use for computation
            order: default increasing, indicated the order the data is on the CSV
                  decreasing means new data first, increasing means old data first
        '''
        self.data = price_data
        if col_names:
            self.data.columns = col_names
        
        if order == 'decreasing':
            self.data = self.data.iloc[::-1]
        elif order == 'increasing':
            pass
        else:
            raise ValueError('The order value is invalid. Only use increasing or decreasing.')
    
    
    def returns(self, method = 'Arithmetic', calc = 'Close', unique = False):
        ''' Calculates the returns for a series of data
        
        Args:
            method: default Arithmetic, either Arithmetic or Logarithmic
                    (also accepts first letter)
            calc: the name of the column to be used for calculations
            unique: default False, assigns the name calc + 'name' if set to True
                    only useful if amend is True
        
        Returns:
            The series with the returns
        '''
        trade = self.data[calc]
        if method.lower() in ['a', 'arithmetic']:
            returns = trade.pct_change()
            if unique:
                self.data[calc+'_Return'] = returns
            else:
                self.data['Return'] = returns
        elif method.lower() in ['l', 'logarithmic']:
            returns = np.log(trade) - np.log(trade.shift(1))
            if unique:
                self.data[calc+'_Log_Return'] = returns
            else:
                self.data['Log_Return'] = returns
    
    
    def bollinger(self, period = 14, ran = 2, calc = 'Close', analyze = False,
                  unique = False):
        '''Calculates bolinger bands based off a pandas data frame or series
        
        Args:
            period: default 14, the amount of data to use in the calcuation
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
                above_top: positive if trade data is above the top bollinger band
                below_bot: positive if trade data is below the bottom bollinger band
        '''
        trade = self.data[calc]
        stdev = trade.rolling(window = period).std()
        
        avg = trade.rolling(window = period).mean()
        up = avg + stdev * ran
        down = avg - stdev * ran
        
        if unique:
            self.data[calc+'_Boll_EMA'] = avg
            self.data[calc+'_Boll_High'] = up
            self.data[calc+'_Boll_Low'] = down
            if analyze:
                above_top = trade - up
                below_bot = down - trade
                self.data[calc+'_Above_Boll_Top'] = above_top
                self.data[calc+'_Below_Boll_Bot'] = below_bot
        else:
            self.data['Boll_EMA'] = avg
            self.data['Boll_High'] = up
            self.data['Boll_Low'] = down
            if analyze:
                above_top = trade - up
                below_bot = down - trade
                self.data['Above_Boll_Top'] = above_top
                self.data['Below_Boll_Bot'] = below_bot
    
    
    def macd(self, slow_span = 26, fast_span = 12, signal_span = 9, calc = 'Close',
             amend = True, analyze = False, unique = False):
        '''Calculates MACD from data from pandas data freme or series
        
        Args:
            slow_run: the higher period used to calculate MACD line, default 26
            fast_span: the lower period used to calculate MACD line, default 12
            signal_span: the period used to calculate the signal like, default 9
            calc: default 'Close', the name of the column used for calculations
            
            amend: default True, whether to add the computed columns to the dataframe
            analyze: default False, whether to perform differential analysis
            unique: default False, assigns the name calc + 'name' if set to True
                    only useful if amend is True
        
        Returns:
            signal: the pandas series of the signal line
            MACD: the pandas series of the MACD like
            
            if analyze:
                MACD_signal: the difference between the MACD and signal line
        '''
        trade = self.data[calc]
        slow = trade.ewm(span = slow_span, min_periods = slow_span).mean()
        fast = trade.close.ewm(span = fast_span, min_periods = slow_span).mean()
        MACD = fast - slow
        
        signal = MACD.ewm(span = signal_span, min_periods = signal_span)
        
        if unique:
            self.data[calc+'_Signal'] = signal
            self.data[calc+'_MACD'] = MACD
            if analyze:
                MACD_signal = MACD - signal
                self.data[calc+'_MACD_Signal'] = MACD_signal
        else:
            self.data['Signal'] = signal
            self.data['MACD'] = MACD
            if analyze:
                MACD_signal = MACD - signal
                self.data['MACD_Signal'] = MACD_signal
    
    
    def stochastic(self, k_window = 14, d_window = 3, absolute = False, 
                   calc = ['Close', 'High', 'Low'], analyze = False, unique = False):
        '''Calculates the stochstic oscillators from a pandas data frame or series
        and returns them and their differnces
        
        Args:
            data: the pandas data frame or pandas series
            k_window: the number of periods used to calculate %K, default 14
            d_window: the number of periods used to calculate %D, default 3
            absolute
            calc: default ['Close', 'High', 'Low'], the name of the columns used
                  for calculations. High and Low are only used if absolute is true
                  the names of the columns must follow the same order if default
                  is overridden
            analyze: default False, whether to perform differential analysis
            unique: default False, assigns the name calc + 'name' if set to True
                    only useful if amend is True
        
        Returns:
            K: the %K line
            D: the %D line
            DS: the %DS line
            DSS: the %DSS line
        '''
        trade = self.data[calc[0]]
        if absolute:
            l = self.data[calc[2]]
            h = self.data[calc[1]]
            l = l.rolling(window = k_window).min()
            h = h.rolling(window = k_window).max()
        else:
            l = trade.rolling(window = k_window).min()
            h = trade.rolling(window = k_window).max()
        k = ((trade - l)/(h - l)) * 100
        
        d = k.rolling(window = d_window).mean()
        ds = d.rolling(window = d_window).mean()
        dss = ds.rolling(window = d_window).mean()
        
        if unique:
            self.data[calc[0]+'_K'] = k
            self.data[calc[0]+'_D'] = d
            self.data[calc[0]+'_DS'] = ds
            self.data[calc[0]+'_DSS'] = dss
            if analyze:
                kd = k - d
                dds = d - ds
                dsdss = ds - dss
                self.data[calc[0]+'_K_D'] = kd
                self.data[calc[0]+'_D_DS'] = dds
                self.data[calc[0]+'_DS_DSS'] = dsdss
        else:
            self.data['K'] = k
            self.data['D'] = d
            self.data['DS'] = ds
            self.data['DSS'] = dss
            if analyze:
                kd = k - d
                dds = d - ds
                dsdss = ds - dss
                self.data['K_D'] = kd
                self.data['D_DS'] = dds
                self.data['DS_DSS'] = dsdss
    
    
    def rsi(self, period = 14, calc = 'Return', unique = False):
        '''Calculates RSI from a pandas data frame or series
        
        Args:
            data: the pandas data frame or series
            period: the number of data points used to calculate RSI, default 14
            dtype: the type of data, either 'trade' or 'return'.
                'trade' specifies price data, 'return' specifies return data
            calc: default 'Close', the name of the column to be used for calculations
            unique: default False, assigns the name calc + 'name' if set to True
                    only useful if amend is True
        
        Returns:
            rsi: the relative strength index
        
        Raises:
            IndexError: if the indexed column is not found
        '''
        try:
            trade = self.data[calc]
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
        
        if unique:
            self.data[calc+'_RSI'] = rsi
        else:
            self.data['RSI'] = rsi
    
    
    def ichimoku(self, conv_per = 9, base_per = 26, lead_per = 56, calc = 'Close',
                 analyze = False, unique = False):
        '''Calculates data necessary to construct an ichimoku cloud
        
        Args:
            data: the pandas data frame or series
            conv_per: the period for calculating the conversion line, defualt 9
            base_per: the period for calculating the base line, default 26
            lead_per: the period used for calculating the leading line, default 56
            calc: default 'Close', the name of the column to be used for calculations
            analyze: default False, whether to perform differential analysis on
                    the base data vs the computed data
            unique: default False, assigns the name calc + 'name' if set to True
                    only useful if amend is True
        
        Returns:
            tenken_sen: the conversion line calculation
            kijun_sen: the base line
            senkou_a: the leading span a
            senkou_b: the leading span b
            chikou: the lagging span
            
            All are formatted as pandas data frames or series
        '''
        trade = self.data[self.data.columns[calc]]
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
        
        if unique:
            self.data[calc+'_Tenken_Sen'] = tenken_sen
            self.data[calc+'_Kijun_Sen'] = kijun_sen
            self.data[calc+'_Senkou_A'] = senkou_a
            self.data[calc+'_Senkou_B'] = senkou_b
            self.data[calc+'_Chikou_Span'] = chikou_span
            if analyze:
                cloud = senkou_a - senkou_b
                self.data[calc+'_Cloud'] = cloud
        else:
            self.data['Tenken_Sen'] = tenken_sen
            self.data['Kijun_Sen'] = kijun_sen
            self.data['Senkou_A'] = senkou_a
            self.data['Senkou_B'] = senkou_b
            self.data['Chikou_Span'] = chikou_span
            if analyze:
                cloud = senkou_a - senkou_b
                self.data['Cloud'] = cloud
    
    
    def chande_mo(self, period = 14, calc = 0, unique = False):
        '''Calculates the Chande Momentum Oscillator from a pandas data frame or series
        
        Args:
            period: the number of data points used to calcualte the oscillator,
                default is 14
            calc: default 'Close', the name of the column to be used for calculations
            
            amend: default True, whether to add the computed columns to the dataframe
            unique: default False, assigns the name calc + 'name' if set to True
                    only useful if amend is True
        
        Returns:
            chande: the chande oscillator values
        '''
        trade = self.data[self.data.columns[calc]]
        high = trade[trade > 0].fillna(0)
        low = trade[trade < 0].fillna(0)
        
        high = high.rolling(window = period).sum()
        low = low.rolling(window = period).sum()
        
        chande = ((high - low) / (high + low)) * 100
        
        if unique:
            self.data[calc+'_Chande_MO'] = chande
        else:
            self.data['Chande_MO'] = chande
    
    
    def accel_bands(self, period = 20, calc = ['Close', 'High', 'Low'],
                    analyze = False, unique = False):
        '''Calculates the acceleration bands for a given period from a
        pandas data frame or series
        
        Args:
            period: the number of data points used to calcualte the bands, 
                default is 20
            calc: default ['Close', 'High', 'Low'], the name of the columns used
                  for calculations. High and Low are only used if absolute is true
                  the names of the columns must follow the same order if default
                  is overridden
            analyze: default False, whether to perform differential analysis on
                    the base data vs the computed data
            unique: default False, assigns the name calc + 'name' if set to True
                    only useful if amend is True
        
        Returns:
            mid: The middle acceleration band line
            up: upper acceleration band line
            down: lower acceleration band line
            above_top: positive if trade data is above the top acceleration band
            below_bot: positive if trade data is below the bottom acceleration band
        '''
        close_data = self.data[calc[0]]
        high_data = self.data[calc[1]]
        low_data = self.data[calc[2]]
        mid = close_data.rolling(window = period).mean()
        
        up = high_data * (1 + 4 * (high_data + low_data) / (high_data + low_data))
        down = low_data * (1 - 4 * (high_data - low_data) / (high_data + low_data))
        
        up = up.rolling(window = period).mean()
        down = down.rolling(window = period).mean()
        
        if unique:
            self.data[calc+'_Acc_Mid'] = mid
            self.data[calc+'_Acc_Up'] = up
            self.data[calc+'_Acc_Low'] = down
            if analyze:
                above_top = close_data - up
                below_bot = down - close_data
                self.data[calc+'_Accel_vs_Top'] = above_top
                self.data[calc+'_Accel_vs_Bot'] = below_bot
        else:
            self.data['Acc_Mid'] = mid
            self.data['Acc_Up'] = up
            self.data['Acc_Low'] = down
            if analyze:
                above_top = close_data - up
                below_bot = down - close_data
                self.data['Accel_vs_Top'] = above_top
                self.data['Accel_vs_Bot'] = below_bot
    
    
    def cci(self, period = 20, calc = ['Close', 'High', 'Low'], amend = True,
            unique = False):
        '''Calculates the Commodity Channel Index from a pandas data frame or series
        
        Args:
            period: the number of data points to use for calculation, default is 20
            calc: default ['Close', 'High', 'Low'], the name of the columns used
                  for calculations. High and Low are only used if absolute is true
                  the names of the columns must follow the same order if default
                  is overridden
            unique: default False, assigns the name calc + 'name' if set to True
                    only useful if amend is True
        
        Returns:
            cci: the commodity channel index of the data set
        '''
        close_data = self.data[calc[0]]
        high_data = self.data[calc[1]]
        low_data = self.data[calc[2]]
        
        typical = (high_data + low_data + close_data) / 3
        
        average = typical.rolling(window = period).mean()
        
        deviation = (typical - average) / period
        
        cci = (typical - average) / (0.015 * deviation)
        
        if unique:
            self.data[calc+'_CCI'] = cci
        else:
            self.data['CCI'] = cci
    
    
    def wilder_smoothing(self, period = 14, calc = 'Close', unique = False):
        '''Calculates the Wilder's Smoothing of given data
        
        Args:
            period: the number of data points for calculation, default is 14
            calc: default 'Close', the name of the column to be used for calculations
            unique: default False, assigns the name calc + 'name' if set to True
                    only useful if amend is True
        
        Returns:
            chande: the wilder's smoothing values
        '''
        trade = self.data[calc]
        prev_trade = trade.shift(period)
        
        trade2 = trade.rolling(window = period).sum()
        prev_trade = prev_trade.rolling(window = period).sum()
        
        wild_smoo = trade2 - prev_trade/period + trade
        if unique:
            self.data[calc+'_Wilder_Smooth'] = wild_smoo
        else:
            self.data['Wilder_Smooth'] = wild_smoo
    
        
    def direct_movement(self, period = 14, calc = ['Close', 'High', 'Low'],
                        analyze = False, unique = False):
        '''Calculates the directional movement oscillators given high, low and close
        
        Args:
            period: the number of data points for calculation, default is 14
            calc: default ['Close', 'High', 'Low'], the name of the columns used
                  for calculations. High and Low are only used if absolute is true
                  the names of the columns must follow the same order if default
                  is overridden
            analyze: default False, whether to perform differential analysis on
                    the base data vs the computed data
            unique: default False, assigns the name calc + 'name' if set to True
                    only useful if amend is True
        
        Returns:
            di_plus: the +DI line
            di_minus: the -DI line
            dx: the difference between the +DI and -DI lines
            adx: the moving average of the DX line
        
        Raises:
            Variable Length if the data sets are not the same length
        '''
        close_data = self.data[calc[0]]
        high_data = self.data[calc[1]]
        low_data = self.data[calc[2]]
        
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
        
        self.data['tr'] = tr
        self.data['dmp'] = dm_plus
        self.data['dmm'] = dm_minus
        atr = self.wilder_smoothing(period=period, calc = 'tr', amend=False)
        dm_plus = self.wilder_smoothing(period=period, calc = 'dmp', amend=False)
        dm_minus = self.wilder_smoothing(period=period, calc = 'dmm', amend=False)
        self.data.drop(['tr','dmp','dmm'], axis = 1)
        
        di_plus = dm_plus / atr * 100
        di_minus = dm_minus / atr * 100
        
        di_plus_abs = pd.DataFrame(map(abs, di_plus - di_minus))
        di_minus_abs = pd.DataFrame(map(abs, di_plus + di_minus))
        
        dx = di_plus_abs / di_minus_abs * 100
        
        self.data['dx'] = dx
        adx = self.wilder_smoothing(period=period, calc = 'dx', amend= False)
        self.data.drop('dx', axis = 1)
        
        if unique:
            self.data[calc+'_DIp'] = di_plus
            self.data[calc+'_DIm'] = di_minus
            self.data[calc+'_DX'] = dx
            self.data[calc+'_ADX'] = adx
            if analyze:
                d_comp = np.where(di_plus > di_minus, 1, -1)
                self.data[calc+'_D_Use'] = d_comp
        else:
            self.data['DIp'] = di_plus
            self.data['DIm'] = di_minus
            self.data['DX'] = dx
            self.data['ADX'] = adx
            if analyze:
                d_comp = np.where(di_plus > di_minus, 1, -1)
                self.data['D_Use'] = d_comp
    
    
    def ultimate(self, period_1 = 7, period_2 = 14, period_3 = 28, weight_1 = 4,
                 weight_2 = 2, weight_3 = 1, calc = ['Close', 'High', 'Low'],
                 unique = False):
        '''Computes the Ultimate Oscillator from a pandas data frame or series
        given short, medium, and long periods
        
        Args:
            period_1: the number of data points for calcualtion, default is 7
            period_2: the number of data points for calcualtion, default is 14
            period_3: the number of data points for calcualtion, default is 28
            weight_1: the weight for period 1
            weight_2: the weight for period 2
            weight_3: the weight for period 3
            calc: default ['Close', 'High', 'Low'], the name of the columns used
                  for calculations. High and Low are only used if absolute is true
                  the names of the columns must follow the same order if default
                  is overriddene
            unique: default False, assigns the name calc + 'name' if set to True
                    only useful if amend is True
        
        Returns:
            ultimate: the ultimate oscillator values as a pandas data frame
        '''
        
        close_data = self.data[calc[0]]
        high_data = self.data[calc[1]]
        low_data = self.data[calc[2]]
        
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
        
        if unique:
            self.data[calc+'_Ultimate'] = ultimate
        else:
            self.data['Ultimate'] = ultimate
