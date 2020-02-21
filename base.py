# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:04:36 2019

@author: James Altemus
"""

import pandas as pd

from pandas.tseries.offsets import CustomBusinessMonthBegin,CustomBusinessMonthEnd
from pandas.tseries.offsets import MonthBegin, MonthEnd
from pandas.tseries.holiday import *
from pandas_datareader.data import DataReader
from datetime import date


def csv_loader(file_name, sep = ',', periodicity = None, term = 'end',
               date_col = 0, region = 'US'):
    ''' Wraps pandas read_csv and perfoms calendar manipulations
    
    Args:
        ticker_list: list of tickers as strings to get information about
        
        sep: the delimiter for the file
        
        periodicity: the period type to change the data to. Accepts 'weekly', 'monthly'
        
        term: from what point to calculate the period from. Accepts 'start' or 'end'
        
        date_col: the index for the date column
        
        region: the country to use for calculations. Accepts 'US' or 'UK'
    
    Returns:
        A pandas data frame object containing the requested information
    '''
    price_data = pd.read_csv(file_name, sep = sep, index_col = date_col)
    price_data.index = pd.to_datetime(price_data.index)
    
    if periodicity:
        price_data = _reduce(price_data, periodicity.lower(), term.lower(), region.upper())
    
    return price_data


def multi_loader(ticker_list, start_date = None, end_date = None, source = 'yahoo',
                 retry = 3, pause = 0.001, sess = None, api_key = None, 
                 periodicity = 'daily', term = 'end', index = -1, region = 'US'):
    ''' Wraps pandas_datareader DataReader methods and perfoms calendar manipulations
    
    Args:
        ticker_list: list of tickers as strings to get information about
        
        start_date: the oldest date to get data for, default 1 year ago. Accepts either a string 'YYYY-MM-DD' or a datetime.date object.
        
        end_date: the most recent date to get data for, default today. Accepts either a string 'YYYY-MM-DD' or a datetime.date object.
        
        source: the place to get the data from, default 'yahoo'. Check the pandas_datareader documentation for a list of valid sources.
        
        retry: the amount of times to retry a query after failure, default 3
        
        pause: seconds to pause between queries, default 0.001
        
        sess: the session instance to be used. Accepts requests.sessions
        
        api_key: the api key for the data source if applicable
        
        periodicity: the period type to use. Accepts 'daily', 'weekly','monthly'
        
        term: from what point to calculate the period from. Accepts 'start' or 'end'
        
        index: the index of the source data frame to use in the portfolio, default -1
        
        region: the country to use for calculations. Accepts 'US' or 'UK'
    
    
    Returns:
        A pandas data frame object containing the requested information
    '''
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = str(date.today()).split('-')
        start_date = date(int(start_date[0])-1,int(start_date[1]),int(start_date[2]))
    
    price_data = {}
    ticker_list = [tk.upper() for tk in ticker_list]
    for ticker in ticker_list:
        price = DataReader(name = ticker, data_source = source, start = start_date, 
                           end = end_date, retry_count = retry, pause = pause,
                           session = sess, api_key = api_key)
        price = price[price.columns[index]]
        price_data[ticker] = price
    price_data = pd.DataFrame(price_data)
    price_data.columns = ticker_list
    
    if periodicity.lower() != 'daily':
        price_data = _reduce(price_data, periodicity.lower(), term.lower(), region.upper())
    
    return price_data


def web_loader(ticker, start_date = None, end_date = None, source = 'yahoo',
               retry = 3, pause = 0.001, sess = None, api_key = None,
               periodicity = 'daily', term = 'end', index = -1, region = 'US'):
    ''' Wraps pandas_datareader DataReader methods and perfoms calendar manipulations
    
    Args:
        
        ticker: list of tickers as strings to get information about
        
        start_date: the oldest date to get data for, default 1 year ago. Accepts either a string 'YYYY-MM-DD' or a datetime.date object.
        
        end_date: the most recent date to get data for, default today. Accepts either a string 'YYYY-MM-DD' or a datetime.date object.
        
        source: the place to get the data from, default 'yahoo'. Check the pandas_datareader documentation for a list of valid sources.
        
        retry: the amount of times to retry a query after failure, default 3
        
        pause: seconds to pause between queries, default 0.001
        
        sess: the session instance to be used. Accepts requests.sessions
        
        api_key: the api key for the data source if applicable
        
        periodicity: the period type to use. Accepts 'daily', 'weekly','monthly'
        
        term: from what point to calculate the period from. Accepts 'start' or 'end'
        
        index: the index of the source data frame to use in the portfolio, default -1
        
        region: the country to use for calculations. Accepts 'US' or 'UK'
    
    
    Returns:
        
        A pandas data frame object containing the requested information
    '''
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = str(date.today()).split('-')
        start_date = date(int(start_date[0])-1,int(start_date[1]),int(start_date[2]))
    
    price_data = []
    ticker = ticker.upper()
    price_data = DataReader(name = ticker, data_source = source, start = start_date, 
                            end = end_date, retry_count = retry, pause = pause,
                            session = sess, api_key = api_key)
    
    if periodicity.lower() != 'daily':
        price_data = _reduce(price_data, periodicity.lower(), term.lower(), region.upper())
    
    return price_data


def _reduce(price_data, period, term, region):
    calendar = build_trading_calendar(region)
    
    if term == 'start':
        if period == 'weekly':
            pmap = price_data.index.dayofweek
            price_data[pmap == 0]
        if period == 'monthly':
            calendar = CustomBusinessMonthBegin(calendar=calendar)
            
            st = price_data.index[0] + MonthBegin()
            end = price_data.index[-1] + MonthBegin()
            pmap = pd.date_range(st,end,freq = calendar)
            price_data['pmap'] = [1 if idx in pmap else 0 for idx in price_data.index]
            price_data = price_data[price_data['pmap'] == 1].drop('pmap', axis = 1)
    
    if term == 'end':
        if period == 'weekly':
            pmap = price_data.index.dayofweek
            price_data[pmap == 4]
        if period == 'monthly':
            calendar = CustomBusinessMonthEnd(calendar=calendar)
            
            st = price_data.index[0] + MonthEnd()
            end = price_data.index[-1] + MonthEnd()
            pmap = pd.date_range(st,end,freq = calendar)
            price_data['pmap'] = [1 if idx in pmap else 0 for idx in price_data.index]
            price_data = price_data[price_data['pmap'] == 1].drop('pmap', axis = 1)
    
    return price_data





def build_trading_calendar(ctype = 'US'):
    '''
    Args:
        ctype: the type of calendar. Accepts 'US' or 'UK'
    '''
    if ctype == 'US':
        return USTradingCalendar()
    if ctype == 'UK':
        return UKTradingCalendar()


class USTradingCalendar(AbstractHolidayCalendar):
    rules = USFederalHolidayCalendar.rules[:3]
    rules.append(GoodFriday)
    rules.extend(USFederalHolidayCalendar.rules[3:6])
    rules.extend(USFederalHolidayCalendar.rules[8:])


class UKTradingCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('NewYearsDay', month = 1, day = 1, observance=nearest_workday),
        GoodFriday,
        EasterMonday,
        Holiday('EarlyMayBankHoliday', month = 5, day = 1, offset=DateOffset(weekday = MO(1))),
        Holiday('LateMayBankHoliday', month = 5, day = 31, offset=DateOffset(weekday = MO(-1))),
        Holiday('SummerBankHoliday', month = 8, day = 31, offset=DateOffset(weekday = MO(-1))),
        Holiday('Christmas', month = 12, day = 25, observance=nearest_workday),
        Holiday('BoxingDay', month = 12, day = 26, observance=nearest_workday)
        ]