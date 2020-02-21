# Fully_Finance
Contains various functions and classes to assist with visualization and analytics of trading data in python
By James Altemus

#############################################################################################

Begin by loading the _base.py file. This file contains 3 functions:
> csv_loader: loads from a CSV. It can convert periods (i.e. from days to months) using either a US trading calendar or a UK trading calendar.

> multi_loader: loads a list of securities from an internet source (default Yahoo Finance) given a list of tickers, a start date, and an end date.
		It converts periods, and can use either US or UK trading calendars. This function will only take 1 column of data for each security.
		The last column is taken by default, this is Adjusted Close for Yahoo Finance.

> web_loader: loads a single security from an internet source (default Yahoo Finance) given a list of tickers, a start date, and an end date.
	      It converts periods, and can use either US or UK trading calendars. It will take all information from the API source.

#############################################################################################

After loading security data, load _core.py and call the PortfolioBuilder class.
Using PortfolioBuilder, returns can be calculated, analytics can be added (currently only supports some indicators), or the portfolio can be optimized.

# Analytics:
Analytics currently supports Bollinger Bands, Chande Momentum Oscillator, Ichimoku Cloud, MACD. RSI, Stochastic Oscillator, and Wilder Smoothing.
To add analytic functions simply call .add_analytics(). Then you can call an indicator function, such as .bollinger(), and it will add a new feature, .BollingerBands,
to the base class with the indicator calculations for each constituent of the portfolio.

# Optimization:
Optimization offers the following goals as of 0.0: Sharpe Ratio, Utility, Value at Risk, Tail Ratio, Alpha, Alpha Edge, Sharpe Edge, and Beta.
To add optimization call .add_optimization(). Then call .OptimizePortfolio.objective() to set an objective. Constraints follow the method .OptimizePortfolio.constraint...
and features follow .OptimizePortfolio.feature.... Calling .OptimizePortfolio.objective_parameters contains the current objectives, .OptimizePortfolio.features
contains the current features, and .OptimizePortfolio.constraints contains the current constraints. After choosing constraints, call .OptimizePortfolio.optimize().
> .optimize: feeds the added features and constraints into an optimizer. It can use either geometric or arithmetic mean operations, and can generate an efficent frontier.
	     The only goal that cannot generate an efficent frontier is Alpha. The remainder of the parameters are for the scipy optimizer.

After optimization is finished, .OptimizePortfolio.'feature_name' contains the optimized portfolio information. I'll use Sharpe_Ratio as an example:
> To get the optimal (tangency) portfolio, call .OptimizePortfolio.SharpeRatio.OptimalSharpeRatio
> To get the scipy optimization information call .OptimizePortfolio.SharpeRatio.OptiParam
> To get the full optimization parameters used call .OptimizePortfolio.SharpeRatio.Parameters
> To get the constituent information call .OptimizePortfolio.SharpeRatio.ConstituentSharpe
> If an efficent frontier was generated, call .OptimizePortfolio.SharpeRatio.EfficentFrontier to list the information for each of the analysed portfolios.
> If an efficent frontier was generated, call .OptimizePortfolio.SharpeRatio.EfficentParam to list the scipy optimization information for each of the analysed portfolios.

If you have any suggestions, or comments, please email me at jamesaltem@gmail.com
