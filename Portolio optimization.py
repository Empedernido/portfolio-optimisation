import sys
import os
import time                                             # Runtime environment
import yfinance as yf                                   # Yahoo Finance API
import numpy as np                                      # Mathematical operations
import pandas as pd                                     # DataFrame manipulation
import altair                                           # Plotting
import matplotlib.pyplot as plt
import scipy.optimize as sc                             # Linprog
import multiprocessing as mp                            # Process branching

# Auxiliary modules
import imageio as io
from pypfopt import risk_models, expected_returns, EfficientFrontier, EfficientSemivariance



                                                        # Stock Ticker input yet to be implemented
Tickers = ['UNH','JNJ','PFE','LLY','AZN',
           'SNY','AMGN','GSK','ANTM','ISRG',
           'SYK','SYK','GILD','REGN','EW','MRNA',
           'BSX','DIS','HUM','BNTX','ALGN',
           'BIIB','VTRS','ALNY','MOH','BMRN',
           'SNN','XRAY','MASI','RHHBY']
Portfolio = pd.DataFrame()
Strategy = None
Weights = np.array(np.zeros(15))
def Strategy_choice():
        global Strategy
        Strategy = input('Enter preferred investor profile: A; B or C ')
        if (Strategy == 'A') or (Strategy == 'B') or (Strategy == 'C'):
                print(f'Input recieved. Strategy ' + Strategy)
        else:
                Strategy = 'B'
                print(f'Input incorrect. Proceeding with the Balanced profile')
        return(Strategy)



class Extractor:
        def __init__(self, Tickers, Portfolio):
                self.Tickers = Tickers
                self.Portfolio = Portfolio
        def Upload(self, Portfolio, Lower_bound = '2017-01-01'):
                Portfolio = pd.DataFrame()
                for i in Tickers:
                        Portfolio[i] = yf.download(i, Lower_bound)['Close']
                return (Portfolio)

        def Wrangle(self, Portfolio):
                print(Portfolio.head(5))
                Returns = Portfolio.pct_change()
                Returns = Returns.fillna(0)
                Returns_15 = Returns.iloc[0: , 0:15]
                Asset_Weights = np.array(np.ones(15)*(1/15))
                print(Asset_Weights)
                Returns_15.plot(figsize=(5,3))
                plt.show()
                print(Returns_15.head(5))
                print(len(Returns_15.index))
                Default_cumulative = Returns_15.cumsum()
                Default_cumulative.plot(figsize=(5, 3))
                plt.show()
                Covariance = Returns_15.cov()*252
                Variance= np.dot(Asset_Weights.T, np.dot(Covariance, Asset_Weights))
                Volatility  = np.sqrt(Variance)
                Volatility_percentage = str(round(Volatility, 2) *100)
                print('Base portfolio volatility: ' + Volatility_percentage + '%')
                Portfolio_Returns = np.sum(Returns_15.mean()*Asset_Weights)*252
                Portfolio_Returns_percentage = str(round(Portfolio_Returns, 3)*100)
                print('Base portfolio returns: ' + Portfolio_Returns_percentage + '%')
                return Portfolio, Returns, Returns_15, Covariance, Variance, Volatility


class Repacking_Opt:
        def __init__(self, Returns_15, Weights, Strategy):
                self.Returns_15 = Returns_15
                self.Weights = Weights
                self.Strategy = Strategy
        # def Process_relay (Function)
        # L1 = mp.process(None)
        # L2 = mp.process(None)
        def Biennial (Returns_15, Volatility, Weights, Strategy):
                pass
        def Fortnight (Returns_15, Volatility, Strategy):
                pass
class Module_Opt:
        def __init__(self, Returns_15, Weights):
                self.Portfolio = Portfolio
                self.Weights = Weights
        def Optimization (Returns_15, Weights):

                Ereturns = Returns_15.mean_historical_return(Returns_15)
                Covariance = Returns_15.cov()
                Eff_Frontier = EfficientFrontier(Ereturns, Covariance, solver = 'SCS')
                Volatility = Eff_Frontier.min_volatility()
                Result = Eff_Frontier.clean_weights()

                print(Result)
                Eff_Frontier.portfolio_performance(verbose=True)



# Execution starts here!
                                                                                # Recieve the investment protfolio profile
Strategy_choice()
                                                                                # Extract the data with Yahoo Finance API, remodel the resulting DataFrame for further optimization
Full = Extractor(Tickers, Portfolio)
Portfolio = Full.Upload(Portfolio)
sys.stdout.write('Extraction finished...')
Portfolio, Returns, Returns_15, Covariance, Variance, Volatility = Full.Wrangle(Portfolio)
                                                                                # Find daily returns, Split the DataFrame, Find the Covariance matrix, Set asset weights to be equal, Find the variance, Find the portfolio st. dev (volatility)
sys.stdout.write('Wrangling finished...')                                       # Find average annual portfolio returns with equal weights
Full_Repacking = Repacking_Opt(Returns_15, Weights, Strategy)                   # Perform a rolling window method portfolio optimization. The window is set by defaul to be half a year, though other time horizons (such as week-wise) should be considered (as well as optimal window size could be found)
                                                                                # Find the optimal weights for each window instance with LP with average returns function being maximized for Aggresive strategy, volatility function for Conservative and average returns function with volatility constrained at 1%
sys.stdout.write('Rolling window method finished...')                           # Find the resulting volatility and returns
Additional = Module_Opt(Returns_15, Weights)
Module_Opt.Optimization(Returns_15, Weights)
sys.stdout.write('Additional optimization finished...')                         # Optimize using the PyPortfolioOpt module, Including min_volatility(), efficient_risk(), max_sharpe(); Plot efficient frontier, output portfolio performance

 # Plot daily return ratio unoptimized, optimized with the Rolling window method, find Sharpe ratio for it, find Sortino ratio for it. Save as GIFs, import to Blender and render (manually)
sys.stdout.write('Finished.')