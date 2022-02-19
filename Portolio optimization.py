import sys
import os
import time                                             # Runtime environment
import yfinance as yf                                   # Yahoo Finance API
import numpy as np                                      # Mathematical operations
import pandas as pd                                     # DataFrame manipulation
import altair                                           # Plotting
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
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
        def Upload(self, Portfolio, Lower_bound = '2015-01-01'):
                Portfolio = pd.DataFrame()
                for i in Tickers:
                        Portfolio[i] = yf.download(i, Lower_bound)['Close']
                return (Portfolio)

        def Wrangle(self, Portfolio):
                Returns = Portfolio.pct_change()
                Returns = Returns.fillna(0)
                Returns_15 = Returns.iloc[0: , 0:15]
                Asset_Weights = np.array(np.ones(15)*(1/15))
                Returns_15.plot(figsize=(5,3))
                plt.ylabel("%")
                plt.title('Daily fluctuations of asset price')
                plt.axvspan('2019', '2020', color='#F9F9F9', label='axvline - full height')
                plt.axvspan('2017', '2018', color='#F9F9F9', label='axvline - full height')
                plt.axvspan('2021', '2022', color='#F9F9F9', label='axvline - full height')
                plt.axvspan('2015', '2016', color='#F9F9F9', label='axvline - full height')
                print(Returns_15.head(5))
                print(len(Returns_15.index))
                Default_cumulative = Returns_15.cumsum()
                Default_cumulative.plot(figsize=(5, 3))
                plt.ylabel("%")
                plt.title('Cumulative returns by asset')
                plt.axvspan('2019','2020', color='#F9F9F9', label='axvline - full height')
                plt.axvspan('2017', '2018', color='#F9F9F9', label='axvline - full height')
                plt.axvspan('2021', '2022', color='#F9F9F9', label='axvline - full height')
                plt.axvspan('2015', '2016', color='#F9F9F9', label='axvline - full height')

                plt.show()
                Covariance = Returns_15.cov()*252
                Variance= np.dot(Asset_Weights.T, np.dot(Covariance, Asset_Weights))
                Volatility  = np.sqrt(Variance)
                Volatility_percentage = str(round(Volatility, 2) *100)
                print('Base portfolio volatility: ' + Volatility_percentage + '%')
                Portfolio_Returns = np.sum(Returns_15.mean()*Asset_Weights)*252
                Portfolio_Returns_percentage = str(round(Portfolio_Returns, 3)*100)
                print('Base portfolio returns: ' + Portfolio_Returns_percentage + '%')
                Historical_Returns = Returns_15.sum(axis=1)
                print(Historical_Returns.head(5))
                Historical_Returns_total = Historical_Returns.cumsum()

                Historical_Returns_total.plot(cmap = 'Accent')
                plt.ylabel("%")
                plt.title('Total portfolio return with equally weighted shares')
                plt.show()
                return Portfolio, Returns, Returns_15, Covariance, Variance, Volatility


class Repacking_Opt:
        def __init__(self, Returns_15, Weights, Strategy):
                self.Returns_15 = Returns_15
                self.Weights = Weights
                self.Strategy = Strategy
        # def Process_relay (Function)
        # L1 = mp.process(None)
        # L2 = mp.process(None)
        def Biannual (Returns_15, Volatility, Weights, Strategy):
                pass
        def Fortnight (Returns_15, Volatility, Strategy):
                pass

class Module_Repacking:
        def __init__(self, Returns_15, Weights, Strategy):
                self.Portfolio = Portfolio
                self.Weights = Weights
                self.Strategy = Strategy
        def Optimization (self, Returns_15, Weights, Strategy):
                if Strategy == 'A' :
                        print('Starting Aggressive portfolio optimization (High volatility acceptable)')
                        Risk_free_rate = 0.01
                        Iterations = round(len(Returns_15.index)/126)
                        n=0
                        rl=0
                        ru=125
                        Optimized_Returns = pd.DataFrame
                        while n < Iterations :
                                # Select a window of 126 days for 14 times, the resultim dataframe is used from now on
                                if n >= 8:
                                        Window = Returns_15.iloc[rl:ru]
                                        rl = rl + 126
                                        ru = ru + 126
                                        # Procure all the optimization stuff
                                        E_Returns = Window.mean()

                                        Covariance_W = Window.cov()

                                        EF = EfficientFrontier(E_Returns, Covariance_W, solver = 'SCS')
                                        EF.efficient_risk(target_volatility=0.026)
                                        Clean_Weights = EF.clean_weights()
                                        print(Clean_Weights)
                                        Weights = EF.weights

                                        # The final outputs are the plot of weights for this period, as well as the 1 column dataframe with daily returns for the period
                                        Window= np.dot(Window, Weights)
                                        print(Window)
                                        n = n + 1  # MOVE THIS COUNTER TO THE BACK OF THE LOOP!!!
                                else:
                                        Window = Returns_15.iloc[rl:ru,0:14]
                                        rl=rl+126
                                        ru=ru+126
                                        # Procure all the optimization stuff
                                        E_Returns = Window.mean()

                                        Covariance_W = Window.cov()

                                        EF = EfficientFrontier(E_Returns, Covariance_W, solver = 'SCS')
                                        EF.efficient_risk(target_volatility=0.026)
                                        Clean_Weights = EF.clean_weights()
                                        print(Clean_Weights)

                                        Weights = EF.weights
                                        print(Window)
                                        # The final outputs are the plot of weights for this period, as well as the 1 column dataframe with daily returns for the period
                                        Window= np.dot(Window, Weights)
                                        Optimized_Returns['Portfolio']=pd.Series(Window)
                                        n = n + 1  # MOVE THIS COUNTER TO THE BACK OF THE LOOP!!!
                        print(Window.portfolio_performance(verbose=True))
                        Historical_Returns = Optimized_Returns.sum(axis=1)

                        Historical_Returns_total = Historical_Returns.cumsum()

                        Historical_Returns_total.plot()
                        plt.ylabel("%")
                        plt.title('Aggressive portfolio yield')
                        plt.show()
                elif Strategy == 'B':
                        print('Starting Balanced portfolio optimization (Maximizing Sharpe ratio)')
                        Risk_free_rate = 0.01
                        Iterations = round(len(Returns_15.index)/126)
                        n=0
                        rl=0
                        ru=125
                        Optimized_Returns = pd.DataFrame()
                        while n < Iterations :
                                # Select a window of 126 days for 14 times, the resultim dataframe is used from now on
                                if n >= 8:
                                        Window = Returns_15.iloc[rl:ru]
                                        rl = rl + 126
                                        ru = ru + 126
                                        # Procure all the optimization stuff
                                        E_Returns = Window.mean()

                                        Covariance_W = Window.cov()

                                        EF = EfficientFrontier(E_Returns, Covariance_W, solver = 'SCS')
                                        Weights = ef.max_sharpe()
                                        Clean_Weights = EF.clean_weights()
                                        print(Clean_Weights)


                                        # The final outputs are the plot of weights for this period, as well as the 1 column dataframe with daily returns for the period
                                        Optimized_Returns.iloc[ru:rl] = np.dot(Optimized_Returns.iloc[ru:rl], Weights)
                                        n = n + 1  # MOVE THIS COUNTER TO THE BACK OF THE LOOP!!!
                                else:
                                        Window = Returns_15.iloc[rl:ru,0:14]
                                        rl=rl+126
                                        ru=ru+126
                                        # Procure all the optimization stuff
                                        E_Returns = Window.mean()

                                        Covariance_W = Window.cov()

                                        EF = EfficientFrontier(E_Returns, Covariance_W, solver='SCS')
                                        Weights = ef.max_sharpe()
                                        Clean_Weights = EF.clean_weights()
                                        print(Clean_Weights)

                                        Weights = EF.weights

                                        # The final outputs are the plot of weights for this period, as well as the 1 column dataframe with daily returns for the period
                                        Optimized_Returns.iloc[ru:rl, 0:14] = np.dot(Optimized_Returns.iloc[ru:rl, 0:14], Weights)
                                        n = n + 1
                        print(Optimized_Returns.portfolio_performance(verbose=True))
                        Historical_Returns = Optimized_Returns.sum(axis=1)

                        Historical_Returns_total = Historical_Returns.cumsum()

                        Historical_Returns_total.plot()
                        plt.ylabel("%")
                        plt.title('Aggressive portfolio yield')
                        plt.show()
                elif Strategy == 'C':
                        print('Starting Conservative portfolio optimization (Minimal volatility acceptable)')

                        Risk_free_rate = 0.01
                        Iterations = round(len(Returns_15.index)/126)
                        n=0
                        rl=0
                        ru=125
                        Optimized_Returns = pd.DataFrame()
                        while n < Iterations :
                                # Select a window of 126 days for 14 times, the resultim dataframe is used from now on
                                if n >= 8:
                                        Window = Returns_15.iloc[rl:ru]
                                        rl = rl + 126
                                        ru = ru + 126
                                        # Procure all the optimization stuff
                                        E_Returns = Window.mean()

                                        Covariance_W = Window.cov()

                                        EF = EfficientFrontier(E_Returns, Covariance_W, solver = 'SCS')
                                        EF.efficient_risk(target_volatility=0.021)
                                        Clean_Weights = EF.clean_weights()
                                        print(Clean_Weights)
                                        Weights = EF.weights

                                        # The final outputs are the plot of weights for this period, as well as the 1 column dataframe with daily returns for the period
                                        Window= np.dot(Window, Weights)
                                        n = n + 1  # MOVE THIS COUNTER TO THE BACK OF THE LOOP!!!
                                else:
                                        Window = Returns_15.iloc[rl:ru,0:14]
                                        rl=rl+126
                                        ru=ru+126
                                        # Procure all the optimization stuff
                                        E_Returns = Window.mean()

                                        Covariance_W = Window.cov()

                                        EF = EfficientFrontier(E_Returns, Covariance_W, solver = 'SCS')
                                        EF.efficient_risk(target_volatility=0.021)
                                        Clean_Weights = EF.clean_weights()
                                        print(Clean_Weights)

                                        Weights = EF.weights

                                        # The final outputs are the plot of weights for this period, as well as the 1 column dataframe with daily returns for the period
                                        Window= np.dot(Window, Weights)
                                        n = n + 1  # MOVE THIS COUNTER TO THE BACK OF THE LOOP!!!
                        print(Optimized_Returns.portfolio_performance(verbose=True))
                        Historical_Returns = Optimized_Returns.sum(axis=1)

                        Historical_Returns_total = Historical_Returns.cumsum()

                        Historical_Returns_total.plot()
                        plt.ylabel("%")
                        plt.title('Aggressive portfolio yield')
                        plt.show()

                # After constructing an optimized returns dataframe based on the optimized dailiy returns
                # Optimized_Daily_Returns.plot()
                # Optimized_Daily_Returns.portfolio_performance(verbose=True)
                # Annualized_returns = None
                # print('Annualized_returns ' + str(Annualized_returns * 100) + ' %')
                # Annualized_volatility = None
                # print('Annualized_volatility '+ str(Annualized_volatility * 100) + ' %')
                # Done separately *Should add graphical depiction later
                # With expected returns plot, find minima and maxima, allowing to calculate drawdown and recovery period for this window
                # i = np.argmax(np.maximum.accumulate(Window) - Window)  # end of the period
                # j = np.argmax(Window[:i])  # start of period

                # plt.plot(Window)
                # plt.plot([i, j], [Window[i], Window[j]], 'o', color='Red', markersize=10)
                # Maximum_drawdown = None
                # print('Maximum_drawdown' + str(Maximum_drawdown) + ' %')
                # Maximum_recovery_period = None
                # print('Maximum_recovery_period' + str(Maximum_recovery_period) + ' Days')




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
#Full_Repacking = Repacking_Opt(Returns_15, Weights, Strategy)                   # Perform a rolling window method portfolio optimization. The window is set by defaul to be half a year, though other time horizons (such as week-wise) should be considered (as well as optimal window size could be found)
                                                                                # Find the optimal weights for each window instance with LP with average returns function being maximized for Aggresive strategy, volatility function for Conservative and average returns function with volatility constrained at 1%
#sys.stdout.write('Rolling window method finished...')                           # Find the resulting volatility and returns
Additional = Module_Repacking(Returns_15, Weights, Strategy)
Additional.Optimization(Returns_15, Weights, Strategy)
sys.stdout.write('Additional optimization finished...')                         # Optimize using the PyPortfolioOpt module, Including min_volatility(), efficient_risk(), max_sharpe(); Plot efficient frontier, output portfolio performance
print('...')
 # Plot daily return ratio unoptimized, optimized with the Rolling window method, find Sharpe ratio for it, find Sortino ratio for it. Save as GIFs, import to Blender and render (manually)
sys.stdout.write('Finished.')
