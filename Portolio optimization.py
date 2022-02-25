import sys
import glob
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
Tickers = ['ORCL','JNJ','PFE','LLY','AZN',
           'ANTM','ISRG','LUV','SAVE','DAL',
           'ATVI','TSLA','NVDA','STX','FTNT',
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
                Default_cumulative = Returns_15.cumsum() * 10
                Default_cumulative.plot(figsize=(5, 3))
                plt.ylabel("%")
                plt.title('Cumulative returns by asset')
                plt.axvspan('2019','2020', color='#F9F9F9', label='axvline - full height')
                plt.axvspan('2017', '2018', color='#F9F9F9', label='axvline - full height')
                plt.axvspan('2021', '2022', color='#F9F9F9', label='axvline - full height')
                plt.axvspan('2015', '2016', color='#F9F9F9', label='axvline - full height')

                # plt.show()
                Covariance = Returns_15.cov()*252
                print(Covariance)
                Variance= np.dot(Asset_Weights.T, np.dot(Covariance, Asset_Weights))
                print('Variance' + str(Variance))
                Volatility  = np.sqrt(Variance)
                Volatility_percentage = str(round(Volatility, 2) *100)
                print('Base portfolio volatility: ' + Volatility_percentage + '%')
                Portfolio_Returns = np.sum(Returns_15.mean()*Asset_Weights)*252
                Portfolio_Returns_percentage = str(round(Portfolio_Returns, 3)*100)
                print('Base portfolio returns: ' + Portfolio_Returns_percentage + '%')
                Historical_Returns = Returns_15.sum(axis=1)
                Historical_Returns_total = Historical_Returns.cumsum()
                Historical_Returns_total.plot(cmap = 'Accent')
                plt.ylabel("%")
                plt.title('Total portfolio return with equally weighted shares')
                # plt.show()
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
        def Optimization (self, Returns_15, Weights, Strategy, Portfolio):
                if Strategy == 'A' :
                        print('Starting Aggressive portfolio optimization (High volatility acceptable)')
                        print(Portfolio)
                        Risk_free_rate = 0.01
                        Iterations = round(len(Portfolio.index) / 126)
                        n = 0
                        rl = 0
                        ru = 125
                        Optimized_Returns = Returns_15
                        while n < Iterations:
                                # Select a window of 126 days for 14 times, the resulting dataframe is used from now on
                                Window = Portfolio.iloc[rl:ru, 1:15]
                                rl = rl + 126
                                ru = ru + 126
                                # Procure all the optimization stuff
                                E_Returns = expected_returns.mean_historical_return(Window)

                                Covariance_W = risk_models.sample_cov(Window)

                                EF = EfficientFrontier(E_Returns, Covariance_W, solver='SCS')
                                Weights = EF.efficient_risk(target_volatility=0.45)
                                # Weights = EF.max_sharpe()
                                Storno_Weights = abs(EF.weights)
                                print(Storno_Weights)
                                Clean_Weights = EF.clean_weights()
                                print(Clean_Weights)
                                plt.pie(Storno_Weights, colors = ['#5B064D','#811F71','#AA529C','#CA6BBB','#B36BCA','#B249D3','#9F31C1','#2373A0','#D3C81A','#D3251A','#1A97D3','#A4C7D8','#64821F','#92BE2E','#020301'], wedgeprops={"edgecolor":"#FAFAFA",'linewidth': 1, 'antialiased': True})
                                plt.legend(Storno_Weights, labels=('ORCL','JNJ','PFE','LLY','AZN','ANTM','ISRG','LUV','SAVE','DAL','ATVI','TSLA','NVDA','STX','FTNT'), loc='best', bbox_to_anchor=(-0.1, 1.), fontsize=8)
                                plt.savefig(r'C:\\FiguresA\piechart' + str(n) + '.png')
                                plt.clf()
                                Window['Portfolio'] = np.dot(Window, Storno_Weights)
                                print(Window)
                                print(EF.portfolio_performance(verbose=True))
                                Window = Window['Portfolio']
                                Window.to_csv(r'C:\\CSVpricesA\Cprices' + str(n + 10) + '.csv')
                                # The final outputs are the plot of weights for this period, as well as the 1 column dataframe with daily returns for the period
                                # Window= np.dot(Window, Weights)

                                # Optimized_Returns= np.hstack((Optimized_Returns, Window))

                                n = n + 1  # MOVE THIS COUNTER TO THE BACK OF THE LOOP!!!
                        
                        os.chdir(r'C:\\CSVpricesA')
                        extension = 'csv'
                        all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
                        # Combine all files in the list
                        combined_csvA = pd.concat([pd.read_csv(f) for f in all_filenames])
                        # Export to csv
                        combined_csvA.to_csv("combined_csvA.csv", index=False, encoding='utf-8-sig')
                        FINAL = pd.read_csv(r'C:\\CSVpricesA\combined_csvA.csv', index_col=0)
                        print(FINAL)
                        FINAL.portfolio_efficiency
                        FINAL = FINAL.pct_change()
                        Historical_Returns_total = FINAL.cumsum() * 10
                        Historical_Returns_total.plot(color='#5B064D')
                        plt.ylabel("%")
                        plt.title('Total portfolio return with the Aggresive strategy')
                        plt.show()

                elif Strategy == 'B':


                        print('Starting Balanced portfolio optimization (Maximum Sharpe Ratio)')
                        print(Portfolio)
                        Risk_free_rate = 0.01
                        Iterations = round(len(Portfolio.index) / 126)
                        n = 0
                        rl = 0
                        ru = 125
                        Optimized_Returns = Returns_15
                        while n < Iterations:
                                # Select a window of 126 days for 14 times, the resulting dataframe is used from now on
                                Window = Portfolio.iloc[rl:ru, 1:15]
                                rl = rl + 126
                                ru = ru + 126
                                # Procure all the optimization stuff
                                E_Returns = expected_returns.mean_historical_return(Window)

                                Covariance_W = risk_models.sample_cov(Window)

                                EF = EfficientFrontier(E_Returns, Covariance_W, solver='SCS')
                                # Weights = EF.efficient_risk(target_volatility=0.45)
                                Weights = EF.max_sharpe()
                                # Weights = EF.min_volatility()
                                Storno_Weights = abs(EF.weights)
                                print(Storno_Weights)
                                Clean_Weights = EF.clean_weights()
                                print(Clean_Weights)
                                plt.pie(Storno_Weights, colors = ['#5B064D','#811F71','#AA529C','#CA6BBB','#B36BCA','#B249D3','#9F31C1','#2373A0','#D3C81A','#D3251A','#1A97D3','#A4C7D8','#64821F','#92BE2E','#020301'], wedgeprops={"edgecolor":"#FAFAFA",'linewidth': 1, 'antialiased': True})
                                plt.legend(Storno_Weights, labels=('ORCL','JNJ','PFE','LLY','AZN','ANTM','ISRG','LUV','SAVE','DAL','ATVI','TSLA','NVDA','STX','FTNT'), loc='best', bbox_to_anchor=(-0.1, 1.), fontsize=8)
                                plt.savefig(r'C:\\FiguresB\piechart' + str(n) + '.png')
                                plt.clf()
                                Window['Portfolio'] = np.dot(Window, Storno_Weights)
                                print(Window)
                                print(EF.portfolio_performance(verbose=True))
                                Window = Window['Portfolio']
                                Window.to_csv(r'C:\\CSVpricesB\Cprices' + str(n + 10) + '.csv')
                                # The final outputs are the plot of weights for this period, as well as the 1 column dataframe with daily returns for the period
                                # Window= np.dot(Window, Weights)

                                # Optimized_Returns= np.hstack((Optimized_Returns, Window))

                                n = n + 1  # MOVE THIS COUNTER TO THE BACK OF THE LOOP!!!
                        os.chdir(r'C:\\CSVpricesB')
                        extension = 'csv'
                        all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
                        # combine all files in the list
                        combined_csvC = pd.concat([pd.read_csv(f) for f in all_filenames])
                        # Export to csv
                        combined_csvC.to_csv("combined_csvB.csv", index=False, encoding='utf-8-sig')
                        FINAL = pd.read_csv(r'C:\\CSVpricesB\combined_csvB.csv', index_col=0)
                        print(FINAL)
                        FINAL = FINAL.pct_change()
                        Historical_Returns_total = FINAL.cumsum() * 10
                        Historical_Returns_total.plot(color='#5B064D')
                        plt.ylabel("%")
                        plt.title('Total portfolio return with the Balanced strategy')
                        plt.show()

                elif Strategy == 'C':
                        print('Starting Conservative portfolio optimization (Minimal volatility acceptable)')

                        print('Starting Aggressive portfolio optimization (High volatility acceptable)')
                        print(Portfolio)
                        Risk_free_rate = 0.01
                        Iterations = round(len(Portfolio.index) / 126)
                        n = 0
                        rl = 0
                        ru = 125
                        Optimized_Returns = Returns_15
                        while n < Iterations:
                                # Select a window of 126 days for 14 times, the resulting dataframe is used from now on
                                Window = Portfolio.iloc[rl:ru, 1:15]
                                rl = rl + 126
                                ru = ru + 126
                                # Procure all the optimization stuff
                                E_Returns = expected_returns.mean_historical_return(Window)

                                Covariance_W = risk_models.sample_cov(Window)

                                EF = EfficientFrontier(E_Returns, Covariance_W, solver='SCS')
                                # Weights = EF.efficient_risk(target_volatility=0.45)
                                # Weights = EF.max_sharpe()
                                Weights = EF.min_volatility()
                                Storno_Weights = abs(EF.weights)
                                print(Storno_Weights)
                                Clean_Weights = EF.clean_weights()
                                print(Clean_Weights)
                                # plt.pie(Storno_Weights, colors = ['#5B064D','#811F71','#AA529C','#CA6BBB','#B36BCA','#B249D3','#9F31C1','#2373A0','#D3C81A','#D3251A','#1A97D3','#A4C7D8','#64821F','#92BE2E','#020301'], wedgeprops={"edgecolor":"#FAFAFA",'linewidth': 1, 'antialiased': True})
                                # plt.legend(Storno_Weights, labels=('ORCL','JNJ','PFE','LLY','AZN','ANTM','ISRG','LUV','SAVE','DAL','ATVI','TSLA','NVDA','STX','FTNT'), loc='best', bbox_to_anchor=(-0.1, 1.), fontsize=8)
                                # plt.savefig(r'C:\\Figures\piechart' + str(n) + '.png')
                                # plt.clf()
                                Window['Portfolio'] = np.dot(Window, Storno_Weights)
                                print(Window)
                                print(EF.portfolio_performance(verbose=True))
                                Window = Window['Portfolio']
                                Window.to_csv(r'C:\\CSVprices\Cprices' + str(n+10) + '.csv')
                                # The final outputs are the plot of weights for this period, as well as the 1 column dataframe with daily returns for the period
                                # Window= np.dot(Window, Weights)


                                # Optimized_Returns= np.hstack((Optimized_Returns, Window))

                                n = n + 1  # MOVE THIS COUNTER TO THE BACK OF THE LOOP!!!
                        os.chdir(r'C:\\CSVprices')
                        extension = 'csv'
                        all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
                        # combine all files in the list
                        combined_csvC = pd.concat([pd.read_csv(f) for f in all_filenames])
                        # Export to csv
                        combined_csvC.to_csv("combined_csvC.csv", index=False, encoding='utf-8-sig')
                        FINAL = pd.read_csv(r'C:\\CSVprices\combined_csvC.csv', index_col=0)
                        print(FINAL)
                        FINAL = FINAL.pct_change()
                        Historical_Returns_total = FINAL.cumsum() * 10
                        Historical_Returns_total.plot(color ='#5B064D')
                        plt.ylabel("%")
                        plt.title('Total portfolio return with the Conservative strategy')
                        plt.show()


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
Additional.Optimization(Returns_15, Weights, Strategy, Portfolio)
sys.stdout.write('Additional optimization finished...')                         # Optimize using the PyPortfolioOpt module, Including min_volatility(), efficient_risk(), max_sharpe(); Plot efficient frontier, output portfolio performance
print('...')
 # Plot daily return ratio unoptimized, optimized with the Rolling window method, find Sharpe ratio for it, find Sortino ratio for it. Save as GIFs, import to Blender and render (manually)
sys.stdout.write('Finished.')
