import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

#defining tickers which this programme will perform calculations on, in this code I will principally focus on AAPL
tickers = 'AAPL'
initial_balance = 10000

#extracting data from yf
raw_data = yf.download(
    tickers = tickers,
    start = '2021-01-01', #defining start date
    end = '2025-01-01', #defining end date
    interval = '1d', #defining interval at which the programme will read data
    ignore_tz= True, #ignores the fact that AAPL is traded in a different timezone as compared to the UK
    auto_adjust = True, #automatically adjusts for splits and dividends
)

raw_data = raw_data['Close'] #selects only the data from markets close

AAPL_data = raw_data[[tickers]].copy()
AAPL_data.columns = ['Close']

#calculating the rolling average for the initial values of 50 days as a slow SMA and 10 days as a fast SMA
AAPL_data['Slow_init'] = AAPL_data['Close'].rolling(50).mean()
AAPL_data['Fast_init'] = AAPL_data['Close'].rolling(10).mean()
AAPL_data = AAPL_data.dropna() #removing NaN lines to ensure smooth running of later mathematical functions
AAPL_data = AAPL_data.assign(Signal = lambda x: np.where(x['Fast_init'] > x['Slow_init'],1,-1)) #reads whether fast SMA rises more quickly than slow SMA, this would indicate a buildup of positive momentum which in this strategy we want to jump on as quickly as possible
AAPL_data['Signal'] = AAPL_data['Signal'].shift(1) #since we are reading the close prices of each day we cannot react on the same day, thus the signal of buying/selling will be shifted down by one day assuming that this is the required time it takes us to react
AAPL_data = AAPL_data.dropna()

AAPL_data['Daily_change'] = AAPL_data['Close'].pct_change()#reading the daily pct change of the market such that we can calculate returns employing this data
AAPL_data['Strategy_daily_change'] = AAPL_data['Daily_change']*AAPL_data['Signal']
AAPL_data['R_strategy'] = initial_balance *(1 + AAPL_data['Strategy_daily_change']).cumprod() #calculating the instantaneous value of our portfolio when using our strategy
AAPL_data['Buy&Hold'] = initial_balance *(1+AAPL_data['Daily_change']) #calculating the portfolio value employing a very simple buy&hold strategy
#AAPL_data[['Buy&Hold','R_strategy']].plot(figsize = (10,5), color = ('Black','Blue')) #this graph very clearly shows that this combination of intervals is not efficient and that as a result Buy&hold is much more effective
#plt.show()
#uncomment lines above to see performance of unoptimised strategy against Buy&Hold


#we will now attempt to optimise the intervals and thus obtain a strategy which is optimised with respect to sharpe ratio and hopefully could beat B&H on this specific interval
def s_ratio(strategy_returns,risk_free_returns = 0.0): #this function calculates the sharpe ratio of a given strategy, it is scaled by 252 because that is the total number of days that the markets are open
    return (
            (strategy_returns.mean() - risk_free_returns.mean()) * 252 /
            (strategy_returns.std() * np.sqrt(252))
    )

AAPL_data = raw_data[[tickers]].copy() #resetting the database for the AAPL ticker as the previous lines will have removed some lines due to dropna, this will affect the results in the next lines
AAPL_data.columns = ['Close']
optimisation_temporary_data = [] #creating a temporary 2D dictionary to hold the combination of slow and fast timeframe tested and the s_ratio it results in

#iterating through the values between 30 and 300 with a step of 5  for the slow timeframe and the values between 5 and 50 for the fast timeframe
for slow in range(30,300,5):
    for fast in range(5,50,5):
        AAPL_data['Slow_temp'] = AAPL_data['Close'].rolling(slow).mean()
        AAPL_data['Fast_temp'] = AAPL_data['Close'].rolling(fast).mean()
        AAPL_data = AAPL_data.dropna()
        AAPL_data = AAPL_data.assign(Signal = lambda x: np.where(x['Fast_temp']>x['Slow_temp'],1,-1))
        AAPL_data['Signal'] = AAPL_data['Signal'].shift(1)
        AAPL_data = AAPL_data.dropna()
        AAPL_data['Daily_change'] = AAPL_data['Close'].pct_change()
        AAPL_data = AAPL_data.dropna()
        AAPL_data['Strategy_daily_change'] = AAPL_data['Daily_change']*AAPL_data['Signal']
        AAPL_data['R_strategy'] = initial_balance*(1+AAPL_data['Strategy_daily_change']).cumprod()
        AAPL_data = AAPL_data.dropna()
        optimisation_temporary_data.append([slow,fast,s_ratio(AAPL_data['Strategy_daily_change'],AAPL_data['Daily_change'])]) #appending the data obtained in one iteration to the temporary list

optimisation_temporary_data = pd.DataFrame(optimisation_temporary_data, columns = ['Slow','Fast','S_ratio']) #converting the temporary list into a pandas dataframe to allow for easy manipulation
max_sharpe_index = optimisation_temporary_data['S_ratio'].dropna().idxmax() #get the index of the maximum sharp value which is obtained
print(optimisation_temporary_data.iloc[max_sharpe_index]) #printing the timeframes and sharpe value obtained out

slow_opt, fast_opt = optimisation_temporary_data.loc[max_sharpe_index,['Slow','Fast']].to_list() #reading out the slow and fast timeframe value into two variables: slow_opt and fast_opt
AAPL_data = raw_data[[tickers]].copy() #again resetting the dataframe
AAPL_data.columns = ['Close']
AAPL_data['Slow_opt'] = AAPL_data['Close'].rolling(int(slow_opt)).mean()
AAPL_data['Fast_opt'] = AAPL_data['Close'].rolling(int(fast_opt)).mean()
AAPL_data = AAPL_data.dropna()
AAPL_data = AAPL_data.assign(Signal_opt = lambda x: np.where(x['Fast_opt']>x['Slow_opt'],1,-1))
AAPL_data = AAPL_data.dropna()
AAPL_data['Daily_change'] = AAPL_data['Close'].pct_change()
AAPL_data['Strategy_daily_change'] = AAPL_data['Daily_change']*AAPL_data['Signal_opt']
AAPL_data['B&Hold'] = initial_balance*(1+AAPL_data['Daily_change']).cumprod()
AAPL_data['Strategy_returns'] = initial_balance*(1+AAPL_data['Strategy_daily_change']).cumprod()
AAPL_data = AAPL_data.dropna()
AAPL_data[['Strategy_returns','B&Hold']].plot(figsize = (10,5), color = ('Black','Blue')) #plotting the instantaneous portfolio value when employing our optimised strategy and employing Buy&Hold, in this case Buy&Hold is beaten by our strategy however this is likely not true for different timeframes
plt.show()




