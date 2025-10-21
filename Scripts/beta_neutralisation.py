import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import optuna

# Load yfinance data
vanguard_finance_etf = "VFH"
vanguard_consumer_staples_etf = "VDC"
vanguard_consumer_disc_etf = "VCR"
vanguard_industrials_etf = "VIS"
vanguard_IT_etf = "VGT"


df = yf.download(
    tickers=[
        vanguard_finance_etf,
        vanguard_consumer_staples_etf,
        vanguard_consumer_disc_etf, vanguard_industrials_etf, vanguard_IT_etf],
    start='2021-07-03',
    end='2025-01-01',
).dropna()['Close']
df = np.log(df).diff().dropna() #obtain the derivative of the log of the data to give a measure of the rate of change of the assets prie
df.index = pd.to_datetime(df.index) #set index format to datetime such that we can use this format to select part of this dataframe later on
df = df.sort_index() #checking that data is in chronological order


window_size = 126 #setting the timeframe size on which our betas will be calculated
assets_to_regress_against = ["VDC", "VCR", "VIS", "VGT"]
rolling_betas = {"VDC":[], "VCR":[], "VIS":[], "VGT":[]} # Dictionary
rolling_dates = []

#calculating betas
for start in range(len(df) - window_size + 1):
    y_window = df['VFH'].iloc[start: start + window_size]
    x_window = df[assets_to_regress_against].iloc[start:start + window_size]
    model = LinearRegression().fit(x_window, y_window) #creating linear regression model with sklearn
    model.fit(x_window, y_window)

    beta_VDC, beta_VCR, beta_VIS, beta_VGT = model.coef_

    #appending retrieved data to the dictionary
    rolling_betas["VDC"].append(beta_VDC)
    rolling_betas["VCR"].append(beta_VCR)
    rolling_betas["VIS"].append(beta_VIS)
    rolling_betas["VGT"].append(beta_VGT)

    rolling_dates.append(df.index[start + window_size - 1])

betas = pd.DataFrame(rolling_betas, index=rolling_dates) #turning betas into dataframe and indexing it using our rolling date values
betas.columns = ['VDC_beta', 'VCR_beta', 'VIS_beta', 'VGT_beta']
df = df[df.index >= betas.index[0]]

#calculating the residuals employing the formula residuals = r_asset - sum beta_hedge*r_hedge
residuals = df['VFH']-(
        betas['VDC_beta']*df['VDC']+betas['VCR_beta']*df['VCR']+betas['VIS_beta']*df['VIS']+betas['VGT_beta']*df['VGT'])
residuals = pd.DataFrame(residuals, index=rolling_dates)
residuals['z'] = residuals - residuals.mean()/residuals.std() #storing the z_score of the data in a Dataframe

#we are now going to employ the data up to 2024-01-01 as our training set, we will then use the rest of our dataset to verify the applicability of our theory, this will show whether it is possible to use our fixed z thresholds to actually generate profits or whether we would have to recalculate these every day after market close
#turning pandas dataframes into np array as this allows for significantly quicker manipulation, not exactly relevant for this usecase as we are employing optuna but does nonetheless speed execution up significantly
vfh = df.loc[:'2024-01-01', 'VFH'].dropna().to_numpy()
vdc = df.loc[:'2024-01-01', 'VDC'].dropna().to_numpy()
vcr = df.loc[:'2024-01-01', 'VCR'].dropna().to_numpy()
vis = df.loc[:'2024-01-01', 'VIS'].dropna().to_numpy()
vgt = df.loc[:'2024-01-01', 'VGT'].dropna().to_numpy()

#shifting our beta values by one day, we are dealing with market close prices and can only react on the following day, thus if we just employed the betas of the same day for our algorithm we will have look-ahead bias
betas_exec = betas.shift(1).reindex(df.index).ffill().bfill()
b_vdc = betas_exec.loc[:'2024-01-01', 'VDC_beta'].to_numpy()
b_vcr = betas_exec.loc[:'2024-01-01', 'VCR_beta'].to_numpy()
b_vis = betas_exec.loc[:'2024-01-01', 'VIS_beta'].to_numpy()
b_vgt = betas_exec.loc[:'2024-01-01', 'VGT_beta'].to_numpy()

#recalculating residual and storing it as s
s = vfh - (b_vdc*vdc + b_vcr*vcr + b_vis*vis + b_vgt*vgt)

#transforming index of residuals dataframe into datetime format
residuals.index = pd.to_datetime(residuals.index)
residuals = residuals.sort_index()

z = np.ascontiguousarray(residuals.loc[:'2024-01-01', 'z'].to_numpy(), dtype=np.float64)
s = np.ascontiguousarray(s, dtype=np.float64)
root_252 = np.sqrt(252)

#defining a function which calculates the sharpe ratio of this strategy given 4 parameters for the bounds, the lag parameter simply aims to fix look-ahead bias
def get_sharpe(z1, z2, z3, z4, lag=1):
    n = z.size #number of time steps
    state = 0 #the trading state that we decide today {-1, 0, 1}
    q = np.zeros(lag, np.int8) #FIFO queue holding past values to apply in the future


    # We’ll compute the mean and variance ONLINE (streaming) using Welford’s algorithm.
    # 'count' = number of valid realised returns observed so far (skips NaNs).
    # 'mean'  = running mean of realised returns r.
    # 'M2'    = running sum of squared deviations from the mean: sum_{i=1..count} (r_i - mean)^2
    count = 0
    mean = 0.0
    M2 = 0

    for k in range(n):
        r = q[0] * s[k] #the realised return today will be equal to yesterday's position times today's raw return

        if r == r: #quick isNan check
            count += 1
            delta = r - mean #delta is difference between new sample and old mean
            mean += delta/count
            M2 += delta * (r - mean)

        #Shift lag queue left by one to 'age' the pending states
        for i in range(lag - 1):
            q[i] = q[i + 1]

        #if we are currently not in any position z1 and z3 will decide whether we will take on a new position
        if state == 0:
            if z[k] >= z1:
                state = 1
            elif z[k] <= z3:
                state = -1

        #if we are currently long z2 will decide when we exit
        elif state == 1:
            if z[k] <= z2:
                state = 0

        #if we are currently short z3 will decide when we exit
        elif state == -1:
            if z[k] >= z4:
                state = 0

        q[lag - 1] = state #adding this new state to the queue such that it can be applied on the next trading day

    #quick sanity check - returns -inf in case of error such that the algorithm picking the maximum sharpe isn't affected
    if count < 2:
        return -np.inf
    var = M2/(count - 1)
    if not var>0.0:
        return -np.inf
    return mean * root_252/np.sqrt(var)


#employing an optuna study to find the best parameters to optimise this strategy with respect to sharpe
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: get_sharpe(
    trial.suggest_float('z1', -3, 3), trial.suggest_float('z2', -3, 3), trial.suggest_float('z3', -3, 3),
    trial.suggest_float('z4', -3, 3)), n_trials=1000)


z1, z2, z3, z4 = (study.best_params for k in ('z1', 'z2', 'z3', 'z4')) #unpacking parameters from the dictionary containing the best variables
z = residuals.loc['2024-01-01':, 'z'].to_numpy()
pos = np.zeros(z.shape, dtype=int)
#updating the positions for these finalised z thresholds
for k in range(len(z)):
    if pos[k] == 0:
        if z[k] > z1: pos[k] = 1
        elif z[k] < z3: pos[k] = -1
    elif pos[k] == 1 and z[k] < z2:
        pos[k] = 0
    elif pos[k] == -1 and z[k] > z4:
        pos[k] = 0
#copying the df data to df_test such that we have only the relevant timestamps in our dataframe
df_test = df['2024-01-01':].copy()
df_test['position'] = pos
#prevents look-ahead bias
df_test['position_dec'] = df_test['position'].shift(1).fillna(0)
#calculating the residuals of the strategy
returns =  df_test['position_dec']*(
        df.loc['2024-01-01':, 'VFH'] - (
        betas_exec.loc['2024-01-01':, 'VDC_beta']*df.loc['2024-01-01':, 'VDC'] + betas_exec.loc['2024-01-01':, 'VCR_beta']*df.loc['2024-01-01':, 'VCR'] + betas_exec.loc['2024-01-01':, 'VIS_beta']*df.loc['2024-01-01':, 'VIS'] + betas_exec.loc['2024-01-01':, 'VGT_beta']*df.loc['2024-01-01':, 'VGT']))

#plotting results to visualise effectiveness of strategy
plt.plot(10000 * (1 + returns).cumprod())
df_test['VFH_returns'] = 10000 * (1 + df_test['VFH']).cumprod()
df_test['VFH_returns'].plot(color='black')
plt.ylabel('returns, (USD)')
plt.xlabel('Datetime')
plt.show()

###---Conclusion
'''
The obtained strategy does in fact not beat a simple Buy&Hold strategy, thus one way to fix this could be to update the 
optimal z thresholds on a rolling basis, however that could lead to our model overfitting to noise. It is possible that 
the strategy adopted here for educational purposes is simply too rudimentary and needs refinement. However it has 
successfully introduced me to the realm of machine learning but also most especially writing quick and efficient functions
this is especially useful for HFT applications which I hope to explore in the future.
'''


