'''
    # Asset Allocation Model 
    1. 2018.06.08 16:15
    **Vincent Wang** : Build 1 to 7 part.
    2. 2018.06.22
    **Vincent Wang** : Build Data importing from bbg
    3. 2018.07.10 10:08
    **Vincent Wang & Martin** : 
        - in  Data importing part, we **drop rows** before 2000.12.30 as 6 out of 18 asset do not have corresponding price data. 
        - after 2000.12.30 we **drop** weekends data
        - after 2000.12.30 we use **Linear interpolation**(or **back fill**) to handle the situation that in weekends and holidays some securities have price data while some do not have.
    4. 2018.07.15 
    **Vincent ** 
      - transfer into script version 

    5. 2018.07.16 
    **Vincent **
      - change upper constraint
      - change return calculation method 
      -   
    6. 2018.07.19
    **Vincent**
      - change group constraint value
      - change asset constraint value
      - change previous column mismatch problem 
      - add equity name for easier understanding
    7. 2018.07.25
    **Vincent**
      - add ETF data 
      - change read csv file method 
    8. 2018.07.31
    **Vincent**
      - add asset constraint option
      - remove lower constarint for each asset 
      - link interface to ppt 
    9. 2018.08.01 
    **Vincent**
      - get backtesting result for new month
      - add comments for optimization and backtesting part for date clear
      - improve checker function to check whether input date is correct or not
      - improve the robustness of backtesting part

    10. 2018.08.14 
    **Vincent""
      - correlation representation improvement 

    11. 2018.08.24
    **Vincent**
      - button up checking and add more information towards the result part
    12. 2018.08.28
    **Vincent**
      - add performance analysis data, annual return, vol, maxdrawdown etc
    13. 2018.08.29 
    **Vincent**
      - add Parameter.py for parameter saving
    14. 2018.09.01
    **Vincent**
      - modify parameter.py for better parameter entering process
'''

import numpy as np
import pandas as pd

# Mathematic constraint
from numpy import matrix, power
from math import sqrt

# optimization
import scipy.optimize as sco

# deltatime calculation
import datetime
from datetime import date
import time

# Plot Package
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# import indicator
from indicator import Indicator

# files checking
import os
import glob

os.chdir(os.path.dirname(__file__))
root = os.getcwd()

# Import parameters
import parameter as para

# Asset information
assets = para.ASSETS_LIST
asset_easier_dict = para.ASSETS_NAME_DICT
assets_code_new = np.array(list(asset_easier_dict.keys()))
assets_name_new = np.array(list(asset_easier_dict.values()))

# 6. asset allocation constraints (min_allocation, max_allocation)
asset_constraints_dict = para.ASSET_CONSTRAINT_DICT

# group constraint
assets_group_dict = para.GROUP_ID_DICT
growth_id = assets_group_dict.get('GROWTH')
defense_id = assets_group_dict.get('DEFENSE')
stable_id = assets_group_dict.get('STABLE')
inflation_id = assets_group_dict.get('INFLATION')
cash_id = assets_group_dict.get('CASH')

group_ids = [growth_id, defense_id, inflation_id, stable_id, cash_id]

asset_name_catelog = {
    'Growth': assets_name_new[growth_id],
    'Defense': assets_name_new[defense_id],
    'Stable': assets_name_new[stable_id],
    'Inflation': assets_name_new[inflation_id],
    'Cash': assets_name_new[cash_id],
}

constraint_growth = para.GROUP_CONSTRAINT_DICT.get('GROWTH')
constraint_defense = para.GROUP_CONSTRAINT_DICT.get('DEFENSE')
constraint_stable = para.GROUP_CONSTRAINT_DICT.get('STABLE')
constraint_inflation = para.GROUP_CONSTRAINT_DICT.get('INFLATION')
constraint_cash = para.GROUP_CONSTRAINT_DICT.get('CASH')

constraint_groups = [
    constraint_growth, constraint_defense, constraint_inflation,
    constraint_stable, constraint_cash
]

# transaction cost
P_transaction_dict = para.ASSET_TRANSACTION_DICT

file_name = ''
result_file = ''


def main():
    global file_name
    global result_file
    result_file = "Result_1113"
    if os.path.exists(result_file):
        print('File Exists')
    else:
        os.mkdir(result_file)

    metrics = ['return','sharpe']
    Optimization_Lens = [1,2]
    P_Starting_Time = '2004-4'
    P_Ending_Time = '2018-12'
    option_assets = [0,1]
    option_groups = [1]
    option_deltas = [1]
    delta_ratios = [0.05,0.1]
    #GROUP CONSTRAINT
    growth_ratios = [0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    defence_ratios = [0.5,0.6,0.7,0.8,0.9]


    option_EWMAs = [1]
    EWMA_values = [
        0.9943
    ]
    NaN_methods = ['BF']
    AssetPrice_Data = 'Index_Price_20001225_20181207.csv'
    Benchmark_Data = 'Benchmark_Price_20001225_20181207.csv'

    def run_it(metric, Optimization_Len, P_Starting_Time, P_Ending_Time,
               option_asset, option_group, option_delta, delta_ratio,
               option_EWMA,EWMA_value,NaN_method,growth_ratio,defence_ratio, AssetPrice_Data, Benchmark_Data):
        optimization(metric, Optimization_Len, P_Starting_Time, P_Ending_Time,
                     option_asset, option_group, option_delta, delta_ratio,
                     option_EWMA, EWMA_value,NaN_method,growth_ratio, defence_ratio, asset_price_path=AssetPrice_Data, benchmark_price_path=Benchmark_Data)
        backtest(metric, Optimization_Len, P_Starting_Time, P_Ending_Time,
                 option_asset, option_group, option_delta, delta_ratio,
                 option_EWMA, EWMA_value,NaN_method,growth_ratio, defence_ratio, asset_price_path=AssetPrice_Data, benchmark_price_path=Benchmark_Data)

    for Optimization_Len in Optimization_Lens:
        # Using the below to dynamically change the ratio
        year = int(P_Starting_Time.split('-')[0])
        month = P_Starting_Time.split('-')[1]
        P_Starting_Year = year + Optimization_Len
        P_Starting_Time = str(P_Starting_Year) + '-' + month
        for metric in metrics:
            for option_asset in option_assets:
                for option_group in option_groups:
                    for option_delta in option_deltas:
                        for option_EWMA in option_EWMAs:
                            for growth_ratio in growth_ratios:
                                for defence_ratio in defence_ratios:
                                    for EWMA_value in EWMA_values:
                                        for delta_ratio in delta_ratios:
                                            file_name = get_file_name(
                                                metric, Optimization_Len,
                                                P_Starting_Time, P_Ending_Time,
                                                option_asset, option_group,
                                                option_delta, delta_ratio,
                                                option_EWMA, EWMA_value,NaN_methods[0],growth_ratio,defence_ratio)
                                            if os.path.exists(result_file + '/' + file_name):
                                                print('File Exists')
                                            else:
                                                os.mkdir(result_file + '/' + file_name)
                                            run_it(metric, Optimization_Len,
                                                   P_Starting_Time,
                                                   P_Ending_Time, option_asset,
                                                   option_group, option_delta,
                                                   delta_ratio, option_EWMA, EWMA_value,NaN_methods[0],growth_ratio, defence_ratio,
                                                   AssetPrice_Data, Benchmark_Data)
    print('Finished All !')


def calcPortfolioPerf(weights, meanReturns, covMatrix):
    '''
    Calculates the expected mean of returns and volatility for a portolio of
    assets, each carrying the weight specified by weights

    INPUT
    weights: array specifying the weight of each asset in the portfolio
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio

    OUTPUT
    tuple containing the portfolio return and volatility
    '''
    # Calculate return and variance

    portReturn = np.sum(meanReturns * weights)
    portStdDev = np.sqrt(np.dot(weights.T, np.dot(covMatrix,
                                                  weights))) * sqrt(250)

    return portReturn, portStdDev


def negSharpeRatio(weights, meanReturns, covMatrix, riskFreeRate):
    '''
    Returns the negated Sharpe Ratio for the speicified portfolio of assets

    INPUT
    weights: array specifying the weight of each asset in the portfolio
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio
    riskFreeRate: time value of money

    OUTPUT
    Sharpe Ratio of this portfolio
    '''
    p_ret, p_std = calcPortfolioPerf(weights, meanReturns, covMatrix)

    return -(p_ret - riskFreeRate) / p_std


def negReturn(weights, meanReturns, covMatrix, riskFreeRate):
    '''
    Returns the negated Sharpe Ratio for the speicified portfolio of assets

    INPUT
    weights: array specifying the weight of each asset in the portfolio
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio
    riskFreeRate: time value of money

    OUTPUT
    Sharpe Ratio of this portfolio
    '''
    p_ret, p_std = calcPortfolioPerf(weights, meanReturns, covMatrix)

    return -p_ret


def getPortfolioStd(weights, meanReturns, covMatrix):
    '''
    Returns the volatility of the specified portfolio of assets

    INPUT
    weights: array specifying the weight of each asset in the portfolio
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio

    OUTPUT
    The portfolio's volatility
    '''
    return calcPortfolioPerf(weights, meanReturns, covMatrix)[1]


def findMaxMetricsPortfolio(metric,
                            meanReturns,
                            covMatrix,
                            riskFreeRate,
                            asset_bg_all,
                            asset_constraints,
                            pre_weights,
                            constraint_deltas,
                            option_group=1,
                            option_delta=1,
                            growth_ratio=0.5,
                            defense_ratio=0.7):
    '''
    Finds the portfolio of assets providing the maximum certain metrics 
    like Sharpe, Return, Variance 

    INPUT:
    --------------------------------------------
    metric: metric name : 'sharpe','std','return'
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio
    riskFreeRate: time value of money
    asset_bg_all : asset beginning weights
    asset_constraints: asset constraints ,list form containning each asset constraint tuple
    pre_weights : last optimization weight, using as a base for delta constraint
    constraint_deltas:  delta value for each asset
    option_group :  whether we have group constraint condition
    option_delta: whether we have delta constraint condition

    RETURN:
    ----------------------------------------------
    optimization output
	'''

    def constraints_all(x):
        return (np.sum(x) - 1)

    con_all = {'type': 'eq', 'fun': lambda x: constraints_all(x)}
    all_constraints = (con_all)
    bnds = tuple((asset_constraint) for asset_constraint in asset_constraints)

    if option_group != 0:
        # group match f(x) >= 0  (inequal side)
        def constraints_growth(x):
            growth_weights = np.sum(x[growth_id])
            return (growth_ratio - growth_weights)

        def constraints_defense(x):
            defense_weights = np.sum(x[defense_id])
            return (defense_ratio - defense_weights)

        def constraints_inflation(x):
            inflation_weights = np.sum(x[inflation_id])
            return (constraint_inflation - inflation_weights)

        def constraints_stable(x):
            stable_weights = np.sum(x[stable_id])
            return (constraint_stable - stable_weights)

        def constraints_cash(x):
            cash_weights = np.sum(x[cash_id])
            return (constraint_stable - cash_weights)

        con_growth = {'type': 'ineq', 'fun': lambda x: constraints_growth(x)}
        con_defense = {'type': 'ineq', 'fun': lambda x: constraints_defense(x)}
        con_inflation = {'type': 'ineq', 'fun': lambda x: constraints_inflation(x)}
        con_stable = {'type': 'ineq', 'fun': lambda x: constraints_stable(x)}
        con_cash = {'type': 'ineq', 'fun': lambda x: constraints_cash(x)}

        all_constraints = (con_all, con_growth, con_defense, con_inflation,
                           con_stable, con_cash)

    if (option_delta != 0) & (type(pre_weights) != int):
        # listable of scalar-output constraints input for SLSQP:
        # add delta constrain through bond scope
        delta_upper = pre_weights + constraint_deltas
        delta_lower = pre_weights - constraint_deltas
        upper_bnds = [
            min(delta_upper[i], asset_constraints[i][1])
            for i in range(len(meanReturns))
        ]
        lower_bnds = [
            max(delta_lower[i], asset_constraints[i][0])
            for i in range(len(meanReturns))
        ]

        bnds = tuple(
            (lower_bnds[i], upper_bnds[i]) for i in range(len(meanReturns)))

    args = (meanReturns, covMatrix, riskFreeRate)

    if metric == 'sharpe':
        opts = sco.minimize(
            negSharpeRatio,
            asset_bg_all,
            args=args,
            method='SLSQP',
            constraints=all_constraints,
            bounds=bnds,
            options={
                'disp': True,
                'maxiter': 1000
            })
    elif metric == 'return':
        opts = sco.minimize(
            negReturn,
            asset_bg_all,
            args=args,
            method='SLSQP',
            constraints=all_constraints,
            bounds=bnds,
            options={
                'disp': True,
                'maxiter': 1000
            })

    elif metric == 'std':
        opts = sco.minimize(
            getPortfolioStd,
            asset_bg_all,
            args=args,
            method='SLSQP',
            constraints=all_constraints,
            bounds=bnds,
            options={
                'disp': True,
                'maxiter': 1000
            })
    else:
        raise AssertionError("Not a correct metric")

    # Not 100% sure Sequential Least SQuares Programming(SLSQP)
    # can produce the same result , as in excel
    # we use Generalized Reduced Gradient(GRG) method,
    # but both of them can be used to solve nonlinear problems
    return opts


def make_correlation_plot(df_results, columns, title, save_name, file_name, fontsize=16, figsize=(14, 12)):
    assets_weights_corr_python = df_results[columns]
    assets_weights_corr_python = assets_weights_corr_python.astype(float).corr().round(2)
    colormap = plt.cm.RdBu
    plt.figure(figsize=(14, 12))
    mask = np.zeros_like(assets_weights_corr_python)
    mask[np.triu_indices_from(mask)] = True
    plt.title(title, y=1.05, size=15)
    font = {'size': fontsize}
    sns.heatmap(
        assets_weights_corr_python,
        mask=mask,
        linewidths=0.1,
        vmax=1.0,
        square=True,
        cmap=colormap,
        linecolor='white',
        annot=True,
        annot_kws={'fontdict': font}
    )
    plt.savefig(make_dir() + save_name + '.png')
    plt.close()


def make_plot(df_results, columns_to_plot, title, file_name):
    df_results[columns_to_plot].plot(figsize=(20, 10), linewidth=2, fontsize=20)
    plt.legend(
        columns_to_plot,
        fontsize=15,
        bbox_to_anchor=(0, 1.02, 1, .102),
        loc=3,
        ncol=6,
        borderaxespad=0)
    plt.title(title, fontsize=20)
    plt.xlabel('Time', fontsize=20)
    save_name = '_'.join(title.split(' '))
    plt.savefig(make_dir() + title + '.png')
    plt.close()


def make_dir():
    global result_file
    global file_name
    dir_str = './' + result_file + '/' + file_name + '/'
    return dir_str


def optimization(metrics='sharpe',
                 time_length='4',
                 t_s_d='2005-1',
                 t_e_d='2018-7',
                 option_asset_constraints=1,
                 option_group_constraints=1,
                 option_delta_constraints=1,
                 delta_value=0.3,
                 option_EWMA=1,
                 EWMA_value=0.9943,
                 NaN_method='BF',
                 growth_ratio=0.5,
                 defence_ratio=0.7,
                 asset_price_path='AssetPrice_19980101_20180710.csv',
                 benchmark_price_path='Benchmark_Data_19980101_20180710.csv',
                 For_Present=0):
    '''
    Core Part. Include the whole process about optimization

    INPUT:
    --------------------------------------------
    metric: metric name , 'sharpe','std','return'
    time_length: optimizatin look back period length, '1'
    t_s_d :  optimization start time, '2002-1'
    t_e_d :  optimization end time  , '2018-8'
    option_asset_constraints : whether we have asset constraint ? 1 or 0
    option_group_constraints : whether we have group constraint ? 1 or 0
    option_delta_constraints : whether we have delta constraint ? 1 or 0
    delta_value : delta value  value from 0-1
    option_EWMA : whether we apply EWMA process? 1 or 0
    EWMA_value  : EWMA value  value from 0-1
    NaN_method  : how we treat NaN value in holiday in asset price. Backfill(BF) or Linear Interpolation(LI) ?
    growth_ratio: growth group constraint value from 0.3-0.7
    defence_ratio: defence group constraint value from 0.5-1

    asset_price_path:  asset price , like  "Index_Price_20001225_20180801.csv"
    benchmark_price_path :  benchmark price, like "Benchmark_Price_20001225_20180801.csv"
    For_Present : for presentation reason, the text line on the each output picture will be empty. 1 or 0

    RETURN:
    ----------------------------------------------
    Nothing

    '''
    global file_name

    # 1. portfolio backtesting period  4 years
    Time_length = int(time_length)  # portfolio rolling window time length

    # 2. Portfolio Date,
    # 't' means text version,
    # with out t means datetime version
    t_start_date = str(int(t_s_d.split('-')[0]) -
                       Time_length) + '-' + t_s_d.split(
        '-')[1]

    t_end_date = t_e_d
    '''
     suppose the oldest data is 2014-01-01
     suppose the latest data is 2018-03-31
     we need to build our portfolio at the first day of the new month
     we use 4 years period to build(train) our portfolio,
     which means we will have 4 portfolio, building on 18-01-01 ,02-01, -03-01, -04-01
	'''
    # 1st Portfolio end date , 'p' means portfolio
    # starts from 2005
    t_p_start_date = t_start_date
    t_p_end_date = str(int(t_start_date.split('-')[0]) +
                       Time_length) + '-' + t_start_date.split('-')[1]

    # 'o' means orgin,which means that it will not be changed and will be used as
    # an reference to other date
    t_p_start_date_o = t_p_start_date
    t_p_end_date_o = t_p_end_date

    # Whole Data Perio
    start_date = pd.to_datetime(t_start_date)
    end_date = pd.to_datetime(t_end_date) - datetime.timedelta(days=1)

    # 3. portfolio general constraints,  1 means on, 0 means off
    # 4. EWMA parameter
    # exponentially weighted moving average (EWMA) estimation.
    # decay factor λ is generally assigned a value between .95 and .99.
    # Lower decay factors tend to weight recent data more heavily.
    EWMA_ratio = float(EWMA_value)  # ok

    constraint_asset_allocations = tuple(asset_constraints_dict.values())
    # 7. delta ratio constraints
    constraint_delta_ratio = float(delta_value)

    # delta constraints for each asset
    constraint_deltas = np.array([
        constraint_delta_ratio *
        (constraint_asset_allocation[1] - constraint_asset_allocation[0])
        for constraint_asset_allocation in constraint_asset_allocations
    ])

    asset_begin_allocation = np.array(
        [asset[0] + 0.001 for asset in constraint_asset_allocations])

    if int(option_asset_constraints) == 1:
        pass
    else:
        constraint_asset_allocations = tuple((0, 1) for asset in assets_name_new)

    # Read Data
    price_dir = str(asset_price_path)
    df_price_whole = pd.read_csv(price_dir, index_col=0, parse_dates=True)
    Datetime = np.array([
        pd.Timestamp.to_pydatetime(time_p)
        for time_p in df_price_whole.index
    ])
    Weekday = np.array([date.isoweekday(dt) for dt in Datetime])
    df_price_whole['week_day'] = Weekday
    # keep only with week day
    df_price_whole = df_price_whole[
        df_price_whole.week_day <= 5]

    # Process Holiday
    if str(NaN_method) == 'BF':
        df_price_whole.fillna(method='ffill', inplace=True)
    else:
        df_price_whole = df_price_whole.interpolate(
            method='linear')

    # Which means only first several rows has missing data, we can drop them and continue.
    df_price_whole.dropna(inplace=True)

    # risk free asset
    df_rf_return_whole = df_price_whole[['US0003M Index']] / 100
    df_price_whole.drop('US0003M Index', axis=1, inplace=True)
    df_price_whole.drop('week_day', axis=1, inplace=True)

    # Transform Price into Return (Simplest Way)
    df_return_whole = df_price_whole.pct_change().dropna(axis=0)
    # rearange the data
    global assets_code_new
    assets_code_new = assets_code_new.tolist()
    cash_place = assets_code_new.index('US0003M Index')
    assets_code_new.remove('US0003M Index')

    df_return_whole = df_return_whole[assets_code_new]
    assets_code_new.insert(cash_place, 'US0003M Index')
    assets_code_new = np.array(assets_code_new)
    df_return_whole = pd.concat([df_return_whole, (1+df_rf_return_whole) **(1/12/21) -1], axis=1)

    df_return_whole.dropna(inplace=True)
    #df_return_whole = df_return_whole[assets_code_new]
    df_return_whole.columns  = assets_name_new

    make_correlation_plot(df_results=df_return_whole, columns=assets_name_new, title='Correlation of Asset Return',
                          save_name='Return_Corr', file_name=file_name, fontsize=10, figsize=(14, 12))

    '''
	## 5.Optimization Part
	1. Seperate Asset and  Risk Free Rate
	2. EWMA Process
	3. Optimization
	4. Save Results (Weight, Past_Return, Past_Std, Past_Sharpe_Ratio)
	'''
    # get number of portfolio, based on the end of 1st portfolio and end of data
    num_portfolio = len(pd.date_range(pd.to_datetime(t_p_end_date), end_date, freq='m')) + 1

    def get_opt_weights_one_time(df_return_whole, df_rf_return_whole, p_start_date, p_end_date, pre_weights):
        # dataframe of return in that selected periods
        df_return_p = df_return_whole.loc[p_start_date:p_end_date, :].copy()
        # get risk free column and drop it from asset return part
        df_rf_return = df_rf_return_whole.loc[p_start_date:p_end_date, :].copy()
        # regard the last Libor rate as risk free rate for each period
        rf_rate = df_rf_return.iloc[-1,][0]  # regard the closest one as risk free rate
        dates_len, assets_len = df_return_p.shape

        print('We include {} trading days'.format(dates_len))
        print('We have {} assets'.format(assets_len))
        # Using EWMA method to treat return data
        matrix_var_cor = np.cov(df_return_p.values.T)
        df_mean_return = df_return_p.mean() * 250
        idx = pd.IndexSlice
        if int(option_EWMA) == 1:
            alpha = 1 - EWMA_ratio
            df_return_p_ewm = df_return_p.ewm(alpha=alpha)
            e_date = df_return_p.index[-1]
            df_mean_return = df_return_p_ewm.mean().loc[e_date, :] * 250
            matrix_var_cor = df_return_p_ewm.cov().loc[idx[e_date, :], :].values
        # Find portfolio with maximum Sharpe ratio
        # pay attention here is different from UI.py
        maxMetrics = findMaxMetricsPortfolio(
            metrics, df_mean_return.values, matrix_var_cor, rf_rate,
            asset_begin_allocation, constraint_asset_allocations, pre_weights,
            constraint_deltas, int(option_group_constraints),
            int(option_delta_constraints),growth_ratio,defence_ratio)
        # get results
        curr_weights = maxMetrics.x
        return_p, std_p = calcPortfolioPerf(maxMetrics['x'], df_mean_return,
                                            matrix_var_cor)
        Sharpe_p = (return_p - rf_rate) / std_p
        result = np.append(curr_weights, [return_p, std_p, Sharpe_p])
        return result, curr_weights

    def get_opt_weights(df_return_whole, df_rf_return_whole, num_portfolio, t_p_start_date, t_p_end_date):
        pre_weights = 0
        results = []
        dates = []
        last_date = df_return_whole.index[-1]
        for num in range(num_portfolio):
            # portfolio begin and end date
            p_start_date = pd.to_datetime(t_p_start_date)
            p_end_date = pd.to_datetime(t_p_end_date) - datetime.timedelta(days=1)

            # 每个月运行两次，一次是一号，一次是十五号
            # 第一次
            result, curr_weights = get_opt_weights_one_time(
                df_return_whole, df_rf_return_whole, p_start_date, p_end_date, pre_weights)
            # save results   1 row = weights + return   + std + Sharpe_p
            # with 21 columns
            results.append(result)
            dates.append(pd.to_datetime(t_p_end_date))
            pre_weights = curr_weights

            if para.DATA_PERIOD == 'semimonth':
                # 第二次
                p_start_date = p_start_date + datetime.timedelta(days=15)
                p_end_date = p_end_date + datetime.timedelta(days=15)
                if p_end_date <= last_date:
                    result, curr_weights = get_opt_weights_one_time(
                        df_return_whole, df_rf_return_whole, p_start_date, p_end_date, pre_weights)
                    results.append(result)
                    dates.append(p_end_date)
                    pre_weights = curr_weights

            print('Finish Optimization from {} to {}'.format(t_p_start_date, t_p_end_date))
            months = int(t_start_date.split('-')[1]) + num
            t_p_start_date = str(
                int(int(t_p_start_date_o.split('-')[0]) +
                    months / 12)) + '-' + str(1 + months % 12)
            t_p_end_date = str(
                int(int(t_p_end_date_o.split('-')[0]) +
                    months / 12)) + '-' + str(1 + months % 12)
            print('Get Weight at date {}'.format(t_p_end_date))
        return results, dates

    results, dates = get_opt_weights(df_return_whole, df_rf_return_whole, num_portfolio, t_p_start_date, t_p_end_date)

    # Save whole result
    def make_result_df(results, dates):

        columns = np.append(assets_name_new, ['Return_p', 'Std_p', 'Sharpe_p'])
        df_results = pd.DataFrame(results, columns=columns, index=dates)

        return df_results

    df_results = make_result_df(results, dates)
    weights_output_dir = 'Opt_Weights.csv'
    df_results.to_csv(make_dir() + weights_output_dir)

    make_plot(df_results, assets_name_new, 'Assets Weights', file_name=file_name, )

    df_results[assets_name_new].plot(subplots=True, figsize=(20, 15))
    plt.savefig(make_dir() + 'Asset_Weights_Specific.png')

    # add 1 year feature
    one_year_ago_time = datetime.date.today() - datetime.timedelta(365)
    df_results_1_year = df_results.loc[one_year_ago_time:, :].copy()
    make_plot(df_results_1_year, assets_name_new, 'Assets Weights 1y', file_name=file_name, )

    df_results_1_year[assets_name_new].plot(subplots=True, figsize=(20, 15))
    plt.savefig(make_dir() + 'Asset_Weights_Specific_1_y.png')
    make_correlation_plot(df_results=df_results, columns=assets_name_new, title='Correlation of Asset Weights',
                          save_name='Weights_Corr', file_name=file_name, fontsize=10, figsize=(14, 12))

    # group change visualizaiton
    if para.OPTION_GROUP:

        asset_types = list(asset_name_catelog.keys())
        for i in asset_types:
            asset_group = asset_name_catelog[i]
            df_results[i] = np.sum(df_results[asset_group], axis=1)
            df_results_1_year[i] = np.sum(df_results_1_year[asset_group], axis=1)

        make_plot(df_results, asset_types, 'Assets Group Weights', file_name=file_name, )

        make_plot(df_results_1_year, asset_types, 'Assets Group Weights 1y', file_name=file_name, )

        make_correlation_plot(df_results=df_results, columns=asset_types, title='Correlation of Asset Group Weights',
                              save_name='Group_Weights_Corr', file_name=file_name, fontsize=20, figsize=(14, 12))
        '''
        # add 2005 year feature 
        df_results_2005_year = df_results.loc['2005-01-01':'2006-01-01',:].copy()
        make_plot(df_results_2005_year, assets_name_new, 'Assets Weights 2005',file_name = file_name,)
        df_results_2005_year[assets_name_new].plot(subplots=True, figsize=(20, 15))
        plt.savefig(make_dir() + 'Asset_Weights_Specific_2005.png')
        make_plot(df_results_2005_year, asset_types, 'Assets Group Weights 2005',file_name = file_name,)
        '''


def backtest(metrics='sharpe',
             time_length=4,
             t_s_d='2005-1',
             t_e_d='2018-7',
             option_asset_constraints=1,
             option_group_constraints=1,
             option_delta_constraints=1,
             delta_value=0.3,
             option_EWMA=1,
             EWMA_value=0.9943,
             NaN_method='BF',
             growth_ratio=0.5,
             defense_ratio=0.7,
             asset_price_path='AssetPrice_19980101_20180710.csv',
             benchmark_price_path='Benchmark_Data_19980101_20180710.csv',
             For_Present=0):
    '''
    Core Part. Include the whole process about backtesting

    INPUT:
    --------------------------------------------
    metric: metric name , 'sharpe','std','return'
    time_length: optimizatin look back period length, '1'
    t_s_d :  optimization start time, '2002-1'
    t_e_d :  optimization end time  , '2018-8'
    option_asset_constraints : whether we have asset constraint ? 1 or 0
    option_group_constraints : whether we have group constraint ? 1 or 0
    option_delta_constraints : whether we have delta constraint ? 1 or 0
    delta_value : delta value  value from 0-1
    option_EWMA : whether we apply EWMA process? 1 or 0
    EWMA_value  : EWMA value  value from 0-1
    NaN_method  : how we treat NaN value in holiday in asset price. Backfill(BF) or Linear Interpolation(LI) ?

    asset_price_path:  asset price , like  "Index_Price_20001225_20180801.csv"
    benchmark_price_path :  benchmark price, like "Benchmark_Price_20001225_20180801.csv"
    For_Present : for presentation reason, the text line on the each output picture will be empty. 1 or 0

    RETURN:
    ----------------------------------------------
    Nothing

    '''
    global file_name

    # BackTesting Part
    weights_input_dir = 'Opt_Weights.csv'
    P_Weights = pd.read_csv(
        make_dir() + weights_input_dir,
        index_col=0,
        parse_dates=True)

    # Corresponding Tickers
    Tickers = np.array(P_Weights.columns[:-3])

    # Get Trading Weights
    P_Trade_Weights = P_Weights.loc[:, Tickers]
    p_e_d = t_e_d
    #when data is updated semimonth, data of first part of the month need to be recorded
    if para.DATA_PERIOD == 'semimonth' and P_Trade_Weights.index[-1] != pd.to_datetime(p_e_d):
        P_Trade_Weights = P_Trade_Weights[P_Trade_Weights.index <= pd.to_datetime(p_e_d)]
    else:
        P_Trade_Weights = P_Trade_Weights[P_Trade_Weights.index < pd.to_datetime(p_e_d)]

    def get_assets_return_data(asset_price_path,dates):
        global assets_code_new
        # Asset Price data
        price_input = str(asset_price_path)
        df_price_whole = pd.read_csv(price_input, index_col=[0], parse_dates=True)
        # datetime transform
        Datetime = np.array([
            pd.Timestamp.to_pydatetime(time_p)
            for time_p in df_price_whole.index
        ])
        # weekday checking --- for deleting weekends
        Weekday = np.array([date.isoweekday(dt) for dt in Datetime])
        df_price_whole['week_day'] = Weekday
        # keep only with week day
        df_price_whole = df_price_whole[
            df_price_whole.week_day <= 5]
        # Holiday -- back fill
        df_price_whole.fillna(method='ffill', inplace=True)

        # 留下dates中的日子，存在df_price_keeped中
        day_num =0
        df_price_keeped = pd.DataFrame(columns=df_price_whole.columns )
        current_date = dates[0]
        # 要计算月底最后一天，而不是月初第一天之间的收益率
        for i in range(len(df_price_whole)):
            if current_date < df_price_whole.index[i]:
                day_num = day_num+1
                if day_num<len(dates):
                    current_date = dates[day_num]
            elif current_date == df_price_whole.index[i]:
                df_price_keeped.loc[current_date] = df_price_whole.iloc[i-1]
            else:
                df_price_keeped.loc[current_date] = df_price_whole.iloc[i]

        # risk free asset
        df_rf_return_whole = df_price_keeped[['US0003M Index']] / 100
        df_price_keeped.drop('US0003M Index', axis=1, inplace=True)

        # Transform Price into Return (Simplest Way)
        monthly_return_data = df_price_keeped.pct_change().dropna(axis=0, how='all')
        #收益率向前移动一格
        monthly_return_data = monthly_return_data.set_index(dates[:-1])

        # rearange the data
        global assets_code_new
        assets_code_new = assets_code_new.tolist()
        cash_place = assets_code_new.index('US0003M Index')
        assets_code_new.remove('US0003M Index')

        monthly_return_data = monthly_return_data[assets_code_new]
        assets_code_new.insert(cash_place, 'US0003M Index')
        assets_code_new = np.array(assets_code_new)
        # if data is updated monthly, date_factor ==1
        if para.DATA_PERIOD == 'semimonth':
            date_factor = 2
        else:
            date_factor = 1
        monthly_return_data = pd.concat([monthly_return_data, df_rf_return_whole / 12/date_factor], join='inner', axis=1)
        monthly_return_data = monthly_return_data[assets_code_new]

        return monthly_return_data

    # Only keep data at the beginning of each month
    # change to hardcoded one  , begin from 2015
    monthly_return_data_finished = get_assets_return_data(asset_price_path,P_Weights.index)

    # Check whether their time lenght is same
    print(P_Trade_Weights.shape)
    print(monthly_return_data_finished.shape)
    assert P_Trade_Weights.shape == monthly_return_data_finished.shape
    '''
    # Sort them , to make their column match
    P_Trade_Weights.sort_index(axis=1, inplace=True)
    monthly_return_data_finished.sort_index(axis=1, level=0, inplace=True)
    '''

    ## 2.1 Return
    P_Monthly_Return = np.array([])
    # Number of rebalance time
    num_trades = P_Trade_Weights.shape[0]
    for i in range(num_trades):
        # Current month asset weights
        P_Weights_curr = P_Trade_Weights.iloc[i, :].values
        # current month asset return
        P_Monthly_Return_curr = monthly_return_data_finished.iloc[i, :]
        if i == 0:
            # portfolio return = asset weights *  asset return
            P_Monthly_Return = P_Monthly_Return_curr * P_Weights_curr
        else:
            P_Monthly_Return = np.vstack(
                [P_Monthly_Return, P_Monthly_Return_curr * P_Weights_curr])

    P_Monthly_Return_df = pd.DataFrame(
        P_Monthly_Return,
        columns=P_Trade_Weights.columns,
        index=P_Trade_Weights.index)

    P_Montly_asset_return_df = P_Monthly_Return_df.copy()
    P_Cum_asset_return_df = (P_Montly_asset_return_df + 1).cumprod()

    # Portfolio Monthly Return
    P_Monthly_Return_array = P_Monthly_Return_df.sum(axis=1)
    # Portfolio Cumulative Return
    P_Cum_Return = (P_Monthly_Return_array + 1).cumprod()

    P_Monthly_Return_df = pd.DataFrame(
        P_Monthly_Return_array, columns=['Return'])

    P_Monthly_Return_df['year'] = P_Monthly_Return_df.index.year

    P_Monthly_Return_df['CumReturn'] = np.log(P_Monthly_Return_df['Return'] +
                                              1)
    P_yearly_Return_df = pd.DataFrame(
        P_Monthly_Return_df.groupby('year')['CumReturn'].sum()) * 100

    ## 2.2 Turnover

    # Turnover is equal to the absolute value of weights change
    P_turnover_df = abs(P_Trade_Weights.diff())

    # about the first row, the initial weighting
    P_turnover_df.iloc[0, :] = P_Trade_Weights.iloc[0, :]
    # Monthly average Turnover
    Monthly_TurnOver = round(
        P_turnover_df.sum(axis=1).iloc[1:].mean() * 100, 1)

    if para.DATA_PERIOD == 'semimonth':
        Monthly_TurnOver = Monthly_TurnOver * 2

    ## 2.3 Transaction

    P_transaction_df = pd.DataFrame(P_transaction_dict, index=[0])

    # Portfolio Transaction cost
    P_Tran_df = P_turnover_df
    for i in range(P_Tran_df.shape[0]):
        P_Tran_df.iloc[
        i, :] = P_turnover_df.iloc[i, :].values * P_transaction_df.values

    Tran_array = P_Tran_df.sum(axis=1)

    P_Monthly_Return_after_tran = P_Monthly_Return_array - Tran_array

    P_Cum_Return_after_tran = (P_Monthly_Return_after_tran + 1).cumprod()

    Benchmark_dir = str(benchmark_price_path)
    Benchmark_data = pd.read_csv(Benchmark_dir, index_col=[0], parse_dates=True)
    Benchmark_data.fillna(method='ffill', inplace=True)

    def get_benchmark_return_data(raw_data,dates):
        Datetime = np.array(
            [pd.Timestamp.to_pydatetime(time_r) for time_r in raw_data.index])
        Weekday = np.array([date.isoweekday(dt) for dt in Datetime])
        raw_data['week_day'] = Weekday
        # keep only with week day
        raw_data = raw_data[raw_data.week_day <= 5]
        # 留下dates中的日子，存在df_price_keeped中
        day_num =0
        raw_data_keeped = pd.DataFrame(columns=raw_data.columns)
        current_date = dates[0]
        #要计算月底最后一天，而不是月初第一天之间的收益率
        for i in range(len(raw_data)):
            if current_date < raw_data.index[i]:
                day_num = day_num+1
                if day_num<len(dates):
                    current_date = dates[day_num]
            elif current_date == raw_data.index[i]:
                raw_data_keeped.loc[current_date] = raw_data.iloc[i-1]
            else:
                raw_data_keeped.loc[current_date] = raw_data.iloc[i]
        # Transform Price into Return (Simplest Way)
        monthly_return_data = raw_data_keeped.pct_change().dropna(how='all')
        monthly_return_data = monthly_return_data.set_index(dates[:-1])
        monthly_return_data = monthly_return_data.sort_index()
        monthly_return_data.drop(
            ['week_day'], axis=1, inplace=True)
        monthly_return_data.rename(
            columns={
                'NDDUWI Index': 'MSCI Developed',
                'LEGATRUU Index': 'Bond Aggregate'
            },
            inplace=True)
        monthly_return_data[
            'Benchmark'] = (monthly_return_data['MSCI Developed'] +
                            monthly_return_data['Bond Aggregate']) / 2

        return monthly_return_data

    Benchmark_Return_data = get_benchmark_return_data(Benchmark_data, P_Weights.index)
    Benchmark_Return_data.to_csv(make_dir() +
                                 'Benchmark_Return.csv')

    Whole_Monthly_Return_df = pd.read_csv(
        make_dir() + 'Benchmark_Return.csv',
        index_col=0, parse_dates=True)

    Whole_Monthly_Return_df = Whole_Monthly_Return_df.apply(
        pd.to_numeric, errors='ignore')
    Whole_Monthly_Return_df.rename_axis('date', 0, inplace=True)
    Whole_Monthly_Return_df[
        'Junson Portfolio'] = P_Monthly_Return_after_tran.values

    Whole_Monthly_Return_df_log = np.log(Whole_Monthly_Return_df + 1)
    # Can I use log here
    Whole_Monthly_Return_df_log[
        'year'] = Whole_Monthly_Return_df_log.index.year
    Whole_Monthly_Return_df_log = pd.DataFrame(
        Whole_Monthly_Return_df_log.groupby('year').sum())

    num_win = ((Whole_Monthly_Return_df_log['Junson Portfolio'] >
                Whole_Monthly_Return_df_log['Benchmark']) * 1).sum()
    pct_win = round(
        float(num_win) / Whole_Monthly_Return_df_log.shape[0] * 100, 1)

    num_pos = (
            (Whole_Monthly_Return_df_log['Junson Portfolio'] >= 0) * 1).sum()
    pct_pos = round(
        float(num_pos) / Whole_Monthly_Return_df_log.shape[0] * 100, 1)

    width = 0.16
    width1 = 0.08
    width2 = 0.24
    fig, ax = plt.subplots(figsize=(15, 9))
    b1 = ax.bar(
        Whole_Monthly_Return_df_log.index - width2,
        Whole_Monthly_Return_df_log['MSCI Developed'] * 100,
        width,
        color='royalblue')
    b2 = ax.bar(
        Whole_Monthly_Return_df_log.index - width1,
        Whole_Monthly_Return_df_log['Bond Aggregate'] * 100,
        width,
        color='grey')
    b3 = ax.bar(
        Whole_Monthly_Return_df_log.index + width1,
        Whole_Monthly_Return_df_log['Benchmark'] * 100,
        width,
        color='limegreen')
    b4 = ax.bar(
        Whole_Monthly_Return_df_log.index + width2,
        Whole_Monthly_Return_df_log['Junson Portfolio'] * 100,
        width,
        color='coral')
    for b in b4:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            y=h,
            s='%d' % int(h),
            ha='center',
            va='bottom',
            color='r')

    leftest = plt.xlim()[0] + 0.5
    lenghth = plt.ylim()[1] - plt.ylim()[0]
    multi = lenghth / 20
    topest = plt.ylim()[1] - multi
    if int(For_Present) == 0:
        plt.text(leftest, topest, file_name, fontsize=14, color='coral')
        plt.text(
            leftest,
            topest - multi,
            'Yearly return beat Benchmark {} times ({}%)'.format(
                num_win, pct_win),
            fontsize=14,
            color='coral')
    else:
        pass

    plt.ylabel('Yearly Return(%)', fontsize=14)
    plt.xticks(Whole_Monthly_Return_df_log.index, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Time', fontsize=14)
    plt.title('Junson Portfolio Yearly Return', fontsize=14)
    plt.legend(
        [
            'MSCI Developed', 'Bond Aggregate', '50 Stock + 50 Bond',
            'Junson Portfolio'
        ],
        fontsize=14,
        bbox_to_anchor=(-0.05, 1.05, 1, .1),
        loc=3,
        ncol=2,
        borderaxespad=0)
    plt.savefig(make_dir() +
                'BT_Yearly_Return.png')
    Whole_Monthly_Return_df_log.to_csv(make_dir() +
                                       'BT_Yearly_Return.csv')
    Whole_monthly_CumReturn_df = Whole_Monthly_Return_df.apply(lambda x: (x + 1).cumprod())
    # Whole_Monthly_Return_df.to_csv('Whole_Monthly_Return.csv')
    Junson_P_Indicator = Indicator(
        Whole_Monthly_Return_df,
        P_Trade_Weights,
        data_period=para.DATA_PERIOD)
    ## 7.Excess Return
    Excess_Return = Whole_monthly_CumReturn_df.copy()
    Excess_Return[
        'Excess_Return'] = Whole_monthly_CumReturn_df.loc[:,
                           'Junson Portfolio'] - Whole_monthly_CumReturn_df.loc[:,
                                                 'Benchmark']
    Excess_Return = Excess_Return[[
        'Excess_Return', 'Junson Portfolio', 'Benchmark'
    ]]
    Excess_Return.index = pd.to_datetime(Excess_Return.index)

    # Excess_Return.plot(figsize=(15,9), linewidth=2, fontsize=15)
    fig = plt.figure(figsize=(14, 7))

    ax1 = fig.add_subplot(111)
    ax1.plot(
        Excess_Return.index,
        Excess_Return.iloc[:, 1:2] * 100,
        'coral',
        ls='--')
    ax1.plot(Excess_Return.index, Excess_Return.iloc[:, 2:3] * 100,
             'limegreen')
    ax1.set_ylabel('Cumulative Return(%)', fontsize=14)
    ax1.legend(
        ['Junson Portfolio', 'Benchmark'],
        fontsize=16,
        bbox_to_anchor=(-0.05, 1.05, 1, .1),
        loc=3,
        ncol=2,
        borderaxespad=0)
    ax1.set_xlabel('Time', fontsize=16)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label1.set_fontsize(14)

    ax2 = ax1.twinx()
    ax2.plot(
        Excess_Return.index, Excess_Return.iloc[:, 0:1] * 100, 'r', alpha=0.8)
    ax2.set_ylabel('Excess Return(%)', color='r', fontsize=16)
    ax2.legend(
        ['Excess_Return'],
        fontsize=14,
        bbox_to_anchor=(0.8, 1.05, 1, .1),
        loc=3,
        ncol=2,
        borderaxespad=0)
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    ax2.set_xlabel('Time', fontsize=16)

    for tick in ax2.xaxis.get_major_ticks():
        tick.label2.set_fontsize(14)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label2.set_fontsize(14)

    plt.title('Monthly Excess Return', fontsize=16)
    plt.savefig(make_dir() +
                'BT_Excess_Return.png')

    Performance = Junson_P_Indicator.get_summary_chart()
    Annual_Results = Junson_P_Indicator.get_annual_results()
    print(Annual_Results)
    Annual_Results.to_csv(make_dir() +
                          'BT_Annual_Results.csv')
    Performance.to_csv(make_dir() +
                       'BT_Results.csv')

    # Visualize it
    J_P = Performance.T['Junson Portfolio']

    def get_certain_period_cum(Whole_monthly_CumReturn_df, start_time='1999-01-01',
                               end_time=date.today().strftime('%Y-%m-%d')):
        certain_period_cum_df = Whole_monthly_CumReturn_df.loc[start_time:end_time, :]
        certain_period_cum_df = certain_period_cum_df / certain_period_cum_df.iloc[0, :]
        return certain_period_cum_df

    #    Whole_monthly_CumReturn_df = get_certain_period_cum(Whole_monthly_CumReturn_df,start_time = '2005-01-01',end_time = '2006-01-01')

    fig = (Whole_monthly_CumReturn_df * 100).plot(
        figsize=(15, 9),
        linewidth=2,
        fontsize=15,
        color=['royalblue', 'grey', 'limegreen', 'coral'],
        style=['-', '-', '-', '--'])
    ax = fig
    leftest = plt.xlim()[0] + 5
    lenghth = plt.ylim()[1] - plt.ylim()[0]
    multi = lenghth / 20
    topest = plt.ylim()[1] - multi
    if int(For_Present) == 0:
        plt.text(leftest, topest, file_name, fontsize=14, color='coral')
        plt.text(
            leftest,
            topest - multi,
            'Junson Portfolio:',
            fontsize=14,
            color='coral')
        for i, (param, value) in enumerate(
                zip(J_P.index[:-2], J_P.values[:-2])):
            if '(%)' in param:
                param = param[:-3]
                plt.text(
                    leftest,
                    topest - multi * (i + 2),
                    '{}  :  {}'.format(param,
                                       str(value) + '%'),
                    fontsize=14,
                    color='coral')
                i_last = i
            else:
                plt.text(
                    leftest,
                    topest - multi * (i + 2),
                    '{}  :  {}'.format(param, round(value, 2)),
                    fontsize=14,
                    color='coral')
                i_last = i

        plt.text(
            leftest,
            topest - multi * (i_last + 3),
            'Yearly return beat Benchmark  {} times ({} % )'.format(
                num_win, pct_win),
            fontsize=14,
            color='coral')
        plt.text(
            leftest,
            topest - multi * (i_last + 4),
            'Yearly return positive return {} times ({} % )'.format(
                num_pos, pct_pos),
            fontsize=14,
            color='coral')
    else:
        pass
    '''
    dot_period = ('20000311','20021009')
    fin_period = ('20071201','20090601')
    chin_period = ('20150612','20160510')
    drawdown_periods = [ dot_period , fin_period , chin_period]
    colors = ['y','b','g']
    periods = ['.Com','Financial Crisis','China A Share']

    x = Whole_monthly_CumReturn_df.index
    fills = []
    for i in range(len(periods)):
        period = ((x <= pd.to_datetime(drawdown_periods[i][1])) & (x >= pd.to_datetime(drawdown_periods[i][0])))    
        p = plot_period(ax, x, period, 0, 1, color = colors[i])
        fills.append(p)
    '''
    plt.legend(
        labels=['MSCI Developed', 'Bond Aggregate', '50 Stock + 50 Bond', 'Junson Portfolio'],
        fontsize=14,
        bbox_to_anchor=(-0.05, 1.05, 1, .1),
        loc=3,
        ncol=2,
        borderaxespad=0)

    plt.ylabel('Cumulative Return(%)', fontsize=14)
    plt.xlabel('Time', fontsize=14)
    plt.title('Junson Portfolio Backtesting Result', fontsize=14)
    plt.savefig(make_dir() + 'BT_Cum_Return.png')
    plt.close()

    print(Performance)

    fig = P_Cum_asset_return_df.plot(
        figsize=(15, 9),
        linewidth=2,
        fontsize=15)
    ax = fig
    x = P_Cum_asset_return_df.index

    plt.legend(
        handles=fig.lines, labels=list(P_Cum_asset_return_df.columns),
        fontsize=14,
        bbox_to_anchor=(-0.05, 1.05, 1, .1),
        loc=3,
        ncol=2,
        borderaxespad=0)

    plt.title('Junson Portfolio Asset Backtesting Result', fontsize=14)
    plt.savefig(make_dir() + 'BT_Cum_Asset_Return.png')
    plt.close()

    # P_Cum_asset_return_df
    P_Cum_asset_return_df[assets_name_new].plot(subplots=True, figsize=(20, 15))
    plt.savefig(make_dir() + 'BT_Cum_Asset_Return_Specific.png')
    plt.close()


def get_file_name(metric,
                  Optimization_Len,
                  P_Starting_Time,
                  P_Ending_Time,
                  option_asset,
                  option_group,
                  option_delta,
                  delta_ratio,
                  option_EWMA,
                  EWMA_value=0.9943,
                  NaN_method='BF',
                  growth_ratio=0.5,
                  defence_ratio=0.7):
    if int(option_asset) == 1:
        str_asset = 'asset'
    else:
        str_asset = '0'

    if int(option_group) == 1:
        str_group = 'group' + '_' + str(growth_ratio) + '_'+str(defence_ratio)
    else:
        str_group = '0'

    if int(option_delta) == 1:
        str_delta = 'delta'
        str_delta_ratio = str(delta_ratio)
    else:
        str_delta = '0'
        str_delta_ratio = '0'

    if int(option_EWMA) == 1:
        str_EWMA = 'EWMA'
    else:
        str_EWMA = '0'

    if str(NaN_method) == 'BF':
        NaN_method = 'BF'
    else:
        NaN_method = 'LI'

    if str(str_EWMA) == 'EWMA':
        file_name = metric + '_' + str(
            Optimization_Len
        ) + '_' + P_Starting_Time + '_' + P_Ending_Time + '_' + str_asset + '_' + str_group + '_' + str_delta + str_delta_ratio + '_' + str_EWMA + str(
            EWMA_value) + '_' + NaN_method+'_'+para.DATA_PERIOD
    else:
        file_name = metric + '_' + str(
            Optimization_Len
        ) + '_' + P_Starting_Time + '_' + P_Ending_Time + '_' + str_asset + '_' + str_group + '_' + str_delta + str_delta_ratio + '_' + str_EWMA + str(
            0) + '_' + NaN_method+'_'+para.DATA_PERIOD

    return file_name


if __name__ == '__main__':
    main()
# 12mins