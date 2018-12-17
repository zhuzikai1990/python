import pandas as pd
import numpy as np
import parameter as para


def get_annual_return_num(monthly_return, data_period):
    if data_period == 'month':
        year_num = len(monthly_return) / 12.0
    elif data_period == 'semimonth':
        year_num = len(monthly_return) / 24.0
    elif data_period == 'quarter':
        year_num = len(monthly_return) / 4.0
    elif data_period == 'year':
        year_num = len(monthly_return) / 1.0
    else:
        raise Exception("Please Enter: semimonth, month, quarter, year")
    monthly_cumreturn = (monthly_return+1).cumprod()
    AnnRet = monthly_cumreturn.iloc[-1]**(1 / year_num) - 1

    return round(AnnRet * 100, 1)


def get_annual_Std_num(monthly_return, data_period):
    if data_period == 'month': multiplier = 12
    elif data_period == 'semimonth': multiplier = 24
    elif data_period == 'quarter': multiplier = 4
    elif data_period == 'year': multiplier = 1
    else: raise Exception("Please Enter: semimonth, month, quarter, year")
    return round(
        (monthly_return.std() * np.sqrt(multiplier)) * 100, 1)


def get_annualized_sharpe_num(monthly_return, data_period):
    if data_period == 'month': multiplier = 12
    elif data_period == 'semimonth':multiplier = 24
    elif data_period == 'quarter': multiplier = 4
    elif data_period == 'year': multiplier = 1
    else: raise Exception("Please Enter: semimonth, month, quarter, year")
    return round(
        np.sqrt(multiplier) * monthly_return.mean() /
        monthly_return.std(), 2)


def get_MaxDrowndown_num(monthly_return):
    best = curr = 1
    curri = starti = besti = 0
    for ind, i in enumerate(monthly_return+1):
        try:
            if curr * i < 1:
                curr *= i
            else:
                curr, curri = 1, ind + 1
            if curr <= best:
                starti, besti, best = curri, ind + 1, curr
        except TypeError:
            continue
    max_sub_array = (monthly_return+1)[starti:besti]
    best = 1 - best
    return round(best,2)


def get_MaxDrowndown_Period_num(array):
    best = curr = 1
    curri = starti = besti = 0
    for ind, i in enumerate(array):
        try:
            if curr * i < 1:
                curr *= i
            else:
                curr, curri = 1, ind + 1
            if curr <= best:
                starti, besti, best = curri, ind, curr  # not sure..
        except TypeError:
            continue
    return array.index[starti], array.index[besti]


def get_excess_returns_num(cum_returns, benchmark_cum_returns):
    excess_return = cum_returns - benchmark_cum_returns
    return excess_return[-1]


def get_min_monthly_return(monthly_return_array):
    return monthly_return_array.min()


class Indicator():
    def __init__(self,
                 Monthly_Return_df,
                 P_Trade_Weights,
                 data_period='month'):
        self.monthly_return_df = Monthly_Return_df
        self.monthly_cumreturn_df = Monthly_Return_df.apply(lambda x: (x + 1).cumprod())
        self.trade_weights_df = P_Trade_Weights
        self.data_period = data_period

    # 1. Annual Return  = average monthly return * 12
    def get_annual_return_array(self):
        AnnRets = self.monthly_return_df.apply(
            get_annual_return_num, args=(self.data_period, ))
        return AnnRets

    def get_annual_Std_array(self):

        StDev = self.monthly_return_df.apply(
            get_annual_Std_num, args=(self.data_period, ))
        return StDev

    # 3. Sharpe Ratio
    def get_annualized_sharpe_array(self):
        Sharpe_Ratio = self.monthly_return_df.apply(
            get_annualized_sharpe_num, args=(self.data_period, ))
        return Sharpe_Ratio

    # 4. MaxDrown Down
    def get_MaxDrowndown_array(self):
        Max_DrownDown = self.monthly_return_df.apply(get_MaxDrowndown_num)
        return Max_DrownDown

    # 5. MaxDrowndown_period
    def get_MaxDrowndown_Period_array(self):
        MaxDrowndownPeriod = (
            self.monthly_return_df + 1).apply(get_MaxDrowndown_Period_num)
        MaxDrownDown_Start_Time = np.array([
            pd.Timestamp.strftime(Time[0], '%Y%m%d')
            for Time in MaxDrowndownPeriod
        ])
        MaxDrownDown_End_Time = np.array([
            pd.Timestamp.strftime(Time[1], '%Y%m%d')
            for Time in MaxDrowndownPeriod
        ])
        return MaxDrownDown_Start_Time, MaxDrownDown_End_Time

    # 6. Winning Ratio
    def get_wining_ratio(self, column_name):

        win_ratio = round(
            sum((self.monthly_return_df.loc[:, column_name] > 0) * 1) /
            len(self.monthly_return_df.loc[:, column_name]) * 100, 2)
        print('Winning Ratio for Junson Portfolio is: {} %'.format(
            win_ratio))
        return win_ratio

    # 7. monthly_turnover
    def get_monthly_turnover(self):
        # Turnover is equal to the absolute value of weights change
        P_turnover_df = abs(self.trade_weights_df.diff())
        # Monthly average Turnover
        Monthly_Turnover = round(
            P_turnover_df.sum(axis=1).iloc[1:].mean() * 100, 1)
        if para.DATA_PERIOD == 'semimonth':
            Monthly_Turnover = Monthly_Turnover * 2
        print(Monthly_Turnover)
        return Monthly_Turnover

    # 8. Excess Return
    def get_excess_returns_array(self):
        benchmark_cum_returns = np.array(
            self.monthly_cumreturn_df.Benchmark)
        excess_returns = round(
            self.monthly_cumreturn_df.apply(
                get_excess_returns_num, args=(benchmark_cum_returns, )) *
            100, 1)
        return excess_returns

    # 9. Summary Chart
    def get_summary_chart(self):
        AnnRets = self.get_annual_return_array()
        StDev = self.get_annual_Std_array()
        Sharpe_Ratio = self.get_annualized_sharpe_array()
        Max_DrownDown = self.get_MaxDrowndown_array()
        excess_returns = self.get_excess_returns_array()
        Monthly_TurnOver = self.get_monthly_turnover()
        MaxDrownDown_Start_Time, MaxDrownDown_End_Time = self.get_MaxDrowndown_Period_array(
        )
        Performance = pd.DataFrame({
            'Annual Return(%)':
            AnnRets,
            'Standard Deviation(%)':
            StDev,
            'Sharpe Ratio':
            Sharpe_Ratio,
            'MaxDrowndown(%)':
            Max_DrownDown*100,
            'Excess Returns(%)':
            excess_returns,
            'Monthly Turnover(%)':
            Monthly_TurnOver,
            'MaxDrowndown_Start':
            MaxDrownDown_Start_Time,
            'MaxDrownDown_End':
            MaxDrownDown_End_Time
        })

        return Performance
    def get_annual_results(self):
        monthly_return         = self.monthly_return_df.copy()
        monthly_return['year'] = monthly_return.index.year
        year_monthly_return_df = monthly_return.groupby('year')
        annual_return_year     = year_monthly_return_df.apply(lambda x : get_annual_return_num(x,self.data_period))
        annual_return_year.drop('year',axis =1 ,inplace =True)
        annual_sharpe_year     = year_monthly_return_df.apply(lambda x : get_annualized_sharpe_num(x,self.data_period))
        annual_sharpe_year.drop('year',axis =1 ,inplace =True)
        annual_vol_year        = year_monthly_return_df.apply(lambda x : get_annual_Std_num(x,self.data_period))
        #annual_drawdown_year   = year_monthly_return_df.apply(lambda x : get_MaxDrowndown_num(x))
        annual_vol_year.drop('year',axis =1 ,inplace =True)
        annual_MinReturn_year  = year_monthly_return_df.apply(lambda x : x.min())
        annual_MinReturn_year.drop('year',axis =1 ,inplace =True)

        def rename_col(columns, name):
            return [(name + '_' + col) for col in columns]

        annual_return_year.columns    = rename_col(annual_return_year.columns, 'return')
        annual_sharpe_year.columns    = rename_col(annual_sharpe_year.columns, 'sharpe')
        annual_vol_year.columns       = rename_col(annual_vol_year.columns, 'vol')
        #annual_drawdown_year.columns  = rename_col(annual_drawdown_year.columns, 'drawdown')
        annual_MinReturn_year.columns = rename_col(annual_MinReturn_year.columns, 'min_return')

        whole_annual_results = pd.concat([ annual_return_year,annual_sharpe_year,annual_vol_year,annual_MinReturn_year],axis = 1)

        return whole_annual_results

