
import datetime
import os
import pandas as pd
# for Bloomberg API
import pdblp
# Import file parameter
import parameter as para

'''
 1_Download Data from bbg
 by: Vincent 2018.08.24
 Monthly download data for asset allocation
 
  2_Download Data from bbg
 by: Brian 2018.11.7
 Rearrange the columns of downloaded data from Bloomberg
'''

# get the asset we want from para in
Index_assets = para.ASSETS_LIST
Asset_Start_Time = para.ASSET_START_TIME
ETF_Start_Time = para.ETF_START_TIME
etf_index_dict = para.ETF_ASSET_DICT
etf_option = para.ETF_OPTION
End_Time = datetime.datetime.today().strftime('%Y%m%d')
End_Time = End_Time[:6] + '01'
print("We should update monthly, at each starting month, suppose at 7am (UTC+8) on"
      " 2018-08-01  (which means data unavalable for that day at US Market)"
      "We should set the end_time as 20180801, or anydate after 2018-07-31 and "
      "use the default start time 20001225，which is longest available data time")

while True:
    End_Time_input = input(
        'Enter End date,blank for default {}:\t'.format(End_Time))
    if len(End_Time_input) == 8 or len(End_Time_input) == 0:
        break
    else:
        print("Try again, please enter the date in correct form: yyyymmdd.\n")

if len(End_Time_input) == 8:
    End_Time = End_Time_input
Fields = 'PX_LAST'


def rearrange_column(column_list, df):
    new_df = pd.DataFrame(index=[i.strftime('%m/%d/%Y') for i in df.index], columns=column_list)
    for i in column_list:
        new_df[i] = df[i]
    return new_df

# Start Connection
con = pdblp.BCon(debug=False, port=8194)
con.start()
Index_Data = con.bdh(Index_assets, Fields, Asset_Start_Time, End_Time)
Index_Data = rearrange_column(Index_assets,Index_Data)
root = os.getcwd()
Index_Data.to_csv(root + '/Index_Price_' + Asset_Start_Time + '_' + End_Time +
                  '.csv')
print(Index_Data.head())

# Get Benchmark Data
benchmark_assets = ['NDDUWI Index', 'LEGATRUU Index']
Benchmark_Data = con.bdh(benchmark_assets, Fields, Asset_Start_Time, End_Time)
Benchmark_Data = rearrange_column(benchmark_assets,Benchmark_Data)
Benchmark_Data.to_csv(root + '/Benchmark_Price_' + Asset_Start_Time + '_' +
                      End_Time + '.csv')
print(Benchmark_Data.head())

# Get ETF Data
if etf_option:
    ETFs = list(etf_index_dict.keys())
    ETFs.append('US0003M Index')  # libor , risk free rate
    ovrds = [('PX957', 1)]
    elms = [('currency', 'USD')]
    ETF_Data = con.bdh(ETFs, Fields, ETF_Start_Time, End_Time, elms=elms, ovrds=ovrds)
    ETF_Data.head()
    ETF_Data.rename(columns=etf_index_dict, inplace=True)
    ETF_Data_processed = ETF_Data.fillna(method='ffill')
    ETF_Data_processed.columns = ETF_Data_processed.columns.droplevel(1)
    ETF_Data_processed.to_csv(root + '/ETF_Price_' + ETF_Start_Time + '_' + End_Time + '.csv')
    print(ETF_Data_processed)

print("Finish Getting Data")
# End Connection
con.stop()
