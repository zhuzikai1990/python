import numpy as np
# Setting Some Parameter for AssetAllocation Model


'''
 in data downloading part, the START_TIME you choose
 should be at least 5 days earlier than the real starting time 
 you wanna download, because in the later process, we will drop weekend
 data, in which case we may drop that important date so we
 cannot get a correct monthly return at beginning.

 e.g  if you want to test period 2002-07-01 to 2018-08-01 
 then you may want to choose 2002-06-30 as your starting date, 
 because in this case you can get a monthly return correctly 
 from 2002-06-30 to 2002-07-31. but that 2002-06-30 might be dropped 
 because of weekend (in practice, yes, as 2002-06-30 is Sunday). So we 
 need to choose 2002-06-28(Fri) as our starting date, in which case , 
 even 2002-06-29 and 2002-06-30 are dropped, we can still get a correct
 monthly return from 2002-06-28 to 2002-07-31  
'''
# Start Data Downloading time for asset allocation model
ASSET_START_TIME = '20001225'
ETF_START_TIME = '20121025'
# if you don't need ETF data , then don't worry about it.
'''
whether we have group constraint in our process,
because sometimes we may just want to only test some asset 
performance after optimization : like these three assets:

        'SPXT Index'    : 'S&P tot',
        'LBUSTRUU Index': 'US Agg',
        'LF98TRUU Index': 'US HY',
we no need to set group level for each of them, so we can set 
Option Group as False. then we no need to care about Group_ID.

Otherwise, if we choose OPTION_GROUP as True, we need to set the group 
id correctly with the sequence of our ASSET_LIST Part.
if True, we need to set the GROUP INFORMATION part correctly
if False, we don't need to care about that 
'''
OPTION_GROUP = True
'''
Whether we need ETF data or not for another optimization process
if True, the downloading process will download the ETF data and you need to 
input those asset index correctly
if False, no need to care about the choice in ETF Information part .
'''
ETF_OPTION = False
'''
The frequency of updating the opt weight 
'''
DATA_PERIOD ='semimonth'

# -------------------------------------------------ASSET INFORMATION----------------------------------------------------
# 1. Enter ASSETS_LIST: official bbg code
# 2. Enter ASSETS_NAME_DICT: dictionary with official bbg code as key and asset name we choose as value
# 3. Enter ASSET_CONSTRAINT_DICT: dictionary with asset name as key and (lower bound, upper bound) as value
# 4. Enter ASSET_TRANSACTION_DICT: dictionary with asset name as key and transaction cost as value

ASSETS_DICT = {
          # bbg code      #asset name   #weight cons #tran cost #group
        'SPTR Index'    : ('S&P500'      , (0, 0.5 )  , 0.001 , 'GROWTH' ),
        'SX5T Index'    : ('Euro'        , (0, 0.15)  , 0.001 , 'GROWTH' ),
        'TPXDDVD Index' : ('JPY'         , (0, 0.1 )  , 0.001 , 'GROWTH' ),
        'HSI 1 Index'   : ('HSI'         , (0, 0.15)  , 0.001 , 'GROWTH' ),
        'TUKXG Index'   : ('FTSE100'     , (0, 0.1 )  , 0.001 , 'GROWTH' ),
        'M1EF Index'    : ('EM'          , (0, 0.1 )  , 0.001 , 'GROWTH' ),
        #'LBUSTRUU Index': ('US Agg'      , (0, 0.1 )  , 0.003 , 'DEFENSE'),
        'LUATTRUU Index': ('US Try'      , (0, 0.5 )  , 0.0075, 'DEFENSE') ,
        'LETGTREU Index': ('Euro Agg Try', (0, 0.2 )  , 0.001 , 'DEFENSE'),
        'BATY0 Index'   : ('Aus Try'     , (0, 0.2 )  , 0.001 , 'DEFENSE') ,
        'EMUSTRUU Index': ('EM Agg'      , (0, 0.15)  , 0.005 , 'STABLE' ) ,
        'LF98TRUU Index': ('US HY'       , (0, 0.2 )  , 0.0075, 'STABLE' ),
        'LP01TREU Index': ('Euro HY'     , (0, 0.15)  , 0.0075, 'STABLE' ),
        'LUACTRUU Index': ('US Corp'     , (0, 0.15)  , 0.0025, 'STABLE' ),
        'LP06TREU Index': ('Euro Agg'    , (0, 0.15)  , 0.0025, 'STABLE' ),
        'BCIT1T Index'  : ('US Govt'     , (0, 0.07)  , 0.0015, 'INFLATION'),
        'BEIG1T Index'  : ('Euro Govt'   , (0, 0.07)  , 0.0015, 'INFLATION'),
        'SPGSCI Index'  : ('Commodity'   , (0, 0.07)  , 0.005 , 'INFLATION'),
        'US0003M Index' : ('Cash'        , (0, 0.1 )  ,     0 , 'CASH'   ),
    }


ASSETS_LIST            =  list(ASSETS_DICT.keys())
ASSETS_NAME_DICT       =  {asset:ASSETS_DICT.get(asset)[0] for asset in ASSETS_DICT.keys()}
ASSET_CONSTRAINT_DICT  =  {asset:ASSETS_DICT.get(asset)[1] for asset in ASSETS_DICT.keys()}
ASSET_TRANSACTION_DICT =  {asset:ASSETS_DICT.get(asset)[2] for asset in ASSETS_DICT.keys()}
# --------------------------------------------------GROUP INFORMATION---------------------------------------------------
# if your OPTION_GROUP is True, then follow the steps below or you can skip it.
# 1. Enter GROUP_MEMBER_DICT: dictionary with group name as key and bbg official code as value
# 2. Enter GROUP_CONSTRAINT_DICT: dictionary with group name as key and group constraint as value
group_list = []
for asset in ASSETS_LIST:
    group_list.append(ASSETS_DICT.get(asset)[3])
GROUP_MEMBER_DICT = {}
for group in np.unique(group_list):
    GROUP_MEMBER_DICT.update({group:[ asset for asset in ASSETS_DICT.keys() if (ASSETS_DICT.get(asset)[3] == group)]})

GROUP_CONSTRAINT_DICT = {
	'GROWTH' : 0.6,
	'DEFENSE' : 0.6,
	'STABLE' : 0.07,
	'INFLATION' : 0.25,
    'CASH':0.1,
}
if len(GROUP_MEMBER_DICT.keys()) != len(GROUP_CONSTRAINT_DICT.keys()):
    print("In the first user-input part we have groups:",GROUP_MEMBER_DICT.keys())
    print("In the second user-input part we have groups:", GROUP_CONSTRAINT_DICT.keys())
    raise AssertionError("Asset Group Number doesn't match, please check parameter file for detail")
# --------------------------------------------------ETF INFORMATION-----------------------------------------------------
# if your OPTION_ETF is True, then follow the steps below or you can skip it.
# 1. Enter ETF_ASSET_DICT: dictionary with etf bbg code as key and index asset bbg code as value

ETF_ASSET_DICT = {
    # growth
    'SPY US Equity': 'SPTR Index',
    'EWU US Equity': 'SX5T Index',
    'EWJ US Equity': 'TPXDDVD Index',
    '2800 HK Equity': 'HSI Index',
    'IEMG US Equity': 'M1EF Index',
    'L100 LN Equity': 'TUKXG Index',

    # defense
    'AGG US Equity': 'LBUSTRUU Index',
    'IEI US Equity': 'LUATTRUU Index',
    'IGB AU Equity': 'BATY0 Index',
    'IDEU LN Equity': 'LETGTREU Index',

    # stable
    'EMB US Equity': 'EMUSTRUU Index',
    'HYG US Equity': 'LF98TRUU Index',
    'LQD US Equity': 'LUACTRUU Index',
    'IHYG LN Equity': 'LP01TREU Index',
    'EURORNT LX Equity': 'LP06TREU Index',

    # inflation
    'TIP US Equity': 'BCIT1T Index',
    'XEIN GR Equity': 'BEIG1T Index',
    'GSG US Equity': 'SPGSCI Index',
}






################################################################ No need Input in following part ###############################################################
################################################################ No need Input in following part ###############################################################
################################################################ No need Input in following part ###############################################################
print('-------------------------------------------------------------ASSET INFORMATION--------------------------------------------------------------------')

# if OPTION_GROUP:
#     ASSETS_LIST = []
#     for group in GROUP_MEMBER_DICT.keys():
#         ASSETS_LIST = ASSETS_LIST + GROUP_MEMBER_DICT.get(group)

print('Asset Data Begin Downloading Time:',ASSET_START_TIME,'\n')

print('We have {} Asset Code'.format(len(ASSETS_LIST)),',they are : \n', ASSETS_LIST,'\n' )

print('We have {} Asset Name'.format(len(ASSETS_NAME_DICT.values())),', they are : \n', ASSETS_NAME_DICT.values(),'\n')

print('We have {} Asset Constraints'.format(len(ASSET_CONSTRAINT_DICT.values())),', they are : \n')

for asset in ASSET_CONSTRAINT_DICT.keys():
    print('Asset ' + asset +':\t'
		+ 'lower Boundary ' + str(round(ASSET_CONSTRAINT_DICT.get(asset)[0]*100,0)) + '%'
		+ '\t,upper Boundary ' + str(round(ASSET_CONSTRAINT_DICT.get(asset)[1]*100,0)) + '%'
		)
print('\n')

print('We have {} Asset transaction cost'.format(len(ASSET_TRANSACTION_DICT.values())),', they are : \n')
for asset in ASSET_TRANSACTION_DICT.keys():
	print('Asset ' + asset +':\t' 
		+  str(round(ASSET_TRANSACTION_DICT.get(asset)*100,2)) + '%'
		)
print('\n')

if len(ASSETS_LIST) == len(ASSETS_NAME_DICT.keys()) == len(ASSET_CONSTRAINT_DICT.keys()) == len(ASSET_TRANSACTION_DICT.keys()):
  print('Your Parameter form for index assets are correct')
else:
  raise ValueError('Your Parameter form for index assets are wrong, please refer to the parameter.py file for modification')

print('-------------------------------------------------------------GROUP INFORMATION--------------------------------------------------------------------')

growth_id = list(np.arange(0,
				len(GROUP_MEMBER_DICT.get('GROWTH'))))

defense_id = list(np.arange(len(growth_id), 
				len(growth_id) + len(GROUP_MEMBER_DICT.get('DEFENSE'))))

stable_id = list(np.arange(len(growth_id + defense_id),
				len(growth_id + defense_id)+  len(GROUP_MEMBER_DICT.get('STABLE') )))

inflation_id = list(np.arange(len(growth_id + defense_id + stable_id),
					len(growth_id + defense_id + stable_id) + len(GROUP_MEMBER_DICT.get('INFLATION') )))

cash_id = list(np.arange(len(growth_id + defense_id + stable_id + inflation_id),
					len(growth_id + defense_id + stable_id + inflation_id) + len(GROUP_MEMBER_DICT.get('CASH') )))

GROUP_ID_DICT = {
	'GROWTH' : growth_id,
	'DEFENSE' : defense_id,
	'STABLE' : stable_id,
	'INFLATION' : inflation_id,
    'CASH':cash_id,
}
if OPTION_GROUP:
  print('We consider group constraint in our asset allocation part\n')
  print('Those Group asset is:')
  for group in GROUP_ID_DICT.keys():
    group_id = GROUP_ID_DICT.get(group)
    print('In group ' + group + ', Our asset is:')
    print(np.array(list(ASSETS_NAME_DICT.values()))[group_id],'\n')

  print('Those GROUP CONSTRAINT  is : \n', GROUP_CONSTRAINT_DICT )
  print('')
else:
  print("We don't consider group constraint in our asset allocation part")

print('-------------------------------------------------------------ETF INFORMATION--------------------------------------------------------------------')
if ETF_OPTION:
  print('We consider ETF in our asset allocation part\n')
  print('ETF Data Begin Downloading Time:',ETF_START_TIME,'\n')
  print('We have {} ETF data to download'.format(len(ETF_ASSET_DICT.keys())),', They are :')
  for ETF in ETF_ASSET_DICT.keys():
  	print('Index Asset: ' + ETF_ASSET_DICT.get(ETF) + ' ------ ETF Asset: ' + ETF)
  if len(ASSETS_LIST) == len(ETF_ASSET_DICT.keys()) == len(ASSET_CONSTRAINT_DICT.keys()) == len(ASSET_TRANSACTION_DICT.keys()):
    print('Your Parameter form for ETF assets are correct')
  else:
    raise ValueError('Your Parameter form for ETF assets are wrong, please refer to the parameter.py file for modification')
else:
  print("We don't consider ETF in our asset allocation part\n")
