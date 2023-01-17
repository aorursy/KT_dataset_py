#!pip install plotly==4.4.1

#!pip install chart_studio

#!pip install xlrd 



import pandas as pd

import numpy as np



def excel_to_df(excel_sheet):

	df = pd.read_excel(excel_sheet)

	df.dropna(how='all', inplace=True)



	index_PL = int(df.loc[df['Data provided by SimFin']=='Profit & Loss statement'].index[0])

	index_CF = int(df.loc[df['Data provided by SimFin']=='Cash Flow statement'].index[0])

	index_BS = int(df.loc[df['Data provided by SimFin']=='Balance Sheet'].index[0])



	df_PL = df.iloc[index_PL:index_BS-1, 1:]

	df_PL.dropna(how='all', inplace=True)

	df_PL.columns = df_PL.iloc[0]

	df_PL = df_PL[1:]

	df_PL.set_index("in million USD", inplace=True)

	(df_PL.fillna(0, inplace=True))

	



	df_BS = df.iloc[index_BS-1:index_CF-2, 1:]

	df_BS.dropna(how='all', inplace=True)

	df_BS.columns = df_BS.iloc[0]

	df_BS = df_BS[1:]

	df_BS.set_index("in million USD", inplace=True)

	df_BS.fillna(0, inplace=True)

	



	df_CF = df.iloc[index_CF-2:, 1:]

	df_CF.dropna(how='all', inplace=True)

	df_CF.columns = df_CF.iloc[0]

	df_CF = df_CF[1:]

	df_CF.set_index("in million USD", inplace=True)

	df_CF.fillna(0, inplace=True)

	

	df_CF = df_CF.T

	df_BS = df_BS.T

	df_PL = df_PL.T

    

	return df, df_PL, df_BS, df_CF



def combine_regexes(regexes):

	return "(" + ")|(".join(regexes) + ")"
import os

#import tk_library_py

#from tk_library_py import excel_to_df
# Show the files and their pathnames

for dirname, _, filenames in os.walk('/kaggle/input/rossdata'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

for dirname, _, filenames in os.walk('/kaggle/input/rossgeneral'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
_, ross_PL, ross_BS, ross_CF = excel_to_df("/kaggle/input/rossdata/SimFin-ross.xlsx")

#I get EPS data from Wind, respectively 1.93 in 2009 and 3.58 in 2018.

#Now I can calculate the EPS ACGR from 2009 to 2018. 

EPSACGR=(3.58/1.93)**(1/9)-1

print(EPSACGR)
#With ACGR, I can estimate the EPS in the following 10 years.

#EPS10 is the EPS from 2019 to 2028

EPS10 = []

for x in range(1, 11):

    EPS10.append((1+EPSACGR)**x*3.58)

print(EPS10)





#EPS2028 = EPS10[9]

#print(EPS2028)
#I got the current Average PE ratio from Wind

PE = 26.699

EPS2028 = EPS10[9]



#The estimate future price of ROST will be



FP2028 = PE * EPS2028



print ("The future prince is:",FP2028)

#I got the Mid WACC from Finbox

#https://finbox.com/NASDAQGS:ROST/models/saved/m-qehwaer8/



WACC = 0.08



# Because I use current(2019) PE, so I only discount 9 years from 2028 

n = 9



# The Current Target Buy Price will be

CTBP = FP2028 / (1 + WACC) ** (n)



print ("The target buy price is",CTBP)
#Add Margin of Safety 

MS = 0.28

CTBPMS = CTBP * (1-MS)

CTBPMS
#Calculate Interest Coverage Ratio

ross_PL["Interest Coverage Ratio"] = ross_PL["Operating Income (Loss)"] / ross_PL["Non-Operating Income (Loss)"]

ross_PL["Interest Coverage Ratio"]

# Debt to Equity Ratio

ross_BS["D/E Ratio"] = ross_BS["Total Liabilities"] / ross_BS["Total Equity"]

ross_BS["D/E Ratio"]
ross_BS["D/E Ratio"].plot()
# Ross stores has no preferred stock, no intengible assets, and no goodwill



ross_BS["book value"] = ross_BS["Total Assets"] - ross_BS["Total Liabilities"]

ross_BS["book value"]
ross_BS["book value"].plot()
%matplotlib inline

ross_BS[["Total Assets", "Total Liabilities", "Total Equity"]].plot()
# Calculate Price-to-Earnings Growth Ratio  

# The current PE ratio is from https://simfin.com/data/companies/200092

# The 2020 growth rate is from https://www.nasdaq.com/market-activity/stocks/rost/price-earnings-peg-ratios



PE = 27.47 

GR = 7.36 



PEG = PE / GR



print("The PEG Ratio is", PEG)

#Calculate ROE

#Book Value = Shareholder's Equity

ROE = ross_CF["Net Income/Starting Line"]/ ross_BS["book value"] 

ROE
ROE.plot()
# End of analysis

# Thanks!