#1.install ipython

!pip install IPython
#2. install simfin

!pip install simfin
#3. install plotly

!pip install plotly
#4. install pandas datareader

!pip install pandas_datareader
#5. install mpl finance

!pip install mpl_finance
#6. chart studio

!pip install chart_studio
#7. install xlrd

!pip install xlrd
#8. install seabon

!pip install seaborn
#9. install quandl

!pip install quandl
#10. install chart studio

!pip install chart_studio
#2020/2/22 22:00  Anthony's code from Wechat message.

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
#1. plt

import matplotlib.pyplot as plt
#2. sp

import scipy as sp
#3. xlrd

import xlrd
#4. os

import os
#5. cahrt studio

import chart_studio
#6. sns

import seaborn as sns

sns.set()
#7. sf

import simfin as sf
#8. py

import chart_studio.plotly as py
#9. go

import plotly.graph_objs as go
#10. time

import time
#11. plot acf

from statsmodels.graphics.tsaplots import plot_acf 
#12. plot pacf

from statsmodels.graphics.tsaplots import plot_pacf    
#13. ADF

from statsmodels.tsa.stattools import adfuller as ADF  
#14. acorr ljungbox

from statsmodels.stats.diagnostic import acorr_ljungbox  
#15. ARIMA

from statsmodels.tsa.arima_model import ARIMA
#16. web

import pandas_datareader as web
#17. datetime

import datetime
#18. plt

import matplotlib.pyplot as plt
#19. mticker

from matplotlib import ticker as mticker
#20. mpl

import mpl_finance as mpl
#21. nvda logo

from IPython.display import Image

import os

Image("../input/nvidialogo/logo.png")
Image("../input/nvdastatements/balancestatement.png")
Image("../input/nvdastatements/incomestatement.png")
Image("../input/nvdastatements/cashflowstatement.png")
def DownloadData(Ticker,FileSavePath):   #required balance statement, income statement and cashflow statement



    #set apikey

    sf.set_api_key('free')

    sf.set_data_dir(FileSavePath)



    #BS,IS,CS

    Balance_Statement = sf.load_balance(variant='annual', market='us')

    Income_Statement = sf.load_income(variant = 'annual')

    Cashflow_Statement =  sf.load_cashflow(variant='annual', market='us')



    #BS,IS,CS

    DataInformation_Balance = Balance_Statement.loc[Ticker]

    DataInformation_Income = Income_Statement.loc[Ticker]

    DataInformation_Cashflow = Cashflow_Statement.loc[Ticker]



    #BS,IS,CS

    Balance_SavePath = [FileSavePath, r'\\',r' Balance.csv']

    Income_SavePath = [FileSavePath,r'\\',r'Income.csv']

    Cashflow_SavePath = [FileSavePath, r'\\', r' Cashflow.csv']



    #BS,IS,CS

    Balance_SavePath = ''.join(Balance_SavePath)

    Income_SavePath = ''.join(Income_SavePath)

    Cashflow_SavePath = ''.join(Cashflow_SavePath)    



    #BS,IS,CS

    DataInformation_Balance.to_csv(Balance_SavePath)

    DataInformation_Income.to_csv(Income_SavePath)

    DataInformation_Cashflow.to_csv(Cashflow_SavePath)   



    #BS,IS,CS

    print(DataInformation_Balance.head())

    [print(x) for x in DataInformation_Balance.columns]   

    print('\n')

    print(DataInformation_Income.head())

    [print(x) for x in DataInformation_Income.columns]

    print(DataInformation_Cashflow.head())

    [print(x) for x in DataInformation_Cashflow.columns]

    

    #BS,IS,CS

    DataInformation_Balance.index = DataInformation_Balance['Fiscal Year']

    DataInformation_Income.index = DataInformation_Income['Fiscal Year']

    DataInformation_Cashflow.index = DataInformation_Cashflow['Fiscal Year']



    #BS,IS,CS

    return DataInformation_Income,DataInformation_Cashflow,DataInformation_Balance
#nvda data

Ticker = 'NVDA'



FileSavePath = os.getcwd()



DataInformation_Income,DataInformation_Cashflow,DataInformation_Balance = DownloadData(Ticker,FileSavePath)
def ReadLoadData(Ticker,FileSavePath ):

   

    #1.BS,IS,CS

    N = '''

    def FindOutTheCurrentPath(Path):

        Balance_Sheet_Path = ''.join([Path, r'\\', r'Balance.csv'])

        CashFlow_Sheet_Path = ''.join([Path, r'\\', r'Cashflow.csv'])

        Income_Sheet_Path = ''.join([Path, r'\\', r'Income.csv'])

        return Balance_Sheet_Path, CashFlow_Sheet_Path, Income_Sheet_Path

    '''



    #2.BS,IS,CS

    Balance_SavePath, Cashflow_SavePath, Income_SavePath = DownloadData(Ticker,FileSavePath)

    N = '''

    DataInformation_Balance = pd.read_csv(Balance_SavePath)

    DataInformation_Income = pd.read_csv(Income_SavePath)

    DataInformation_Cashflow = pd.read_csv(Cashflow_SavePath)

    '''



    #3.BS,IS,CS

    DataInformation_Balance.index = DataInformation_Balance['Fiscal Year']

    DataInformation_Income.index = DataInformation_Income['Fiscal Year']

    DataInformation_Cashflow.index = DataInformation_Cashflow['Fiscal Year']



    #4.BS,IS,CS

    print('\n','Balance Statement','\n',DataInformation_Balance.head())

    print('The list of Balance Sheet Columns:')

    [print(x) for x in DataInformation_Balance.columns]

    print('Income Statement','\n',DataInformation_Income.head())

    print('The list of Income Statement Columns:')

    [print(x) for x in DataInformation_Income.columns]

    print('\n','Cash flow Statement','\n',DataInformation_Cashflow.head())

    print('The list of Cash flow Statement Columns:')

    [print(x) for x in DataInformation_Cashflow.columns]

    

    #5.BS,IS,CS

    return DataInformation_Income, DataInformation_Cashflow, DataInformation_Balance
Image("../input/excelpic/excelpic.png")
def excel_to_df(excel_sheet): #excel→df

    

    df = pd.read_excel(excel_sheet)

    df.dropna(how='all', inplace=True)



    #1.BS,PL,CF

    index_BS = int(df.loc[df['Data provided by SimFin'] == 'Balance Sheet'].index[0])

    index_PL = int(df.loc[df['Data provided by SimFin'] == 'Profit & Loss statement'].index[0])

    index_CF = int(df.loc[df['Data provided by SimFin'] == 'Cash Flow statement'].index[0])



    #2.PL

    df_PL = df.iloc[index_PL:index_BS - 1, 1:]

    df_PL.dropna(how='all', inplace=True)

    df_PL.columns = df_PL.iloc[0]

    df_PL = df_PL[1:]

    df_PL.set_index("in million USD", inplace=True)

    (df_PL.fillna(0, inplace=True))



    #3.BS

    df_BS = df.iloc[index_BS - 1:index_CF - 2, 1:]

    df_BS.dropna(how='all', inplace=True)

    df_BS.columns = df_BS.iloc[0]

    df_BS = df_BS[1:]

    df_BS.set_index("in million USD", inplace=True)

    df_BS.fillna(0, inplace=True)



    #4.CF

    df_CF = df.iloc[index_CF - 2:, 1:]

    df_CF.dropna(how='all', inplace=True)

    df_CF.columns = df_CF.iloc[0]

    df_CF = df_CF[1:]

    df_CF.set_index("in million USD", inplace=True)

    df_CF.fillna(0, inplace=True)



    #5.CF,BS,PL

    df_CF = df_CF.T

    df_BS = df_BS.T

    df_PL = df_PL.T



    

    return df, df_PL, df_BS, df_CF
def combine_regexes(regexes):

    

    return "(" + ")|(".join(regexes) + ")"
Image("../input/yahoologo/yahoo.png")
def get_today():  #get the latest data for next step calculation

    

    #time settings

    today = time.localtime(time.time())

    today_year  = today.tm_year    

    today_month = today.tm_mon    

    today_date = today.tm_mday    

    today_format = datetime.datetime(today_year, today_month, today_date)    

    today_format = today_format.strftime('%Y%m%d')

    

    return today_format



    

class GetData(object):

    

    #No.1

    def __init__(self,Name,Startdate,Enddate,Datasource='yahoo'):

        

        #self

        self.Name = Name

        self.Startdate = Startdate

        self.Enddate = Enddate

        self.Datasource  = Datasource

        

        #update

        New_Startdate_format = time.strptime(self.Startdate,'%Y%m%d')

        New_Enddate_format = time.strptime(self.Enddate,'%Y%m%d')

        New_Startdate = datetime.datetime(New_Startdate_format.tm_year,New_Startdate_format.tm_mon,New_Startdate_format.tm_mday).strftime('%m/%d/%Y')

        New_Enddate = datetime.datetime(New_Enddate_format.tm_year,New_Enddate_format.tm_mon,New_Enddate_format.tm_mday).strftime('%m/%d/%Y')

        

        #self

        self.NewStartdate = New_Startdate

        self.NewEnddate = New_Enddate        

        

        

    #No.2

    def DownloadData(self):

        DataInformation = web.DataReader(self.Name,self.Datasource,self.NewStartdate,self.NewEnddate)

        

        ##1

        Trade_Date_List = []

        for TradeDate in range(len(DataInformation.index)):

            Trade_Date_Format = time.strptime(str(DataInformation.index[TradeDate]), '%Y-%m-%d %H:%M:%S')

            Trade_Date = datetime.datetime(Trade_Date_Format.tm_year, Trade_Date_Format.tm_mon,Trade_Date_Format.tm_mday).strftime('%Y%m%d')

            Trade_Date_List.append(Trade_Date)

            

        ##2    

        Index_Trade_Date_List = []

        for TradeDate in range(len(DataInformation.index)):

            Index_Trade_Date_Format = time.strptime(str(DataInformation.index[TradeDate]), '%Y-%m-%d %H:%M:%S')

            Index_Trade_Date = datetime.datetime(Index_Trade_Date_Format.tm_year, Index_Trade_Date_Format.tm_mon,Index_Trade_Date_Format.tm_mday).strftime('%Y-%m-%d')

            Index_Trade_Date_List.append(Index_Trade_Date)           

  

        ##3 info

        DataInformation['IndexTradeDate'] = Index_Trade_Date_List

        DataInformation['TradeDate'] = Trade_Date_List

        DataInformation['Name'] = self.Name

       

        ##4 columns

        all_columns = DataInformation.columns.tolist()

        all_columns.pop(-1)

        all_columns.insert(0, 'Name')

        all_columns.pop(-1)

        all_columns.insert(1, 'TradeDate')

        all_columns.pop(-1)

        all_columns.insert(2, 'IndexTradeDate')

        

        ##5

        DataInformation = DataInformation.reindex(columns = all_columns) 

        CurrentFilePath = GetData.SaveData(self,DataInformation)

        

        ##6

        print('Data has already been downloaded from %s'%self.Datasource , '!')

        print('Information : \n','Name : %s'%self.Name,'\n','Columns : ',all_columns,'\n','Total index ',len(DataInformation.index),'\n'

              ,'Saved Path in csv: ',CurrentFilePath[0],'\n','Saved Path in excel:',CurrentFilePath[1])

        

        ##7

        return DataInformation    

    

    

    #No.3

    def GetCurrentPath(self):

        

        CurrentFilePath = os.getcwd()

        

        return CurrentFilePath    

    

    

    #No.4

    def SaveData(self,DataInformation):

        

        ##1

        CurrentFilePath = GetData.GetCurrentPath(self)    

        N='''

        CurrentFilePath = CurrentFilePath + str(r'\DataBase')

        if not os.path.exists(CurrentFilePath):

            os.makedirs(CurrentFilePath)



        CurrentFilePath_csv = CurrentFilePath + str(r'\Csv')

        if not os.path.exists(CurrentFilePath_csv):

            os.makedirs(CurrentFilePath_csv)

            

        CurrentFilePath_excel = CurrentFilePath + str(r'\Xlsx')

        if not os.path.exists(CurrentFilePath_excel):

            os.makedirs(CurrentFilePath_excel)

        '''

        

        ##2

        CurrentFilePath_csv = CurrentFilePath

        CurrentFilePath_excel = CurrentFilePath

        CurrentFilePath_csv = CurrentFilePath_csv + str(r'/') + str(self.Name)

        CurrentFilePath_csv = CurrentFilePath_csv + str(r'.csv')

        CurrentFilePath_excel = CurrentFilePath_excel + str(r'/')+ str(self.Name)

        CurrentFilePath_excel = CurrentFilePath_excel + str(r'.xlsx')

        

        ##3 csv excel

        DataInformation.to_csv(CurrentFilePath_csv,encoding='utf_8_sig')

        DataInformation.to_excel(excel_writer=CurrentFilePath_excel, encoding='utf_8_sig')

        

        ##4 drawplot

        GetData.DrawPlot(self,DataInformation)

        

        

        return CurrentFilePath_csv,CurrentFilePath_excel  

    

    

    #No.5

    def ReadDataFromCsv(self,path):

        

        DataInformation = pd.read_csv(path)

        

        return DataInformation    



    

    #No.6

    def ReadDataFromExcel(self,path):

        

        DataInformation = pd.read_excel(path)

        

        return  DataInformation    



    

    #No.7

    def __str__(self):

        

        return 'Name : '+str(self.Name)+'\n'+'Start Date : '+str(self.NewStartdate)+'\n'+'End Date : '+str(self.NewEnddate)+'\n'+'Data Source : '+str(self.Datasource)
Ticker = 'NVDA'



FileSavePath = os.getcwd()



DataInformation_Income, DataInformation_Cashflow, DataInformation_Balance = ReadLoadData(Ticker,FileSavePath)



Startdate = '20180101'



Enddate = get_today()



NVDA = GetData('NVDA',Startdate,Enddate,'yahoo')
Image("../input/assignment1/keymultiples.png")
#balance,income,cashflow

DataInformation_Balance.index = DataInformation_Balance['Fiscal Year']

DataInformation_Income.index = DataInformation_Income['Fiscal Year']

DataInformation_Cashflow.index = DataInformation_Cashflow['Fiscal Year']



#formula

DataInformation_Balance['NAV'] = DataInformation_Balance['Total Equity'] / DataInformation_Balance['Shares (Basic)']

DataInformation_Balance['Working Capital'] = DataInformation_Balance['Total Current Assets'] - DataInformation_Balance['Total Current Liabilities']

DataInformation_Balance['Current Ratio'] = DataInformation_Balance['Total Current Assets'] / DataInformation_Balance['Total Current Liabilities']

DataInformation_Balance['Book Value'] = DataInformation_Balance['Total Assets'] - DataInformation_Balance['Total Liabilities']





print(DataInformation_Balance[[ 'Total Equity', 'Total Current Assets', 'Total Current Liabilities', 'Total Liabilities','Total Assets', 'NAV', 'Working Capital', 'Book Value', 'Current Ratio','Shares (Diluted)']])
#No.1

DataInformation_Balance[['Total Assets', 'Total Liabilities', 'Total Equity']].plot()



#No.2

DataInformation_Balance[['Current Ratio']].plot()



#No.3

DataInformation_Balance[['Book Value']].plot()



plt.show()
Image("../input/assignment1/additional.png")
#formula

DataInformation_Income['EPS'] = DataInformation_Income['Net Income (Common)'] / DataInformation_Income['Shares (Basic)']

DataInformation_Income['EPS ACGR'] = np.log(DataInformation_Income['EPS'] / DataInformation_Income['EPS'].shift(1))



print(DataInformation_Income['EPS ACGR'])
#No.1

DataInformation_Income[['EPS ACGR']].plot()

plt.show()
print(DataInformation_Income['EPS'])
def GetLinearRegression(DataInformation,k=1):

    

    #No.1

    Data_X = [x + 1 for x in range ( len (DataInformation ) ) ]

    Data_X = np.array(Data_X)

    

    #No.2

    Data_Y = np.array(DataInformation['EPS'])

    Data_Fit = np.polyfit(Data_X,Data_Y,k)

    Data_YY = np.zeros(len(Data_X))

    

    #No.3

    for x in range(k+1):

        Data_YY += Data_Fit[x]*Data_X**(k-x)

    Data_XX = np.arange(max(Data_X)+1,max(Data_X)+12,1)

    Data_XX = np.array(Data_XX)

    Data_ZZ = np.zeros(len(Data_XX))

    

    #No.4

    for x in range(k+1):

        Data_ZZ += Data_Fit[x]*Data_XX**(k-x)

        

    #No.5    

    plt.scatter(Data_X,Data_Y)

    plt.plot(Data_X,Data_YY)

    plt.scatter(Data_XX,Data_ZZ)

    Title_Name = ''.join(['The ',str(k),'th Polynomial Curve Fitting'])

    plt.legend([Title_Name,'Actual','Forecast'])

    plt.title('EPS Forecast')

    plt.show()
#when k=1

GetLinearRegression(DataInformation_Income, k=1)
#data resources

EPS_Forecast = 1.34 #https://simfin.com/data/companies/172199

PE_Ratio = 37.63 #https://simfin.com/data/companies/172199

Discount_Rate = 0.106 #https://finbox.com/NASDAQGS:NVDA/models/ddm-sg



#formula

Current_Target_Buy_Price = (EPS_Forecast * PE_Ratio) / (1.00 + Discount_Rate)**10

Current_Target_Buy_Price
def DataInformation_to_df(excel_sheet):

    df_Contribution_Margin = (df["Revenue"] + df["Cost of Revenue"]) / df["Revenue"]

    df_Break_Even_Point = (-df["Operating Expenses"]) / df_Conribution_Margin

    df_Margin_of_Safety = (df["Revenue"] - df_Break_Even_Point) / df["Revenue"]

    df_PL_Margin_of_Safety.mean()
#data resources

FC = 3367

VC = 4545

net_sales = 10018

gross_margin = 0.595



#formula

margin_of_safety = (net_sales - (FC /((net_sales - VC) / net_sales)))/net_sales

margin_of_safety
DataInformation_Balance['Debt to Equity Ratio'] = DataInformation_Balance['Total Liabilities'] / DataInformation_Balance['Total Equity']



print(DataInformation_Balance[['Debt to Equity Ratio']])
DataInformation_Income['Interest Coverage Ratio'] = DataInformation_Income['Pretax Income (Loss)'] / abs(DataInformation_Income['Interest Expense, Net'])



print(DataInformation_Income[['Interest Coverage Ratio']])