from IPython.display import Image

Image("/kaggle/input/fundamentals-of-risk-and-return-of-a-portfolio/Lesson1GoalHeaderImage.png")
from IPython.display import Image

Image("/kaggle/input/fundamentals-of-risk-and-return-of-a-portfolio/Lesson1Fig1_1.png")

from IPython.display import Image

Image("/kaggle/input/fundamentals-of-risk-and-return-of-a-portfolio/Lesson1Eqn1_1.png")
#Python code to compute the daily returns in percentage, of Dow Stocks listed in Sec.1.1

#calls function StockReturnsComputing to compute asset returns



#dependencies

import numpy as np

import pandas as pd



#input stock prices dataset

stockFileName = '/kaggle/input/fundamentals-of-risk-and-return-of-a-portfolio/DJIA_Apr112014_Apr112019.csv'

rows = 1259  #excluding header

columns = 29  #excluding date



#read stock prices 

df = pd.read_csv(stockFileName,  nrows= rows)



#extract asset labels

assetLabels = df.columns[1:columns+1].tolist()

print(assetLabels)



#extract asset prices data

stockPrice = df.iloc[0:, 1:]

print(stockPrice.shape)



#print stock price

print(stockPrice)

#function to compute asset returns 

def StockReturnsComputing(StockPrice, Rows, Columns):

    

    import numpy as np

    

    StockReturn = np.zeros([Rows-1, Columns])

    for j in range(Columns):        # j: Assets

        for i in range(Rows-1):     # i: Daily Prices

            StockReturn[i,j]=((StockPrice[i+1, j]-StockPrice[i,j])/StockPrice[i,j])* 100



    return StockReturn
#compute daily returns in percentage of the Dow stocks



import numpy as np



stockPriceArray = np.asarray(stockPrice)

[Rows, Cols]=stockPriceArray.shape

stockReturns = StockReturnsComputing(stockPriceArray, Rows, Cols)

print('Daily returns of selective Dow 30 stocks\n', stockReturns)

from IPython.display import Image

Image("/kaggle/input/fundamentals-of-risk-and-return-of-a-portfolio/Lesson1Fig1_2.png")
#compute mean returns and variance covariance matrix of returns

meanReturns = np.mean(stockReturns, axis = 0)

print('Mean returns of Dow Stocks:\n',  meanReturns)

covReturns = np.cov(stockReturns, rowvar=False)

print('Variance-covariance matrix of returns of Dow Stocks:\n')

print(covReturns)
from IPython.display import Image

Image("/kaggle/input/fundamentals-of-risk-and-return-of-a-portfolio/Lesson1Eqn1_2.png")
#compute betas of Dow stocks over a 3-year historical period,

#DJIA Index- April 2016 to April 2019



#dependencies

import numpy as np

import pandas as pd



#input stock prices and market datasets

stockFileName = '/kaggle/input/fundamentals-of-risk-and-return-of-a-portfolio/DJIAkpf1Apr2016to20193YBeta.csv'

marketFileName = '/kaggle/input/fundamentals-of-risk-and-return-of-a-portfolio/DJIAMarketDataApr2016to20193YBeta.csv'

stockRows = 756  #excluding header 

stockColumns = 15  #excluding date

marketRows = 756

marketColumns = 7



#read stock prices dataset and market dataset 

dfStock = pd.read_csv(stockFileName,  nrows= stockRows)

dfMarket = pd.read_csv(marketFileName, nrows = marketRows)



#extract asset labels of stocks in the portfolio

assetLabels = dfStock.columns[1:stockColumns+1].tolist()

print('Portfolio stocks\n', assetLabels)



#extract asset prices data and market data

stockData = dfStock.iloc[0:, 1:]

marketData = dfMarket.iloc[0:, [4]] #closing price 



#compute asset returns

arrayStockData = np.asarray(stockData)

[sRows, sCols]=arrayStockData.shape

stockReturns = StockReturnsComputing(arrayStockData, sRows, sCols)



#compute market returns

arrayMarketData = np.asarray(marketData)

[mRows, mCols]=arrayMarketData.shape

marketReturns = StockReturnsComputing(arrayMarketData, mRows, mCols)



#compute betas of assets in the portfolio

beta= []

Var = np.var(marketReturns, ddof =1)

for i in range(stockColumns):

    CovarMat = np.cov(marketReturns[:,0], stockReturns[:, i ])

    Covar  = CovarMat[1,0]

    beta.append(Covar/Var)

    

    

#output betas of assets in the portfolio

print('Asset Betas:  \n')

for data in beta:

    print('{:9.3f}'.format(data))

from IPython.display import Image

Image("/kaggle/input/fundamentals-of-risk-and-return-of-a-portfolio/Lesson1Eqn1_3.png")
from IPython.display import Image

Image("/kaggle/input/fundamentals-of-risk-and-return-of-a-portfolio/Lesson1Eqn1_4.png")
from IPython.display import Image

Image("/kaggle/input/fundamentals-of-risk-and-return-of-a-portfolio/Lesson1Eqn1_5.png")
from IPython.display import Image

Image("/kaggle/input/fundamentals-of-risk-and-return-of-a-portfolio/Lesson1Eqn1_6.png")
from IPython.display import Image

Image("/kaggle/input/fundamentals-of-risk-and-return-of-a-portfolio/Lesson1Fig1_3.png")

#portfolio risk, expected return and portfolio beta computation



#input weights and asset betas for portfolio P as described in Sec. 1.5

weights = np.array([0.09, 0.07, 0.03, 0.02, 0.07, 0.06, 0.04, 0.07, 0.11, \

                    0.08, 0.09, 0.07, 0.05, 0.11, 0.04])

assetBeta = np.array([1.13, 1.09, 1.39, 1.53, 1.15, 0.77, 1.32, 0.94, 0.98,\

                      1.12, 0.46, 0.55, 0.74, 0.95, 0.85])



#compute mean and covariance of asset returns of portfolio P available in stockReturns

meanReturns = np.mean(stockReturns, axis = 0)

covReturns = np.cov(stockReturns, rowvar=False)



#compute portfolio risk

portfolioRisk = np.matmul((np.matmul(weights,covReturns)), np.transpose(weights))



#compute annualized portfolio risk for trading days = 251

annualizedRisk  =   np.sqrt(portfolioRisk*251) 



#compute expected portfolio return

portfolioReturn = np.matmul(np.array(meanReturns),weights.T)



#compute annualized expected portfolio return

annualizedReturn = 251*np.array(portfolioReturn) 



#compute portfolio beta

portfolioBeta = np.matmul(assetBeta,weights.T)



#display results

print("\n Annualized Portfolio Risk: %4.2f" % annualizedRisk,"%")

print("\n Annualized Expected Portfolio Return: %4.2f" % annualizedReturn,"%")

print("\n Portfolio Beta:%4.2f" % portfolioBeta)

from IPython.display import Image

Image("/kaggle/input/fundamentals-of-risk-and-return-of-a-portfolio/Lesson1ExitTailImage.png")