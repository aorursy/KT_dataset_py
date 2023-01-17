from IPython.display import Image

Image("/kaggle/input/sharpe-ratio-based-portfolio-optimization/Lesson6GoalHeaderImage.png")
from IPython.display import Image

Image("/kaggle/input/sharpe-ratio-based-portfolio-optimization/Lesson6Eqn6_1.png")
from IPython.display import Image

Image("/kaggle/input/sharpe-ratio-based-portfolio-optimization/Lesson6Eqn6_2.png")

from IPython.display import Image

Image("/kaggle/input/sharpe-ratio-based-portfolio-optimization/Lesson6Eqn6_3.png")
#function to undertake Sharpe Ratio maximization subject to 

#basic constraints of the portfolio



#dependencies

import numpy as np

from scipy import optimize 



def MaximizeSharpeRatioOptmzn(MeanReturns, CovarReturns, RiskFreeRate, PortfolioSize):

    

    # define maximization of Sharpe Ratio using principle of duality

    def  f(x, MeanReturns, CovarReturns, RiskFreeRate, PortfolioSize):

        funcDenomr = np.sqrt(np.matmul(np.matmul(x, CovarReturns), x.T) )

        funcNumer = np.matmul(np.array(MeanReturns),x.T)-RiskFreeRate

        func = -(funcNumer / funcDenomr)

        return func



    #define equality constraint representing fully invested portfolio

    def constraintEq(x):

        A=np.ones(x.shape)

        b=1

        constraintVal = np.matmul(A,x.T)-b 

        return constraintVal

    

    

    #define bounds and other parameters

    xinit=np.repeat(0.33, PortfolioSize)

    cons = ({'type': 'eq', 'fun':constraintEq})

    lb = 0

    ub = 1

    bnds = tuple([(lb,ub) for x in xinit])

    

    #invoke minimize solver

    opt = optimize.minimize (f, x0 = xinit, args = (MeanReturns, CovarReturns,\

                             RiskFreeRate, PortfolioSize), method = 'SLSQP',  \

                             bounds = bnds, constraints = cons, tol = 10**-3)

    

    return opt

    
# function computes asset returns 

def StockReturnsComputing(StockPrice, Rows, Columns):

    

    import numpy as np

    

    StockReturn = np.zeros([Rows-1, Columns])

    for j in range(Columns):        # j: Assets

        for i in range(Rows-1):     # i: Daily Prices

            StockReturn[i,j]=((StockPrice[i+1, j]-StockPrice[i,j])/StockPrice[i,j])* 100



    return StockReturn
#obtain mean and variance-covariance matrix of returns for k-portfolio 1



#Dependencies

import numpy as np

import pandas as pd







#input k portfolio 1 dataset comprising 15 stocks

StockFileName = '/kaggle/input/sharpe-ratio-based-portfolio-optimization/DJIA_Apr112014_Apr112019_kpf1.csv'

Rows = 1259  #excluding header

Columns = 15  #excluding date



#read stock prices 

df = pd.read_csv(StockFileName,  nrows= Rows)



#extract asset labels

assetLabels = df.columns[1:Columns+1].tolist()

print('Asset labels of k-portfolio 1: \n', assetLabels)



#read asset prices data

StockData = df.iloc[0:, 1:]



#compute asset returns

arStockPrices = np.asarray(StockData)

[Rows, Cols]=arStockPrices.shape

arReturns = StockReturnsComputing(arStockPrices, Rows, Cols)



#set precision for printing results

np.set_printoptions(precision=3, suppress = True)



#compute mean returns and variance covariance matrix of returns

meanReturns = np.mean(arReturns, axis = 0)

covReturns = np.cov(arReturns, rowvar=False)

print('\nMean Returns:\n', meanReturns)

print('\nVariance-Covariance Matrix of Returns:\n', covReturns)

from IPython.display import Image

Image( "/kaggle/input/sharpe-ratio-based-portfolio-optimization/Lesson6Eqn6_4.png")



#obtain maximal Sharpe Ratio for k-portfolio 1 of Dow stocks



#set portfolio size

portfolioSize = Columns



#set risk free asset rate of return

Rf=3  # April 2019 average risk  free rate of return in USA approx 3%

annRiskFreeRate = Rf/100



#compute daily risk free rate in percentage

r0 = (np.power((1 + annRiskFreeRate),  (1.0 / 360.0)) - 1.0) * 100 

print('\nRisk free rate (daily %): ', end="")

print ("{0:.3f}".format(r0)) 



#initialization

xOptimal =[]

minRiskPoint = []

expPortfolioReturnPoint =[]

maxSharpeRatio = 0



#compute maximal Sharpe Ratio and optimal weights

result = MaximizeSharpeRatioOptmzn(meanReturns, covReturns, r0, portfolioSize)

xOptimal.append(result.x)



    

#compute risk returns and max Sharpe Ratio of the optimal portfolio   

xOptimalArray = np.array(xOptimal)

Risk = np.matmul((np.matmul(xOptimalArray,covReturns)), np.transpose(xOptimalArray))

expReturn = np.matmul(np.array(meanReturns),xOptimalArray.T)

annRisk =   np.sqrt(Risk*251) 

annRet = 251*np.array(expReturn) 

maxSharpeRatio = (annRet-Rf)/annRisk 



#set precision for printing results

np.set_printoptions(precision=3, suppress = True)





#display results

print('Maximal Sharpe Ratio: ', maxSharpeRatio, '\nAnnualized Risk (%):  ', \

      annRisk, '\nAnnualized Expected Portfolio Return(%):  ', annRet)

print('\nOptimal weights (%):\n',  xOptimalArray.T*100 )
from IPython.display import Image

Image("/kaggle/input/sharpe-ratio-based-portfolio-optimization/Lesson6ExitTailImage.png")