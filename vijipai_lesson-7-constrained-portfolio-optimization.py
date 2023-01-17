from IPython.display import Image

Image("/kaggle/input/constrained-portfolio-optimization/Lesson7GoalHeaderImage.png")

from IPython.display import Image

Image("/kaggle/input/constrained-portfolio-optimization/Lesson7Eqn7_1.png")
# function computes asset returns 

def StockReturnsComputing(StockPrice, Rows, Columns):

    

    import numpy as np

    

    StockReturn = np.zeros([Rows-1, Columns])

    for j in range(Columns):        # j: Assets

        for i in range(Rows-1):     # i: Daily Prices

            StockReturn[i,j]=((StockPrice[i+1, j]-StockPrice[i,j])/StockPrice[i,j])*100



    return StockReturn
#compute stock returns for k-portfolio 1 and market returns to compute asset betas



#Dependencies

import numpy as np

import pandas as pd





#input k portfolio 1 dataset  comprising 15 Dow stocks and DJIA market dataset 

#over a 3 Year period (April 2016 to April 2019)

stockFileName = '/kaggle/input/constrained-portfolio-optimization/DJIAkpf1Apr2016to20193YBeta.csv'

marketFileName = '/kaggle/input/constrained-portfolio-optimization/DJIAMarketDataApr2016to20193YBeta.csv'

stockRows = 756    #excluding header of stock dataset 

stockColumns = 15  #excluding date of stock dataset 

marketRows = 756   #excluding header of market dataset

marketColumns = 7  #excluding date of market dataset



#read stock prices and closing prices of market data (column index 4),  into dataframes

dfStock = pd.read_csv(stockFileName,  nrows= stockRows)

dfMarket = pd.read_csv(marketFileName, nrows = marketRows)

stockData = dfStock.iloc[0:, 1:]

marketData = dfMarket.iloc[0:, [4]] 



#extract asset labels in the portfolio

assetLabels = dfStock.columns[1:stockColumns+1].tolist()

print('Asset labels of k-portfolio 1: \n', assetLabels)



#compute asset returns

arStockPrices = np.asarray(stockData)

[sRows, sCols]=arStockPrices.shape

arStockReturns = StockReturnsComputing(arStockPrices, sRows, sCols)



#compute market returns

arMarketPrices = np.asarray(marketData)

[mRows, mCols]=arMarketPrices.shape

arMarketReturns = StockReturnsComputing(arMarketPrices, mRows, mCols)

#compute betas of the assets in k-portfolio 1

beta= []

Var = np.var(arMarketReturns, ddof =1)

for i in range(stockColumns):

    CovarMat = np.cov(arMarketReturns[:,0], arStockReturns[:, i ])

    Covar  = CovarMat[1,0]

    beta.append(Covar/Var)





#display results

print('Asset Betas:\n')

for data in beta:

    print('{:9.3f}'.format(data))
from IPython.display import Image

Image("/kaggle/input/constrained-portfolio-optimization/Lesson7Eqn7_2.png")
from IPython.display import Image

Image("/kaggle/input/constrained-portfolio-optimization/Lesson7Eqn7_3.png")
#obtain mean returns and variance-covariance matrix of returns of k-portfolio 1

#historical dataset: DJIA Index April 2014 to April 2019



#Dependencies

import numpy as np

import pandas as pd



#input k portfolio 1 dataset comprising 15 Dow stocks

StockFileName = '/kaggle/input/constrained-portfolio-optimization/DJIA_Apr112014_Apr112019_kpf1.csv'

Rows = 1259  #excluding header

Columns = 15  #excluding date



#read stock prices 

df = pd.read_csv(StockFileName,  nrows= Rows)



#extract asset labels

assetLabels = df.columns[1:Columns+1].tolist()

print('Asset labels for k-portfolio 1: \n', assetLabels)



#extract the asset prices data

stockData = df.iloc[0:, 1:]



#compute asset returns

arStockPrices = np.asarray(stockData)

[Rows, Cols]=arStockPrices.shape

arReturns = StockReturnsComputing(arStockPrices, Rows, Cols)



#set precision for printing data

np.set_printoptions(precision=3, suppress = True)



#compute mean returns and variance covariance matrix of returns

meanReturns = np.mean(arReturns, axis = 0)

covReturns = np.cov(arReturns, rowvar=False)

print('\nMean Returns:\n', meanReturns)

print('\nVariance-Covariance Matrix of Returns:\n', covReturns)



#function to handle bi-criterion portfolio optimization with constraints



#dependencies

import numpy as np

from scipy import optimize 



def BiCriterionFunctionOptmzn(MeanReturns, CovarReturns, RiskAversParam, PortfolioSize):

       

    def  f(x, MeanReturns, CovarReturns, RiskAversParam, PortfolioSize):

        PortfolioVariance = np.matmul(np.matmul(x, CovarReturns), x.T) 

        PortfolioExpReturn = np.matmul(np.array(MeanReturns),x.T)

        func = RiskAversParam * PortfolioVariance - (1-RiskAversParam)*PortfolioExpReturn

        return func



    def ConstraintEq(x):

        A=np.ones(x.shape)

        b=1

        constraintVal = np.matmul(A,x.T)-b 

        return constraintVal

    

    def ConstraintIneqUpBounds(x):

        A= [[0,0,0,0,0, 1,0,1,1,0, 1,1,1,1,1], [1,1,1,1,1,0,1,0,0,1,0,0,0,0,0]]

        bUpBounds =np.array([0.6,0.4]).T

        constraintValUpBounds = bUpBounds-np.matmul(A,x.T) 

        return constraintValUpBounds



    def ConstraintIneqLowBounds(x):

        A= [[0,0,0,0,0,1,0,1,1,0, 1,1,1,1,1], [1,1,1,1,1,0,1,0,0,1,0,0,0,0,0]]

        bLowBounds =np.array([0.01, 0.01]).T

        constraintValLowBounds = np.matmul(A,x.T)-bLowBounds  

        return constraintValLowBounds

    

    xinit=np.repeat(0.01, PortfolioSize)

    cons = ({'type': 'eq', 'fun':ConstraintEq}, \

            {'type':'ineq', 'fun': ConstraintIneqUpBounds},\

            {'type':'ineq', 'fun': ConstraintIneqLowBounds})

    bnds = [(0,0.1),(0,0.1), (0,0.1), (0,0.1), (0,0.1), (0,1), (0,0.1), (0,1),\

            (0,1), (0,0.1), (0,1),  (0,1),(0,1),(0,1),(0,1)]



    opt = optimize.minimize (f, x0 = xinit, args = ( MeanReturns, CovarReturns,\

                                                    RiskAversParam, PortfolioSize), \

                             method = 'SLSQP',  bounds = bnds, constraints = cons, \

                             tol = 10**-3)

    print(opt)

    return opt



#obtain optimal portfolios for the constrained portfolio optimization model

#Maximize returns and Minimize risk with fully invested, bound and 

#class constraints



#set portfolio size 

portfolioSize = Columns



#initialization

xOptimal =[]

minRiskPoint = []

expPortfolioReturnPoint =[]



for points in range(0,60):

    riskAversParam = points/60.0

    result = BiCriterionFunctionOptmzn(meanReturns, covReturns, riskAversParam, \

                                       portfolioSize)

    xOptimal.append(result.x)



#compute annualized risk and return  of the optimal portfolios for trading days = 251  

xOptimalArray = np.array(xOptimal)

minRiskPoint = np.diagonal(np.matmul((np.matmul(xOptimalArray,covReturns)),\

                                     np.transpose(xOptimalArray)))

riskPoint =   np.sqrt(minRiskPoint*251) 

expPortfolioReturnPoint= np.matmul(xOptimalArray, meanReturns )

retPoint = 251*np.array(expPortfolioReturnPoint) 



#set precision for printing results

np.set_printoptions(precision=3, suppress = True)



#display optimal portfolio results

print("Optimal weights of the efficient set portfolios\n:", xOptimalArray)

print("\nAnnualized Risk and Return of the efficient set portfolios:\n",\

      np.c_[riskPoint, retPoint])

import matplotlib.pyplot as plt



#Graph Efficient Frontier for the constrained portfolio model

NoPoints = riskPoint.size



colours = "blue"

area = np.pi*3



plt.title('Efficient Frontier for constrained k-portfolio 1 of Dow stocks')

plt.xlabel('Annualized Risk(%)')

plt.ylabel('Annualized Expected Portfolio Return(%)' )

plt.scatter(riskPoint, retPoint, s=area, c=colours, alpha =0.5)

plt.show()

from IPython.display import Image

Image("/kaggle/input/constrained-portfolio-optimization/Lesson7ExitTailImage.png")