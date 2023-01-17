from IPython.display import Image

Image("/kaggle/input/mean-variance-optimization-of-portfolios/Lesson5GoalHeaderImage.png")
from IPython.display import Image

Image("/kaggle/input/mean-variance-optimization-of-portfolios/Lesson5Eqn5_1.png")
from IPython.display import Image

Image("/kaggle/input/mean-variance-optimization-of-portfolios/Lesson5Eqn5_2.png")
from IPython.display import Image

Image("/kaggle/input/mean-variance-optimization-of-portfolios/Lesson5Eqn5_3.png")
from IPython.display import Image

Image("/kaggle/input/mean-variance-optimization-of-portfolios/Lesson5Eqn5_4.png")
#function obtains maximal return portfolio using linear programming



def MaximizeReturns(MeanReturns, PortfolioSize):

    

    #dependencies

    from scipy.optimize import linprog

    import numpy as np

    

    c = (np.multiply(-1, MeanReturns))

    A = np.ones([PortfolioSize,1]).T

    b=[1]

    res = linprog(c, A_ub = A, b_ub = b, bounds = (0,1), method = 'simplex') 

    

    return res
from IPython.display import Image

Image("/kaggle/input/mean-variance-optimization-of-portfolios/Lesson5Eqn5_5.png")
#function obtains minimal risk portfolio 



#dependencies

import numpy as np

from scipy import optimize 



def MinimizeRisk(CovarReturns, PortfolioSize):

    

    def  f(x, CovarReturns):

        func = np.matmul(np.matmul(x, CovarReturns), x.T) 

        return func



    def constraintEq(x):

        A=np.ones(x.shape)

        b=1

        constraintVal = np.matmul(A,x.T)-b 

        return constraintVal

    

    xinit=np.repeat(0.1, PortfolioSize)

    cons = ({'type': 'eq', 'fun':constraintEq})

    lb = 0

    ub = 1

    bnds = tuple([(lb,ub) for x in xinit])



    opt = optimize.minimize (f, x0 = xinit, args = (CovarReturns),  bounds = bnds, \

                             constraints = cons, tol = 10**-3)

    

    return opt

    
from IPython.display import Image

Image("/kaggle/input/mean-variance-optimization-of-portfolios/Lesson5Eqn5_6.png")
#function obtains Minimal risk and Maximum return portfolios



#dependencies

import numpy as np

from scipy import optimize 



def MinimizeRiskConstr(MeanReturns, CovarReturns, PortfolioSize, R):

    

    def  f(x,CovarReturns):

         

        func = np.matmul(np.matmul(x,CovarReturns ), x.T)

        return func



    def constraintEq(x):

        AEq=np.ones(x.shape)

        bEq=1

        EqconstraintVal = np.matmul(AEq,x.T)-bEq 

        return EqconstraintVal

    

    def constraintIneq(x, MeanReturns, R):

        AIneq = np.array(MeanReturns)

        bIneq = R

        IneqconstraintVal = np.matmul(AIneq,x.T) - bIneq

        return IneqconstraintVal

    



    xinit=np.repeat(0.1, PortfolioSize)

    cons = ({'type': 'eq', 'fun':constraintEq},

            {'type':'ineq', 'fun':constraintIneq, 'args':(MeanReturns,R) })

    lb = 0

    ub = 1

    bnds = tuple([(lb,ub) for x in xinit])



    opt = optimize.minimize (f, args = (CovarReturns), method ='trust-constr',  \

                        x0 = xinit,   bounds = bnds, constraints = cons, tol = 10**-3)

    

    return  opt

    

# function computes asset returns 

def StockReturnsComputing(StockPrice, Rows, Columns):

    

    import numpy as np

    

    StockReturn = np.zeros([Rows-1, Columns])

    for j in range(Columns):        # j: Assets

        for i in range(Rows-1):     # i: Daily Prices

            StockReturn[i,j]=((StockPrice[i+1, j]-StockPrice[i,j])/StockPrice[i,j])* 100



    return StockReturn
# Obtain optimal portfolio sets that maximize return and minimize risk



#Dependencies

import numpy as np

import pandas as pd





#input k-portfolio 1 dataset comprising 15 stocks

StockFileName = '/kaggle/input/mean-variance-optimization-of-portfolios/DJIA_Apr112014_Apr112019_kpf1.csv'



Rows = 1259  #excluding header

Columns = 15  #excluding date

portfolioSize = Columns #set portfolio size



#read stock prices in a dataframe

df = pd.read_csv(StockFileName,  nrows= Rows)



#extract asset labels

assetLabels = df.columns[1:Columns+1].tolist()

print(assetLabels)



#extract asset prices

StockData = df.iloc[0:, 1:]





#compute asset returns

arStockPrices = np.asarray(StockData)

[Rows, Cols]=arStockPrices.shape

arReturns = StockReturnsComputing(arStockPrices, Rows, Cols)





#compute mean returns and variance covariance matrix of returns

meanReturns = np.mean(arReturns, axis = 0)

covReturns = np.cov(arReturns, rowvar=False)



#set precision for printing results

np.set_printoptions(precision=3, suppress = True)



#display mean returns and variance-covariance matrix of returns

print('Mean returns of assets in k-portfolio 1\n', meanReturns)

print('Variance-Covariance matrix of returns\n', covReturns)

#Maximal expected portfolio return computation for the k-portfolio

result1 = MaximizeReturns(meanReturns, portfolioSize)

maxReturnWeights = result1.x

maxExpPortfolioReturn = np.matmul(meanReturns.T, maxReturnWeights)

print("Maximal Expected Portfolio Return:   %7.4f" % maxExpPortfolioReturn )

#expected portfolio return computation for the minimum risk k-portfolio 

result2 = MinimizeRisk(covReturns, portfolioSize)

minRiskWeights = result2.x

minRiskExpPortfolioReturn = np.matmul(meanReturns.T, minRiskWeights)

print("Expected Return of Minimum Risk Portfolio:  %7.4f" % minRiskExpPortfolioReturn)

#compute efficient set for the maximum return and minimum risk portfolios

increment = 0.001

low = minRiskExpPortfolioReturn

high = maxExpPortfolioReturn



#initialize optimal weight set and risk-return point set

xOptimal =[]

minRiskPoint = []

expPortfolioReturnPoint =[]



#repeated execution of function MinimizeRiskConstr to determine the efficient set 

while (low < high):

    

    result3 = MinimizeRiskConstr(meanReturns, covReturns, portfolioSize, low)

    xOptimal.append(result3.x)

    expPortfolioReturnPoint.append(low)

    low = low+increment

    

#gather optimal weight set    

xOptimalArray = np.array(xOptimal)



#obtain annualized risk for the efficient set portfolios 

#for trading days = 251

minRiskPoint = np.diagonal(np.matmul((np.matmul(xOptimalArray,covReturns)),\

                                     np.transpose(xOptimalArray)))

riskPoint =   np.sqrt(minRiskPoint*251) 



#obtain expected portfolio annualized return for the 

#efficient set portfolios, for trading days = 251

retPoint = 251*np.array(expPortfolioReturnPoint) 



#display efficient set portfolio parameters

print("Size of the  efficient set:", xOptimalArray.shape )

print("Optimal weights of the efficient set portfolios: \n", xOptimalArray)

print("Annualized Risk and Return of the efficient set portfolios: \n", \

                                                np.c_[riskPoint, retPoint])
#Graph Efficient Frontier

import matplotlib.pyplot as plt



NoPoints = riskPoint.size



colours = "blue"

area = np.pi*3



plt.title('Efficient Frontier for k-portfolio 1 of Dow stocks')

plt.xlabel('Annualized Risk(%)')

plt.ylabel('Annualized Expected Portfolio Return(%)' )

plt.scatter(riskPoint, retPoint, s=area, c=colours, alpha =0.5)

plt.show()



from IPython.display import Image

Image("/kaggle/input/mean-variance-optimization-of-portfolios/Lesson5Fig5_1.png")

from IPython.display import Image

Image("/kaggle/input/mean-variance-optimization-of-portfolios/Lesson5Fig5_2.png")

from IPython.display import Image

Image("/kaggle/input/mean-variance-optimization-of-portfolios/Lesson5ExitTailImage.png")