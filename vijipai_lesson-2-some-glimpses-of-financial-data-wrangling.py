from IPython.display import Image

Image("/kaggle/input/some-glimpses-of-financial-data-wrangling/Lesson2GoalHeaderImage.png")

from IPython.display import Image

Image("/kaggle/input/some-glimpses-of-financial-data-wrangling/Lesson2Fig2_1.png")
#function to eliminate empty rows in a dataset

def EmptyRowsElimination(dfAssetPrices):



    # read dataset and extract its dimensions

    [Rows, Columns] = dfAssetPrices.shape

    dFrame = dfAssetPrices.iloc[0:Rows, 0:Columns]

    

    # call dropna method from Pandas 

    dFClean = dFrame.dropna(axis =0, how ='all')

    return dFClean
#empty rows elimination from stock prices dataset



#dependencies

import numpy as np

import pandas as pd



#input dataset and dimensions of the dataset

StockFileName = '/kaggle/input/some-glimpses-of-financial-data-wrangling/Lesson2FinDataWranglingSampledata.csv'

Rows = 12      #excluding headers

Columns = 18  #excluding date



#read stock prices 

df = pd.read_csv(StockFileName,  nrows= Rows)



#extract asset Names

assetNames = df.columns[1:Columns+1].tolist()

print(assetNames)



#clean the stock dataset of empty rows

StockData = df.iloc[0:, 1:]

dfClean = EmptyRowsElimination(StockData)

print('\nData cleaning completed!')

[rows, cols]=dfClean.shape

print('Dimensions of the cleaned dataset', dfClean.shape)

print('Cleaned dataset: \n', dfClean)
from IPython.display import Image

Image("/kaggle/input/some-glimpses-of-financial-data-wrangling/Lesson2Fig2_1.png")

#function to fill missing values of daily stock prices

#Mandatory requirements: (1) The dataset should have been cleaned of all empty rows 

#before missing values are filled, and 

#(2) the opening row of the dataset should not have any empty fields



def FillMissingValues(StockPrices):

    

    import numpy as np

    print('Fill missing values...')

    

    #identify positions of the missing values in StockPrices

    [rows, cols] = np.where(np.asarray(np.isnan(StockPrices)))

    

    #replace missing value with the previous day's price

    for t in range(rows.size):

        i=rows[t]

        j = cols[t]

        if (i-1) >= 0:           

            StockPrices.iloc[i,j]= StockPrices.iloc[i-1, j].copy()

        else:

            print('error')

    return StockPrices
#filling missing values of stock prices dataset



#dependencies

import numpy as np

import pandas as pd



#input dataset and the dimensions of the cleaned dataset

StockFileName = '/kaggle/input/some-glimpses-of-financial-data-wrangling/Lesson2MissingValBSE200.csv'

Rows = 11  #excluding header

Columns = 5  #excluding date



#read stock prices from the dataset as a pandas dataframe

df = pd.read_csv(StockFileName,  nrows= Rows)

StockData = df.iloc[0:, 1:]



#extract asset labels

assetLabels = df.columns[1:Columns+1].tolist()

print('Asset Labels:',assetLabels)



#impute missing data with previous day's trading price

stockDataClean = FillMissingValues(StockData)

print('Filling missing values completed!\n')

print(stockDataClean)

from IPython.display import Image

Image("/kaggle/input/some-glimpses-of-financial-data-wrangling/Lesson2ExitTailImage.png")