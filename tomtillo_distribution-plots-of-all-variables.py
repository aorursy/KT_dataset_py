import numpy as np  

import pandas as pd  



import matplotlib.pyplot as plt # For plots

#import matplotlib.pyplot.figure as fig



#import seaborn as sns
train = pd.read_csv('../input/train.csv')

no_of_columns = len(train.columns)

print("No of columns = " , no_of_columns)
plt.rcParams["figure.figsize"] = [2,2]  #set the graph to a smaller size 

#no_of_columns = 5                # in-case you want to try with a smaller subset of graphs to save execution time / CPU consumption

for i in range(1,no_of_columns) :

    col_name = train.columns[i]  # get the column name to identify the graph

    x=train[col_name]

    plt.xlabel(col_name)         # The column name will be the x-label ( just to identify )

    k=plt.hist(x,bins=100)       # Change the right number of bins to get it more/less granular 

    plt.show()


