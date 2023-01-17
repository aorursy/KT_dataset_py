# Basics of Series 
import pandas as pd
data = ['d1','d2','d3','d4']
series = pd.Series(data)
series
sCustom = pd.Series(data,index = [1,3,2,1])
sCustom