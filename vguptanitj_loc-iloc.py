import pandas as pd
df=pd.read_csv("../input/brics.csv",index_col=0)
df
df.iloc[2]
df.loc["IN"]
df.iloc[[2]]
df.loc[["IN"]]
df.iloc[[0,2,4]]
df.loc[["BR","IN","SA"]]
df.iloc[2:4,1:3]                            # DataFrame_name.iloc[ start_row : end_row+1 , start_column : end_column+1 ]
df.loc["IN":"SA","capital":"population"]    # DataFrame_name.loc[ start_row : end_row , start_column : end_column ]
df.iloc[[0,2,4],[1,3]]
df.loc[["BR","IN","SA"],["capital","population"]]
df.iloc[:,1:3]
df.loc[:,"capital":"area"]
df.iloc[1:3,:]
df.loc["RU":"IN",:]