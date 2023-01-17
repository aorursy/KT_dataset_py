import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import os
print(os.listdir("../input/election-data-wrangling"))
## 2002 Elections ## 
NA_All = pd.read_csv("../input/election-data-wrangling/NA2002-18.csv", encoding = "ISO-8859-1")
NA_Less = pd.DataFrame([])
NA_Less['Seat'] = NA_All['ConstituencyTitle'] + '-' + NA_All['Year'].astype(str) + '-' + NA_All['Seat']
NA_Less['Party'] = NA_All['Party']
NA_Less['Votes'] = NA_All['Votes']
NA_Less['Year'] = NA_All['Year']
CombinedDF = NA_Less
CombinedDF.shape
CombinedDF = CombinedDF.sort_values(by=['Seat','Year'])
CombinedDF.shape
CombinedDF2 = CombinedDF.groupby(['Seat', 'Party'])['Votes'].sum().reset_index()
#CombinedDF2
len(CombinedDF2['Seat'].unique())
#CombinedDF2.reset_index()
CombinedDF3 = CombinedDF2.set_index(['Seat','Party'])
print(CombinedDF3.index.names)
seat_names=CombinedDF3.index.levels[0]
party_names=CombinedDF3.index.levels[1]
#matrix = pd.DataFrame(index=seat_names,columns=party_names)
#matrix.shape
#matrix.iloc[0,0]=50
#matrix
matrix = np.zeros((len(seat_names),len(party_names)))
#CombinedDF3.loc[('NA-1-2002-PESHAWAR-I','Muttahidda Majlis-e-Amal Pakistan')].item()
for s in range(0,len(seat_names)):
    for p in range (0,len(party_names)):
        try:
            matrix[s,p] = CombinedDF3.loc[(seat_names[s], party_names[p])].item()
            #matrix.iloc[s,p] = CombinedDF3.loc[(seat_names[s], party_names[p])].item()
        except KeyError:
            continue
        #without loopup - very slow
        #matrix[s,p] = CombinedDF2.loc[(CombinedDF2['Seat'] == seat_names[s]) & (CombinedDF2['Party'] == party_names[p]), ['Votes']]
matrix2 = pd.DataFrame(matrix, index=seat_names, columns=party_names)
#matrix2 = matrix
#matrix2 = matrix2.fillna(0)
#matrix2 = matrix2.astype(int)
matrix2.style.background_gradient(cmap='summer',axis=1)