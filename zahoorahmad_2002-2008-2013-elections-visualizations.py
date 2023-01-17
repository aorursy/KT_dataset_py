import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import os
#print(os.listdir("../input"))
NA2013 = pd.read_csv("../input/National Assembly 2013.csv", encoding = "ISO-8859-1")
Winner13 = pd.DataFrame([])

for i in range(272):
    Const = "NA-%d"%(i+1)
    NA13 = NA2013.loc[NA2013['ConstituencyTitle']==Const, ['Party','Votes','TotalVotes']]
#    if NA.empty == True:
    if NA13.empty == False:
        MAX = (NA13.loc[NA13['Votes'].idxmax()])
        temp = pd.DataFrame([Const,MAX['Party'],MAX['Votes'],MAX['TotalVotes']]).T
        Winner13 = pd.concat([Winner13,temp])
Winner13 = Winner13.rename(columns = {0:'Constituency', 1:'Party', 2:'Votes', 3:'TotalVotes'})
Parties13 = np.unique(Winner13.loc[:,'Party'])
Num = pd.DataFrame([])

for i in range (len(Parties13)):
    temp = pd.DataFrame([Parties13[i], len(Winner13.loc[Winner13['Party'] == Parties13[i]])/len(Winner13) ,len(Winner13.loc[Winner13['Party'] == Parties13[i]])]).T
    Num = pd.concat([Num,temp])

Top_13 = Num.rename(columns = {0: 'Party', 1:'Percentage', 2:'Seats Won'})
Top_13 = Top_13.sort_values(by = 'Seats Won', ascending= False ) 

NA2008 = pd.read_csv("../input/National Assembly 2008.csv", encoding = "ISO-8859-1")
Winner08 = pd.DataFrame([])

for i in range(272):
    Const = "NA-%d"%(i+1)
    NA08 = NA2008.loc[NA2008['ConstituencyTitle']==Const, ['Party','Votes','TotalVotes']]
#    if NA.empty == True:
    if NA08.empty == False:
        MAX = (NA08.loc[NA08['Votes'].idxmax()])
        temp = pd.DataFrame([Const,MAX['Party'],MAX['Votes'],MAX['TotalVotes']]).T
        Winner08 = pd.concat([Winner08,temp])
Winner08 = Winner08.rename(columns = {0:'Constituency', 1:'Party', 2:'Votes', 3:'TotalVotes'})
Parties08 = np.unique(Winner08.loc[:,'Party'])
Num = pd.DataFrame([])
for i in range (len(Parties08)):
    temp = pd.DataFrame([Parties08[i], len(Winner08.loc[Winner08['Party'] == Parties08[i]])/len(Winner08) ,len(Winner08.loc[Winner08['Party'] == Parties08[i]])]).T
    Num = pd.concat([Num,temp])

Top_08 = Num.rename(columns = {0: 'Party', 1:'Percentage', 2:'Seats Won'})
Top_08 = Top_08.sort_values(by = 'Seats Won', ascending= False )

NA2002 = pd.read_csv("../input/National Assembly 2002 - Updated.csv", encoding = "ISO-8859-1")
Winner02 = pd.DataFrame([])

for i in range(272):
    Const = "NA-%d"%(i+1)
    NA02 = NA2002.loc[NA2002['Constituency_title']==Const, ['Party','Votes','TotalVotes']]
#    if NA.empty == True:
    if NA02.empty == False:
        MAX = (NA02.loc[NA02['Votes'].idxmax()])
        temp = pd.DataFrame([Const,MAX['Party'],MAX['Votes'],MAX['TotalVotes']]).T
        Winner02 = pd.concat([Winner02,temp])
Winner02 = Winner02.rename(columns = {0:'Constituency', 1:'Party', 2:'Votes', 3:'TotalVotes'})
Parties02 = np.unique(Winner02.loc[:,'Party'])
Num = pd.DataFrame([])
for i in range (len(Parties02)):
    temp = pd.DataFrame([Parties02[i], len(Winner02.loc[Winner02['Party'] == Parties02[i]])/len(Winner02) ,len(Winner02.loc[Winner02['Party'] == Parties02[i]])]).T
    Num = pd.concat([Num,temp])

Top_02 = Num.rename(columns = {0: 'Party', 1:'Percentage', 2:'Seats Won'})
Top_02 = Top_02.sort_values(by = 'Seats Won', ascending= False )
plt.figure(figsize=(5,5))
plt.title('Elections 2002')
labels02 = Top_02.loc[:,'Party']
values02 = Top_02.loc[:,'Percentage']
explode = (0.05, 0, 0, 0, 0 , 0 )  # explode 1st slice

plt.pie(values02[0:6], labels=labels02[0:6],explode=explode, shadow=True, startangle=90, autopct='%1.1f%%')
plt.show()

plt.figure(figsize=(5,5))
plt.title('Elections 2008')
labels08 = Top_08.loc[:,'Party']
values08 = Top_08.loc[:,'Percentage']


plt.pie(values08[0:6], labels=labels08[0:6], explode=explode, shadow=True, startangle=90, autopct='%1.1f%%')
plt.show()

plt.figure(figsize=(5,5))
plt.title('Elections 2013')
labels13 = Top_13.loc[:,'Party']
values13 = Top_13.loc[:,'Percentage']

plt.pie(values13[0:6], labels=labels13[0:6],explode=explode, shadow=True, startangle=90, autopct='%1.1f%%')
plt.show()
