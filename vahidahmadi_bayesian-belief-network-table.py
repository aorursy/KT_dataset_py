import pandas as pd

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv("../input/Assignment 2 Titanic Dataset.csv")
def condProbTable(a , given , data):

    if len(given)!=0:

        Given=list(given)

        Given.append(a)

        Probs=data.groupby(Given).size()/data.groupby(given).size()

        A = pd.DataFrame(Probs)

        Indexes = A.index

        C=pd.DataFrame()

        FirstColName='('

        for i in range(0,len( Indexes.names )-1):

            FirstColName=FirstColName+Indexes.names[i]+' '

            if(i==len(Indexes.names)-2):

                FirstColName=FirstColName+')'

            else:

                FirstColName=FirstColName+','

        C[FirstColName]=[Indexes[i][0:len(given)] for i in range(0,len(Indexes))]

        D=C.copy()

        C=C.drop_duplicates().reset_index(drop =True)

        aValues= [Indexes[i][len(given)] for i in range(0,len(Indexes))]

        for i in list(set(aValues)):

            C[a+' = '+str(i)] = 0.

        for i in range(0,len(C)):

            for j in D[D.iloc[:,0]==C.iloc[:,0][i]].index.values:

                x=list(C.iloc[i,0])

                x.append(aValues[j])

                x=tuple(x)

                C[a+' = '+str(aValues[j])][i]=A.loc[x]

    else:

        Probs=data.groupby(a).size()/len(data)

        C = pd.DataFrame(Probs)

        C.columns=['Probability']

        C.reset_index()    

    return C

condProbTable('survived',given= ['age'], data=df)
Ex2=condProbTable('survived',given= ['passenger_sex','ticket_price','age'], data=df)

Ex2
Ex3=condProbTable('age',given= [], data=df)

Ex3