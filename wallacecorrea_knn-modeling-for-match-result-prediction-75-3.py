#importing necessary libraries



import pandas as pd

from sklearn.preprocessing import MaxAbsScaler

from sklearn.neighbors import KNeighborsClassifier



df = pd.read_csv('../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')



print ('Importing and reading done successfully!')
#checking for nan



nan_column_count = 0

for column in df.isna().sum():

    if column>0:

        print(column)

        nan_column_count+=1

if nan_column_count == 0:

    print('No missing values in your dataset!')

    

#checking dtypes and listing features

print (df.dtypes)
#check blueWins distribution

print (df.blueWins.value_counts())

#creating new features



df['WardPlaceDiff']=df['blueWardsPlaced']-df['redWardsPlaced']

df['WardDestroyDiff']=df['blueWardsDestroyed']-df['redWardsDestroyed']

df['FirstBloodDiff']=df['blueFirstBlood']-df['redFirstBlood']

df['KillDiff']=df['blueKills']-df['redKills']

df['DeathDiff']=df['blueDeaths']-df['redDeaths']

df['AssistDiff']=df['blueAssists']-df['redAssists']

df['EliteMonsterDiff']=df['blueEliteMonsters']-df['redEliteMonsters']

df['DragonDiff']=df['blueDragons']-df['redDragons']

df['HeraldDiff']=df['blueHeralds']-df['redHeralds']

df['TowerDestroyDiff']=df['blueTowersDestroyed']-df['redTowersDestroyed']

df['AvgLevelDiff']=df['blueAvgLevel']-df['redAvgLevel']

df['MinionsDiff']=df['blueTotalMinionsKilled']-df['redTotalMinionsKilled']

df['JungleMinionsDiff']=df['blueTotalJungleMinionsKilled']-df['redTotalJungleMinionsKilled']

df['CSdiff']=df['blueCSPerMin']-df['redCSPerMin']

df['GPMdiff']=df['blueGoldPerMin']-df['redGoldPerMin']



#selecting relevant features



relevant=[

          'blueWins',

          'WardPlaceDiff',

          'WardDestroyDiff',

          'FirstBloodDiff',

          'KillDiff',

          'DeathDiff',

          'AssistDiff',

          'EliteMonsterDiff',

          'DragonDiff',

          'HeraldDiff',

          'TowerDestroyDiff',

          'AvgLevelDiff',

          'MinionsDiff',

          'JungleMinionsDiff',

          'blueGoldDiff',

          'blueExperienceDiff',

          'CSdiff',

          'GPMdiff'

           ]



print ('Step saved successfully!')
dados = df[relevant]



scaler2 = MaxAbsScaler()

scaler2.fit(dados)

analisedados=scaler2.transform(dados)

analisedf = pd.DataFrame(data=analisedados)

print (pd.DataFrame(data=analisedados).groupby(by=0).mean().T)

#getting the subset of our elected features and randomizing it using a seed to get reproductable results



dados = df[relevant]



dados_embaralhados=dados.sample(frac=1, random_state = 4234)



#splitting the target column out of the dataframe



x = dados_embaralhados.loc[:,dados_embaralhados.columns!='blueWins'].values

y = dados_embaralhados.loc[:,dados_embaralhados.columns=='blueWins'].values



#defining our training sample size and splitting our data



q = 7750



x_treino = x[:q,:]

y_treino = y[:q].ravel()



x_teste = x[q:,:]

y_teste = y[q:].ravel()



#scaling the features



scaler = MaxAbsScaler()

scaler.fit(x_treino)



x_treino = scaler.transform(x_treino)

x_teste = scaler.transform(x_teste)



print ('Step saved successfully!')
print ( "\n  K TRAINING  TEST")

print ( " -- ------ ------")



for k in range(40,60):



    classificador = KNeighborsClassifier(

        n_neighbors = k,

        weights     = 'uniform',

        p           = 1

        )

    classificador = classificador.fit(x_treino,y_treino)



    y_resposta_treino = classificador.predict(x_treino)

    y_resposta_teste  = classificador.predict(x_teste)

    

    acuracia_treino = sum(y_resposta_treino==y_treino)/len(y_treino)

    acuracia_teste  = sum(y_resposta_teste ==y_teste) /len(y_teste)

    

    print(

        "%3d"%k,

        "%6.1f" % (100*acuracia_treino),

        "%6.1f" % (100*acuracia_teste)

        )

    

    
classificador = KNeighborsClassifier(

    n_neighbors = 58,

    weights     = 'uniform',

     p           = 1

    )

classificador = classificador.fit(x_treino,y_treino)



y_resposta_treino = classificador.predict(x_treino)

y_resposta_teste  = classificador.predict(x_teste)

    

acuracia_treino = sum(y_resposta_treino==y_treino)/len(y_treino)

acuracia_teste  = sum(y_resposta_teste ==y_teste) /len(y_teste)

    

print(

        "%3d"%k,

        "%6.1f" % (100*acuracia_treino),

        "%6.1f" % (100*acuracia_teste)

        )
relevant=['blueWins',

          # 'WardPlaceDiff',

          # 'WardDestroyDiff',

          # 'FirstBloodDiff',

          'KillDiff',

          'DeathDiff',

          # 'AssistDiff',

          'EliteMonsterDiff',

          'DragonDiff',

          # 'HeraldDiff',

          # 'TowerDestroyDiff',

          'AvgLevelDiff',

          'MinionsDiff',

          #'JungleMinionsDiff',

          'blueGoldDiff',

          'blueExperienceDiff',

          'CSdiff',

          'GPMdiff'

          ]

print ('Step saved successfully!')
#getting the subset of our elected features and randomizing it using a seed to get reproductable results



dados = df[relevant]



dados_embaralhados=dados.sample(frac=1, random_state = 4234)



#splitting the target column out of the dataframe



x = dados_embaralhados.loc[:,dados_embaralhados.columns!='blueWins'].values

y = dados_embaralhados.loc[:,dados_embaralhados.columns=='blueWins'].values



#defining our training sample size and splitting our data



q = 7750



x_treino = x[:q,:]

y_treino = y[:q].ravel()



x_teste = x[q:,:]

y_teste = y[q:].ravel()



#scaling the features



scaler = MaxAbsScaler()

scaler.fit(x_treino)



x_treino = scaler.transform(x_treino)

x_teste = scaler.transform(x_teste)



print ('Step saved successfully!')
print ( "\n  K TRAINING  TEST")

print ( " -- ------ ------")



for k in range(40,60):



    classificador = KNeighborsClassifier(

        n_neighbors = k,

        weights     = 'uniform',

        p           = 1

        )

    classificador = classificador.fit(x_treino,y_treino)



    y_resposta_treino = classificador.predict(x_treino)

    y_resposta_teste  = classificador.predict(x_teste)

    

    acuracia_treino = sum(y_resposta_treino==y_treino)/len(y_treino)

    acuracia_teste  = sum(y_resposta_teste ==y_teste) /len(y_teste)

    

    print(

        "%3d"%k,

        "%6.1f" % (100*acuracia_treino),

        "%6.1f" % (100*acuracia_teste)

        )

    
classificador = KNeighborsClassifier(

    n_neighbors = 48,

    weights     = 'uniform',

     p           = 1

    )

classificador = classificador.fit(x_treino,y_treino)



y_resposta_treino = classificador.predict(x_treino)

y_resposta_teste  = classificador.predict(x_teste)

    

acuracia_treino = sum(y_resposta_treino==y_treino)/len(y_treino)

acuracia_teste  = sum(y_resposta_teste ==y_teste) /len(y_teste)

    

print(

        'accuracy:',

        "%6.1f" % (100*acuracia_teste)

        )