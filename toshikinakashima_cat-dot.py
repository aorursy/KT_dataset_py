import numpy as np

import pandas as pd

import csv

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import cross_val_score

from category_encoders  import OneHotEncoder, OrdinalEncoder



if (0==1):

    train= pd.read_csv("../input/cat-in-the-dat-ii/train.csv")

    test= pd.read_csv("../input/cat-in-the-dat-ii/test.csv")



    y_train= train["target"].values

    train = train.drop(["target"], axis=1)

    

    print("loaded")

    map_ord_1 = {'Novice':1, 

            'Contributor':2, 

            'Expert':3, 

            'Master':4, 

            'Grandmaster':5}

    map_ord_2 = {'Freezing':1, 

            'Cold':2, 

            'Warm':3, 

            'Hot':4, 

            'Boiling Hot':5, 

            'Lava Hot':6}

    

    train.fillna(train.mode().iloc[0], inplace=True)

    test.fillna(test.mode().iloc[0], inplace=True)

    

    oe = OrdinalEncoder(cols=['bin_3','bin_4'],handle_unknown='impute')

    train = oe.fit_transform(train)

    oh = OneHotEncoder(cols=['nom_0','nom_1','nom_2','nom_3','nom_4'],handle_unknown='impute')

    train = oh.fit_transform(train)

    train.drop(['nom_0_1','nom_1_1','nom_2_1','nom_3_1','nom_4_1'], axis=1, inplace=True)

    train.ord_1 = train.ord_1.map(map_ord_1)

    train.ord_2 = train.ord_2.map(map_ord_2)

    train['ord_3'].fillna('0', inplace=True)

    train['ord_3']=train['ord_3'].apply(lambda x: ord(x)-ord('a'))

    train['ord_4'].fillna('0', inplace=True)

    train['ord_4']=train['ord_4'].apply(lambda x: ord(x)-ord('A'))

    train['ord_5f'] = train['ord_5'].str[0]

    train['ord_5f'].fillna('`', inplace=True)

    train['ord_5f']=train['ord_5f'].apply(lambda x: ord(x)-ord('A'))

    train.drop('ord_5', axis=1, inplace=True)

    train[['nom_5','nom_6','nom_7','nom_8','nom_9']] = train[['nom_5','nom_6','nom_7','nom_8','nom_9']].applymap(lambda x:int(x,16))



    train.to_csv("train_labeled.csv")

    

    test = oe.transform(test)

    test = oh.transform(test)

    test.drop(['nom_0_1','nom_1_1','nom_2_1','nom_3_1','nom_4_1'], axis=1, inplace=True)

    test.ord_1 = test.ord_1.map(map_ord_1)

    test.ord_2 = test.ord_2.map(map_ord_2)

    test['ord_3'].fillna('0', inplace=True)

    test['ord_3']=test['ord_3'].apply(lambda x: ord(x)-ord('a'))

    test['ord_4'].fillna('0', inplace=True)

    test['ord_4']=test['ord_4'].apply(lambda x: ord(x)-ord('A'))

    test['ord_5f'] = test['ord_5'].str[0]

    test['ord_5f'].fillna('`', inplace=True)

    test['ord_5f']=test['ord_5f'].apply(lambda x: ord(x)-ord('A'))

    test.drop('ord_5', axis=1, inplace=True)

    test[['nom_5','nom_6','nom_7','nom_8','nom_9']] = test[['nom_5','nom_6','nom_7','nom_8','nom_9']].applymap(lambda x:int(x,16))





    test.to_csv("test_labeled.csv")

    

else:

    train= pd.read_csv("train_labeled.csv")

    test= pd.read_csv("test_labeled.csv")



#train.drop(['nom_5','nom_6','nom_7','nom_8','nom_9'], axis=1, inplace=True)

#test.drop(['nom_5','nom_6','nom_7','nom_8','nom_9'], axis=1, inplace=True)





x_train = train.drop(["id"], axis=1).values

x_test = test.drop(["id"], axis=1).values



print(x_train.shape)



estimators =[

    ('rf', RandomForestClassifier(n_estimators=30, max_depth=8)),

    ('et', ExtraTreesClassifier(n_estimators=30, max_depth=8)),

    ('ld', LinearDiscriminantAnalysis())

]

#clf = VotingClassifier(estimators)

#clf = RandomForestClassifier()

#clf = GradientBoostingClassifier(n_estimators=30, learning_rate=0.1)

clf = LinearDiscriminantAnalysis()

#clf=LogisticRegression(C=0.123456789, solver="lbfgs", max_iter=50)

#clf = DecisionTreeClassifier()

#clf = MLPClassifier(max_iter=10000, activation='relu', solver='adam', alpha=0.0001, early_stopping=True, verbose=1)



scores = cross_val_score(clf, x_train, y_train, cv=3, verbose=1, scoring='roc_auc')

print(np.mean(scores))



clf.fit(x_train, y_train)



if (1==1):

    pred_test = clf.predict(x_test)

    #print(pred_test[:100])

    with open("predicted_data.csv", "w") as f:

        writer = csv.writer(f, lineterminator='\n')

        writer.writerow(["id", "terget"])

        for pid, survived in zip(test['id'].astype(int), pred_test.astype(int)):

            writer.writerow([pid, survived])



print("finished")
import numpy as np

import pandas as pd

import csv

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import OrdinalEncoder



train= pd.read_csv("../input/cat-in-the-dat-ii/train.csv")

test= pd.read_csv("../input/cat-in-the-dat-ii/test.csv")



print(train.head(10))



#test.fillna(test.mode().iloc[0], inplace=True)

map_ord_1 = {'Novice':1, 

            'Contributor':2, 

            'Expert':4, 

            'Master':5, 

            'Grandmaster':6}

train.ord_1 = train.ord_1.map(map_ord_1)

map_ord_2 = {'Freezing':1, 

            'Cold':2, 

            'Warm':3, 

            'Hot':4, 

            'Boiling Hot':5, 

            'Lava Hot':6}

train.ord_2 = train.ord_2.map(map_ord_2)

train['ord_3'].fillna('0', inplace=True)

train['ord_3']=train['ord_3'].apply(lambda x: ord(x)-ord('a'))

train['ord_4'].fillna('0', inplace=True)

train['ord_4']=train['ord_4'].apply(lambda x: ord(x)-ord('A'))

train['ord_5f'] = train['ord_5'].str[0]

train['ord_5b'] = train['ord_5'].str[1]

train['ord_5f'].fillna('`', inplace=True)

train['ord_5b'].fillna('`', inplace=True)

train['ord_5f']=train['ord_5f'].apply(lambda x: ord(x)-ord('A'))

train['ord_5b']=train['ord_5b'].apply(lambda x: ord(x)-ord('A'))





train.fillna(0, inplace=True)

print(train.head(10))



g = sns.catplot(x="ord_5b", y="target",  data=train, height=6, kind="bar", palette="muted")

#train[['']]

#train['nom_5'] = train['nom_5'].str.lower()

train['nom_6f'] = train['nom_6'].str[8].int(16)

#train[['ord_1','ord_2','ord_3','ord_4','ord_5']] = oe.fit_transform(train[['ord_1','ord_2','ord_3','ord_4','ord_5']].values)





#g.despine(left=True)

#g = g.set_ylabels("survival probability")



'''

for col in ['bin_3','bin_4','nom_0','nom_1','nom_2','nom_3','nom_4','ord_1','ord_2','ord_3','ord_4','ord_5']:

    train_vals = set(train[col].unique())

    test_vals  = set(test[col].unique())

   

    xor_cat_vals=train_vals ^ test_vals

    print(col, xor_cat_vals)

'''

        
import numpy as np

import pandas as pd

from sklearn import preprocessing as pp

import category_encoders as ce

train= pd.read_csv("../input/cat-in-the-dat-ii/train.csv")

train=train.head(5)

    

train.fillna(train.mode().iloc[0], inplace=True)



print(train["nom_5"])

a = train[['nom_5','nom_6','nom_7','nom_8','nom_9']].applymap(lambda x:list("{:0=36b}".format(int(x,16))))

a = a['nom_5'].apply(pd.Series)

#train[['nom_5']] = train[['nom_5']].applymap(lambda x:int(x,16))

train[[str(i) for i in range[36]]]=a



print(train)

g = sns.catplot(x="ord_5b", y="target",  data=a, height=6, kind="bar", palette="muted")



print(a)

#print(df_session_ce_onehot[["nom_1"]])