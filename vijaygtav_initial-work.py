# THis is a simple SGD classifier example

# for details of SGD go to http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier

import numpy as np

from sklearn import linear_model

import pandas as pd

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])

Y = np.array([1, 1, 2, 2])

linear_model.SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,

        eta0=0.0, fit_intercept=True, l1_ratio=0.15,

        learning_rate='optimal', loss='log', n_iter=5, n_jobs=1,

        penalty='l2', power_t=0.5, random_state=None, shuffle=True,

        verbose=0, warm_start=False)



clf.fit(X, Y)



print(clf.predict([[0.8, 1]]))

print(clf.predict([[-0.8, -1]]))
#This cell proves that all displayid in ClickTrain are in Events

df_ct = pd.read_csv('~/outbrain_data/clicks_train.csv',usecols=['display_id'] )

df_events = pd.read_csv('~/outbrain_data/events.csv' ,usecols=['display_id']  )



X= pd.DataFrame(df_ct.display_id.unique())

Y= pd.DataFrame(df_events.display_id.unique())

Z= pd.DataFrame(X.isin(Y))

Z[0].unique()

#This is a code from someone on Kaggle for a weak learner  to make prediction and create file in format for kaggle submission

#The algorithm doesnt does not use any standard ML algorithm, just how often a particular ad was clicked

import pandas as pd

import numpy as np



dtypes = {'ad_id': np.float32, 'clicked': np.int8}



train = pd.read_csv("~/outbrain_data/clicks_train.csv", usecols=['ad_id','clicked'], dtype=dtypes )



ad_likelihood = train.groupby('ad_id').clicked.agg(['count','sum','mean']).reset_index()

M = train.clicked.mean()

print(M)

del train



ad_likelihood['likelihood'] = (ad_likelihood['sum'] + 12*M) / (12 + ad_likelihood['count'])



test = pd.read_csv("~/outbrain_data/clicks_test.csv")

test = test.merge(ad_likelihood, how='left')

test.likelihood.fillna(M, inplace=True)



test.sort_values(['display_id','likelihood'], inplace=True, ascending=False)

subm = test.groupby('display_id').ad_id.apply(lambda x: " ".join(map(str,x))).reset_index()



subm.to_csv("subm.csv", index=False)
import numpy as np

import gc

from sklearn import linear_model

import pandas as pd

pc=pd.read_csv('~/outbrain_data/promoted_content.csv')

df_ct = pd.read_csv('~/outbrain_data/clicks_train.csv')

print(df_ct.size)

print((pd.DataFrame(pc['document_id'].unique())).count())

print(pc.size)

M = df_ct.clicked.mean()

pc.groupby('document_id',as_index=False).count()['advertiser_id'].unique()



df_mrg=df_ct.merge(pc,on='ad_id' ,how='left')



df_cmpg= df_mrg.groupby('campaign_id').clicked.agg(['count' ,'sum']).reset_index()

df_cmpg['cmpg_Score']= (df_cmpg['sum'] + M) / (1 + df_cmpg['count'])

df_cmpg=df_cmpg.drop('count',1)

df_cmpg=df_cmpg.drop('sum',1)



df_adv= df_mrg.groupby('advertiser_id').clicked.agg(['count' ,'sum']).reset_index()

df_adv['adv_Score']= (df_adv['sum'] + M) / (1 + df_adv['count'])

df_adv=df_adv.drop('count',1)

df_adv=df_adv.drop('sum',1)

df_adv



pc=pc.merge(df_adv).merge(df_cmpg)

pc=pc.drop('campaign_id',1)

pc=pc.drop('advertiser_id',1)

pc=pc.drop('document_id',1)



X=df_ct.merge(pc)



Y=np.array(X['clicked'])

X= X.drop('clicked',1)

X= X.drop('display_id',1)

X= X.drop('ad_id',1)

X=np.array (X)



from sklearn import datasets

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

import numpy as np

import pandas as pd

from sklearn.cross_validation import KFold

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

%matplotlib inline



X_test= pd.read_csv('~/outbrain_data/clicks_test.csv' )

X_test=X_test.merge(pc,how='left')

adv_score_mean = pc.adv_Score.mean()

cmpg_score_mean = pc.cmpg_Score.mean()

X_test.adv_Score.fillna(adv_score_mean, inplace=True)

X_test.cmpg_Score.fillna(cmpg_score_mean, inplace=True)

X_test= X_test.drop('display_id',1)

X_test= X_test.drop('ad_id',1)

X_test=np.array (X_test)



reg=SGDClassifier(loss= 'log', penalty= 'l2')

reg.fit(X,Y)

a=accuracy_score(Y, reg.predict(X))

print(a)

output= reg.decision_function(X_test)

plt.hist(output)   

plt.show()



output=pd.read_csv('~/outbrain_data/clicks_test.csv').merge(pd.DataFrame(output,columns=['prediction']), left_index=True,right_index=True)

output.sort_values(['display_id','prediction'], inplace=True, ascending=False)

subm = output.groupby('display_id').ad_id.apply(lambda x: " ".join(map(str,x))).reset_index()



subm.to_csv("subm_22_11_2016_21_45.csv", index=False)
print(reg.coef_)

print(reg.intercept_)
output= reg.decision_function(X_test)



output=pd.read_csv('clicks_test.csv').merge(pd.DataFrame(output,columns=['prediction']), left_index=True,right_index=True)

output.sort_values(['display_id','prediction'], inplace=True, ascending=False)

subm = output.groupby('display_id').ad_id.apply(lambda x: " ".join(map(str,x))).reset_index()

subm.to_csv("subm2.csv", index=False)


X_test_orig= pd.read_csv('clicks_test.csv' )

print(X_test_orig.size)



X_test=X_test_orig.join(pc ,how='left',lsuffix='_left',rsuffix='_right')

print(X_test.size)



X_test= X_test.drop('display_id',1)

X_test=X_test.drop('ad_id_left',1)

X_test=X_test.drop('ad_id_right',1)

print(X_test.size)



X_test=np.array (X_test)



output= reg.decision_function(X_test)

output=pd.DataFrame(output,columns=['prediction'])

print(output.size)





output2=X_test_orig.join(output)

print(output2.size)
numFolds = 10

kf = KFold(len(X), numFolds, shuffle=True)



# These are "Class objects". For each Class, find the AUC through

# 10 fold cross validation.

Models = [LogisticRegression, SGDClassifier]

params = [{}, {"loss": "log", "penalty": "l2"}]

for param, Model in zip(params, Models):

    total = 0

    for train_indices, test_indices in kf:



        train_X = X[train_indices, :]; train_Y = Y[train_indices]

        test_X = X[test_indices, :]; test_Y = Y[test_indices]



        reg = Model(**param)

        reg.fit(train_X, train_Y)

        predictions = reg.predict(test_X)

        total += accuracy_score(test_Y, predictions)

    accuracy = total / numFolds

    print ("Accuracy score of {0}: {1}".format(Model.__name__, accuracy))
