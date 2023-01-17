import numpy as np 



import pandas as pd 



import os



import seaborn as sns



import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import log_loss

from sklearn.metrics import confusion_matrix



import time



from lightgbm import LGBMClassifier



import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv("../input/bioresponse/train.csv")

print(data.head(10))
print("Missing Values",data.isnull().sum().sum())

print("Column names",data.columns)
# select the float columns

df_float = data.select_dtypes(include=[np.float])

print("Float Columns",df_float.columns)



# select int columns

df_int = data.select_dtypes(include=[np.int])

print("Int columns",df_int.columns)



# select object columns

df_object = data.select_dtypes(include=[object])

print("object columns",df_object.columns)
g1 = sns.countplot(x=data["Activity"])

    



g1.set_xticklabels(g1.get_xticklabels(),rotation=45)

plt.show()
Scaleddata = MinMaxScaler().fit_transform(data)

data = pd.DataFrame(Scaleddata,columns=data.columns)
pca = PCA().fit(data)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance');
# Reduce dimensionality through PCA



y = data["Activity"]

X = data.drop(columns=['Activity'])



pca = PCA(n_components=750)

X_pca = pca.fit_transform(X) 





# Then reduce further with t-sne

tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000, learning_rate=200)

tsne_results = tsne.fit_transform(X_pca[:,:])



df_tsne = pd.DataFrame(tsne_results, columns=['comp1', 'comp2'])

df_tsne['label'] = y[:]

sns.scatterplot(x='comp1', y='comp2', data=df_tsne, hue='label')
train = pd.read_csv("../input/bioresponse/train.csv")

test = pd.read_csv("../input/bioresponse/test.csv")





# Label for predicting a strong adherence to the testing sets

train['label'] = 0

test['label'] = 1



training = train.drop('Activity',axis=1) 



# Combine testing and training sets

combine = training.append(test)

y =combine['label']

combine.drop(columns = ['label'],inplace=True)





model = LGBMClassifier(n_estimators = 50)

drop_list = []



for col in combine.columns:

    score = cross_val_score(model,pd.DataFrame(combine[col]),y,cv=2,scoring='roc_auc')

    #print(score)

    if (np.mean(score) > 0.7):

        drop_list.append(col)

        print("Column with covariate shift:", col)
if len(drop_list)==0: 

    print("No presence of covariate shift")
data = pd.read_csv("../input/bioresponse/train.csv")



X = data

X = X.drop(columns='Activity')



y = data['Activity']

y = y.values

y = y.reshape((len(y), 1))



# split into train and test sets

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=32)



columns  = X_train.columns



# Normalizing Column values

for col in columns:

    MinMax = MinMaxScaler()

    

    X_train_arr = X_train[col].astype(float).values

    X_test_arr = X_test[col].astype(float).values   

        

            

    X_train_arr = MinMax.fit_transform(X_train_arr.reshape(-1,1))

    X_test_arr = MinMax.transform(X_test_arr.reshape(-1,1))

            

    X_train[col]  = X_train_arr 

    X_test[col]   = X_test_arr

        

#fit_params={"early_stopping_rounds":100, 

           # "eval_metric" : 'binary_logloss', 

          #"eval_set" : [(X_test,y_test)],

          #'eval_names': ['valid'],

          #'verbose': 100,

          #'categorical_feature': 'auto'}



#param_test ={  'n_estimators': [50,100,200,400, 700, 1000],

  #'colsample_bytree': [0.7, 0.8],

  #'max_depth': [15,20,25],

  #'num_leaves': [50, 100, 200],

  #'reg_alpha': [1.1, 1.2, 1.3],

 #'reg_lambda': [1.1, 1.2, 1.3],

# 'min_split_gain': [0.3, 0.4],

#'subsample': [0.7, 0.8, 0.9],

 # 'subsample_freq': [20]}
#clf = LGBMClassifier(random_state=314, silent=True, metric='None', n_jobs=2)

#model = RandomizedSearchCV(

   #estimator=clf, param_distributions=param_test, 

   # scoring='neg_log_loss',

   # cv=3,

   # refit=True,

  # random_state=314,

  # verbose=True)
#model.fit(X_train, y_train, **fit_params)

#print('Best score reached: {} with params: {} '.format(model.best_score_, model.best_params_))
model = LGBMClassifier(random_state=314, silent=True, n_jobs=2,subsample_freq = 20, subsample = 0.9, 

                       reg_lambda = 1.2, reg_alpha = 1.1,num_leaves= 200, n_estimators = 700, 

                       min_split_gain =  0.4, max_depth =  15, colsample_bytree = 0.8)
model.fit(X_train, y_train)

pred = model.predict_proba(X_test)



print("Log Loss Probability: ",log_loss(y_test,pred))
pred = model.predict(X_test)

y_test = y_test.flatten()



#print(np.shape(y_test))



data = {'y_Actual':    y_test,

        'y_Predicted': pred

        }



df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])

confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])



sns.heatmap(confusion_matrix, annot=True)

plt.show()
def plotImp(model, X , num = 20):

    

    feature_imp = pd.DataFrame({'Value':model.feature_importances_,'Feature':X.columns})

    plt.figure(figsize=(100, 500))    

    sns.set(font_scale = 5)

    

    #columns = feature_imp.sort_values(by="Value",ascending=False)[0:num]['Feature'].to_list()

    

    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", 

                                                        ascending=False)[0:num])

    plt.title('LightGBM Feature imprtances')

    plt.tight_layout()

    

    plt.show()

    

    

plotImp(model, X_train , num =300)

X_train = pd.read_csv("../input/bioresponse/train.csv")

X_test =  pd.read_csv("../input/bioresponse/test.csv")



y_train = X_train["Activity"]



X_train.drop(columns=["Activity"],inplace=True)





columns  = X_train.columns



# Normalizing column values

for col in columns:

    MinMax = MinMaxScaler()

    

    X_train_arr = X_train[col].astype(float).values

    X_test_arr = X_test[col].astype(float).values   

        

            

    X_train_arr = MinMax.fit_transform(X_train_arr.reshape(-1,1))

    X_test_arr = MinMax.transform(X_test_arr.reshape(-1,1))

            

    X_train[col]  = X_train_arr 

    X_test[col]   = X_test_arr
model = LGBMClassifier(random_state=314, silent=True, n_jobs=2,subsample_freq = 20, subsample = 0.9, 

                       reg_lambda = 1.2, reg_alpha = 1.1,num_leaves= 200, n_estimators = 700, 

                       min_split_gain =  0.4, max_depth =  15, colsample_bytree = 0.8)
model.fit(X_train,y_train)
predicted_prob = model.predict_proba(X_test)
Probability = predicted_prob[:,1]

MoleculeId = np.array(range(1,len(X_test)+1))
submission = pd.DataFrame()

submission["MoleculeId"] = MoleculeId

submission['PredictedProbability'] = Probability
submission.to_csv('submission.csv',index=None)
print(submission)