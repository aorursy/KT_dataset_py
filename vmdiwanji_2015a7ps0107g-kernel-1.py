import numpy as np

import pandas as pd

%matplotlib inline

data = pd.read_csv('../input/firstoffour/train.csv')
data.head()
import xgboost as xgb
data.describe()
data.columns
import matplotlib.pyplot as plt

import seaborn as sns

#TODO

sns.scatterplot(x = "year",y = "AveragePrice",data = data)

plt.show()
x=data

# print(x.reindex())

col1=np.divide(x['Total Bags'].tolist(),x['Total Volume'].tolist())

col1=np.multiply(x['Small Bags'].tolist(),x['4225'].tolist())

print(col1)

cols = pd.DataFrame(col1, columns=["New Volume"])

print(cols.head())

x=pd.concat([x,cols],axis=1)

x=x.loc[data['type'] == 0]

print(x.head())

g = sns.FacetGrid(data=x,  col="type")

g = g.map(plt.scatter, "Large Bags", "AveragePrice", edgecolor="w")

# sns.scatterplot(x = "4225",y = "AveragePrice",data = x)

# sns.scatterplot(x = "4046",y = "AveragePrice",data = x)

# sns.scatterplot(x = "Total Volume",y = "AveragePrice",data = x)

plt.show()
# Compute the correlation matrix

corr = data.corr(method="kendall")



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0.5,

            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)



plt.show()
#TODO

from sklearn.preprocessing import MinMaxScaler

m = MinMaxScaler()

X = data

X=data.loc[data['type'] == 1]

# X = x

y = X['AveragePrice']

X = X.drop(['AveragePrice'],axis=1)

# X=m.fit_transform(X)

# X.head()
#TODO

from sklearn.model_selection import train_test_split

tr_X, te_x, tr_Y, te_Y = train_test_split(X, y, test_size=0.25, random_state=42)
#TODO



from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import explained_variance_score

from sklearn.metrics import mean_squared_error



def performance_metrics(y_true,y_pred):

    rmse = mean_absolute_error(y_true,y_pred)

    r2 = r2_score(y_true,y_pred)

    explained_var_score = explained_variance_score(y_true,y_pred)

    mse=mean_squared_error(y_true,y_pred)

    

    return rmse,r2,explained_var_score,mse



# from sklearn.ensemble import GradientBoostingRegressor

# from sklearn.ensemble import RandomForestRegressor

# from sklearn.model_selection import GridSearchCV

# from sklearn.model_selection import GridSearchCV

# from sklearn.metrics import make_scorer





# #TODO

# clf = RandomForestRegressor()        #Initialize the classifier object



# parameters = {'n_estimators':[10,50,100],'min_samples_split':[2,3,4,5]}    #Dictionary of parameters



# scorer = make_scorer(mean_squared_error)         #Initialize the scorer using make_scorer



# grid_obj = GridSearchCV(clf,parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



# grid_fit = grid_obj.fit(tr_X,tr_Y)        #Fit the gridsearch object with X_train,y_train



# best_clf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object

# print(best_clf)

# unoptimized_predictions = (clf.fit(tr_X, tr_Y)).predict(te_x)      #Using the unoptimized classifiers, generate predictions

# optimized_predictions = best_clf.predict(te_x)        #Same, but use the best estimator



# acc_unop =mean_squared_error(te_Y, unoptimized_predictions)       #Calculate accuracy for unoptimized model

# acc_op = mean_squared_error(te_Y, optimized_predictions)       #Calculate accuracy for optimized model



# print("Accuracy score on unoptimized model:{}".format(acc_unop))

# print("Accuracy score on optimized model:{}".format(acc_op))
# from sklearn.svm import SVR

# svr_rbf = SVR(kernel='rbf', C=100000, gamma=0.8)

# y_rbf = svr_rbf.fit(tr_X,tr_Y).predict(te_x)

# acc_unop =mean_squared_error(te_Y, y_rbf) 

# print(acc_unop)
# print(clf)

print(X.shape)

print()

testdata = pd.read_csv('../input/firstoffour/test.csv')

testdata.head()

tests = testdata

tests=testdata.loc[testdata['type'] == 1]

tests1=testdata.loc[testdata['type'] == 0]

print(tests1.shape)

# tests= testdata.drop(['year'],axis=1)
est = [4000, 5000]

learningrate = [0.05,0.1]

maxdepth = [5,6]

boo = ['gbtree']

for es in est:

    for lr in learningrate:

        for md in maxdepth:

            for b in boo:

                print(es,lr,md,b)

                model = xgb.XGBRegressor(n_estimators=es,learning_rate=lr,max_depth=md,booster=b)

                unoptimized_predictions = (model.fit(tr_X, tr_Y)).predict(te_x)

                acc_unop =mean_squared_error(te_Y, unoptimized_predictions)

                print(es,lr,md,b,acc_unop)

    
model = xgb.XGBRegressor(n_estimators=2000,learning_rate=0.1,max_depth=7)

unoptimized_predictions = (model.fit(X, y)).predict(te_x)

acc_unop =mean_squared_error(te_Y, unoptimized_predictions)

print(acc_unop)
print(tests.shape)

# print(tests.columns)

# tests=m.transform(tests)

testing_predictions = model.predict(tests)

# acc_op = mean_squared_error(te_Y, testing_predictions)

print(testing_predictions[0:5])
from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

m1 = MinMaxScaler()

# pca=PCA(n_components=11)

# X = data

X1=data.loc[data['type'] == 0]

# X = x

Y1 = X1['AveragePrice'].tolist()

# X1 = X1.drop(['Total Volume','Total Bags','year','4770','XLarge Bags','AveragePrice','type'],axis=1)

X1 = X1.drop(['AveragePrice'],axis=1)

# pca.fit(X1)

# X1=pca.fit_transform(X1)

# X1=m1.fit_transform(X1)

#TODO

# colz=np.divide(testdata['Total Bags'].tolist(),testdata['Total Volume'].tolist())

# print(colz)

# cols = pd.DataFrame(colz, columns=["Density"])

# tes=pd.concat([testdata,cols],axis=1)

tests1=testdata.loc[testdata['type'] == 0]

col2=tests1['id'].tolist()

# tests1=pca.fit_transform(tests1)

# tests1 = tests1.drop(['Total Volume','Total Bags','year','4770','XLarge Bags','type'],axis=1)

# tests1=m1.transform(tests1)

print(X1.shape)

print(len(Y1))

from sklearn.model_selection import train_test_split

tr_X1, te_x1, tr_Y1, te_Y1 = train_test_split(X1, Y1, test_size=0.25, random_state=42)

print(te_x1.shape)

print(len(te_Y1))

gam=[2000]

for g in gam:

    model1 = xgb.XGBRegressor(n_estimators=g,learning_rate=0.1,max_depth=7,min_child_weight=1)

    unoptimized_predictions1 = (model1.fit(X1, Y1)).predict(te_x1)

    acc_unop1 =mean_squared_error(te_Y1, unoptimized_predictions1)

    print(acc_unop1)

testing_predictions1 = model1.predict(tests1)



col2=np.array(col2).astype('int32')

print(col2)

ans1=np.column_stack((col2,testing_predictions1))

print(ans1.shape)
ans1[0:5]
sampledata = pd.read_csv('../input/firstoffour/sample_sub.csv')
sampledata.head()
# col1=testdata['id'].tolist()

col1=testdata.loc[testdata['type'] == 1]['id'].tolist()

col1=np.array(col1).astype('int32')

print(len(col1),len(testing_predictions))

print(col1)

ans=np.column_stack((col1,testing_predictions))

print(ans.shape)

# print(ans1.shape)
final_ans=np.concatenate((ans,ans1),axis=0)

print(final_ans.shape)
df = pd.DataFrame(final_ans, columns=["id","AveragePrice"])

a=df['id'].astype('int32')

b=df['AveragePrice']

df=pd.concat([a,b],axis=1)

df.to_csv('list.csv', index=False)
listdata = pd.read_csv('list.csv')

listdata.head()

# listdata.shape