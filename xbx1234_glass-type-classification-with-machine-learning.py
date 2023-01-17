import numpy as np  # linear algebra

import pandas as pd  # read dataframes

import matplotlib.pyplot as plt # visualization

import seaborn as sns # statistical visualizations and aesthetics

from sklearn.preprocessing import StandardScaler # preprocessing 

from sklearn.decomposition import PCA # dimensionality reduction

from scipy.stats import boxcox # data transform

from sklearn.model_selection import (train_test_split, KFold , cross_val_score, GridSearchCV ) # model selection modules

from sklearn.pipeline import Pipeline # streaming pipelines

# load models

from sklearn.tree import DecisionTreeClassifier

from xgboost import (XGBClassifier, plot_importance)

from sklearn.svm import SVC

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

%matplotlib inline 
df = pd.read_csv('../input/glass.csv')



print(df.shape)
df.head(15)
df.dtypes
df.describe()
df['Type'].value_counts()
features = df.columns[:-1].tolist()

for feat in features:

    skew = df[feat].skew()

    sns.distplot(df[feat], label='Skew = %.3f' %(skew))

    plt.legend(loc='best')

    plt.show()
sns.boxplot(df[features])

plt.show()
plt.figure(figsize=(8,8))

sns.pairplot(df[features],palette='coolwarm')

plt.show()
corr = df[features].corr()

plt.figure(figsize=(8,8))

sns.heatmap(corr, cbar = True,  square = True, annot=True,

           xticklabels= df.columns.tolist(), yticklabels= df.columns.tolist(),

           cmap= 'coolwarm')

plt.show()

print(corr)
df.info()
# Define X as features and y as lablels

X = df[features]

y = df['Type']

# set a seed and a test size for splitting the dataset 

seed = 7

test_size = 0.2



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size , random_state = seed)
features_boxcox = []



for feature in features:

    bc_transformed, _ = boxcox(X_train[feature]+1)  # shift by 1 to avoid computing log of negative values

    features_boxcox.append(bc_transformed)



features_boxcox = np.column_stack(features_boxcox)

df_bc = pd.DataFrame(data=features_boxcox, columns=features)

df_bc['Type'] = df['Type']
df_bc.head()
for feature in features:

    fig, ax = plt.subplots(1,2,figsize=(7,3.5))    

    ax[0].hist(df[feature], color='blue', bins=30, alpha=0.3, label='Skew = %s' %(str(round(X_train[feature].skew(),3))) )

    ax[0].set_title(str(feature))   

    ax[0].legend(loc=0)

    ax[1].hist(df_bc[feature], color='red', bins=30, alpha=0.3, label='Skew = %s' %(str(round(df_bc[feature].skew(),3))) )

    ax[1].set_title(str(feature)+' after a Box-Cox transformation')

    ax[1].legend(loc=0)

    plt.show()
# check if skew is closer to zero after a box-cox transform

for feature in features:

    delta = np.abs( df_bc[feature].skew() / df[feature].skew() )

    if delta < 1.0 :

        print('Feature %s is less skewed after a Box-Cox transform' %(feature))

    else:

        print('Feature %s is more skewed after a Box-Cox transform'  %(feature))
df_bc["Si"] = df["Si"]



for feature in features:

    if feature not in ["Si"]:

        X_train[feature], lmbda = boxcox(X_train[feature]+1)  # shift by 1 to avoid computing log of negative values

        X_test[feature] = X_test[feature].apply(lambda x: ((x+1.0)**lmbda - 1.0)/lmbda if lmbda !=0 else np.log(x+1) )





X_train, X_test = X_train.values, X_test.values

y_train, y_test = y_train.values, y_test.values
# Standarize the dataset 

for i in range(X.shape[1]):

    sc = StandardScaler()

    X_train[:,i] = sc.fit_transform(X_train[:,i].reshape(-1,1)).reshape(1,-1)

    X_test[:,i] = sc.transform(X_test[:,i].reshape(-1,1)).reshape(1,-1)
model_importances = XGBClassifier(n_estimators=200)

model_importances.fit(X_train, y_train)

plot_importance(model_importances)

plt.show()
pca = PCA(random_state = seed)

pca.fit(X_train)

var_exp = pca.explained_variance_ratio_

cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1,len(cum_var_exp)+1), var_exp, align= 'center', label= 'individual variance explained', \

       alpha = 0.7)

plt.step(range(1,len(cum_var_exp)+1), cum_var_exp, where = 'mid' , label= 'cumulative variance explained', \

        color= 'red')

plt.ylabel('Explained variance ratio')

plt.xlabel('Principal components')

plt.xticks(np.arange(1,len(var_exp)+1,1))

plt.legend(loc='best')

plt.show()



# Cumulative variance explained

for i, sum in enumerate(cum_var_exp):

    print("PC" + str(i+1), "Cumulative variance: %.3f% %" %(cum_var_exp[i]*100))
pca = PCA(n_components = 6, random_state= seed)

X_train_pca = pca.fit_transform(X_train)

X_test_pca = pca.transform(X_test)



models = []

models.append(('XGBoost', XGBClassifier(seed = seed) ))

models.append(('SVC', SVC(random_state=seed)))

models.append(('RF', RandomForestClassifier(random_state=seed, n_jobs=-1 )))

tree = DecisionTreeClassifier(max_depth=4, random_state=seed)

models.append(('KNN', KNeighborsClassifier(n_jobs=-1)))



results, names  = [], []

num_folds = 10

scoring = 'accuracy'



for name, model in models:

    kfold = KFold(n_splits=num_folds, random_state=seed)

    cv_results = cross_val_score(model, X_train_pca, y_train, cv=kfold, scoring = scoring, n_jobs= -1) 

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)

    

fig = plt.figure(figsize=(8,6))    

fig.suptitle("Algorithms comparison")

ax = fig.add_subplot(1,1,1)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()