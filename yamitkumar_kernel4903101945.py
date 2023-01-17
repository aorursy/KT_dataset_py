import numpy as np

import pandas as pd





import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(color_codes=True)

%matplotlib inline



from sklearn import model_selection

from sklearn import metrics



from sklearn.decomposition import PCA



from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.svm import SVC



from sklearn.model_selection import GridSearchCV

vehicle_df = pd.read_csv("../input/vehicle/vehicle.csv")



print('Dataframe Shape : ', vehicle_df.shape);

print('\n')

vehicle_df.info()
vehicle_df.head(10)
data = vehicle_df.isnull().sum()

df = pd.DataFrame({'columns': vehicle_df.columns, 'missing_count': data.values})

df = df[df['missing_count'] > 0]

print(df.sort_values(['missing_count'], ascending=False))

print()

print('Missing data in ', df['columns'].size, ' columns.')

print('Missing data columns : ', df[df['missing_count'] > 0]['columns'].values)
# d. 5 point summary of numerical attributes

vehicle_df.describe().round(2).T

vehicle_df2 = vehicle_df.copy()

vehicle_df2.fillna(vehicle_df2.mean(), inplace=True)

vehicle_df2.drop("class", axis=1, inplace=True)



fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(25, 25))

for i, column in enumerate(vehicle_df2.columns):

    sns.distplot(vehicle_df2[column],ax=axes[i//3,i%3])
# Class column data distribution

vehicle_df['class'].value_counts()
#plt.subplots(figsize=(100, 100))

#sns.boxplot(data=vehicle_df2, orient="h")



fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(25, 25))

#fivepoint = pd.DataFrame(columns=['Model Name', 'Accuracy', 'Recall', 'Precision'])

for i, column in enumerate(vehicle_df2.columns):

    sns.boxplot(vehicle_df2[column],ax=axes[i//3,i%3], dodge=False, whis=1.5)

    
# Max value based on boxplot to filter outliers of 8 columns where outliers are identified. 

max_df = pd.DataFrame([[255,77,13,288,990,87,19,40]],columns=['radius_ratio', 'pr.axis_aspect_ratio', 'max.length_aspect_ratio', 'scaled_variance', 'scaled_variance.1', 'scaled_radius_of_gyration.1', 'skewness_about', 'skewness_about.1'])



total_outliers = 0

for i, column in enumerate(max_df.columns):

    #print(column, max_df[column][0], vehicle_df[column][vehicle_df[column] > max_df[column][0]].size)

    total_outliers += vehicle_df[column][vehicle_df[column] > max_df[column][0]].size

    

print('Total Outliers ', total_outliers)

print('Total Outliers %', round((total_outliers/len(vehicle_df.index))*100) )

vehicle_df_new = vehicle_df.copy();



# Fill null

vehicle_df_new.fillna(vehicle_df_new.mean(), inplace=True)



# Remove outliers based on max value identified earlier from boxplot

for i, column in enumerate(max_df.columns):

    vehicle_df_new = vehicle_df_new[vehicle_df_new[column] < max_df[column][0]]

    

# Convert class column to categorical 

vehicle_df_new['class'] = pd.Categorical(vehicle_df_new['class']).codes



## rest the index post cleaning the outliers

vehicle_df_new = vehicle_df_new.reset_index(drop=True)



vehicle_df_new.info()

vehicle_df_new.head()
# independant variables

X = vehicle_df_new.drop(['class'], axis=1)

# the dependent variable

y = vehicle_df_new[['class']]



sns.pairplot(X, diag_kind='kde')   # plot density curve instead of histogram on the diag
corr = vehicle_df_new.corr().round(2)

plt.figure(figsize=(20,20))

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,cmap='RdBu')
c = ['class']



# High correlation columns (Group 1)

cols_hc1 = ['compactness','circularity','distance_circularity','radius_ratio','scatter_ratio','elongatedness',

           'pr.axis_rectangularity','max.length_rectangularity','scaled_variance','scaled_variance.1',

           'scaled_radius_of_gyration']

sns.pairplot(vehicle_df_new[[*cols_hc1, *c]], diag_kind='kde', hue='class')
# High correlated columns (Group 2)

cols_hc2 = ['hollows_ratio', 'scaled_radius_of_gyration.1', 'skewness_about.2']



sns.pairplot(vehicle_df_new[[*cols_hc2, *c]], diag_kind='kde', hue='class')
# Low correlation columns

cols_lc = ['pr.axis_aspect_ratio','max.length_aspect_ratio','skewness_about','skewness_about.1']



sns.pairplot(vehicle_df_new[[*cols_lc, *c]], diag_kind='kde', hue='class')
#Scale the values

from scipy.stats import zscore

XScaled=X.apply(zscore)

XScaled.head()
# Apply PCA on Group 1 of high Corelation columns

X1 = XScaled[cols_hc1]

pca1 = PCA(n_components=len(cols_hc1), whiten=False)

pca1.fit(X1)
print('Original number of features:', len(cols_hc1))

#print('Reduced number of features:', pca1.shape[1])

print()

print('Eigen Values', pca1.explained_variance_)

print()

#print('Eigen Vector', pca1.components_)

#print()

#print('Percentage  ', pca1.explained_variance_ratio_)



percent_variance = np.asarray([float(format(num, '.3f')) for num in pca1.explained_variance_ratio_])

percent_variance = np.round(np.asarray(percent_variance) * 100, decimals =2)

print('Percentage  ', percent_variance)
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 5))



ax1.bar(list(range(0,len(cols_hc1))),pca1.explained_variance_ratio_,align='center')

ax1.set(xlabel='Eigen Value', ylabel='Variation explained')



ax2.step(list(range(0,len(cols_hc1))),np.cumsum(pca1.explained_variance_ratio_), where='mid')

#ax2.plot(pca1.explained_variance_)

ax2.set(xlabel='Eigen Value', ylabel='Cumulative of variation explained')

plt.show()
# With 4 variables we can explain over 95% of the variation in the original data of group1 columns. 

# And with 5 variables we can explain more than 98%

pca1_95 = PCA(n_components=0.95, whiten=True)

X_pca1_95 = pca1_95.fit_transform(X1)

print('Original number of features:', len(cols_hc1))

print('Reduced number of features:', X_pca1_95.shape[1])

print(X_pca1_95.shape)

sns.pairplot(pd.DataFrame(X_pca1_95))
# Apply PCA on Group 2 of high Corelation columns

X2 = XScaled[cols_hc2]



pca2 = PCA(n_components=len(cols_hc2), whiten=False)

pca2.fit(X2)

print('Eigen Values', pca2.explained_variance_)

print('Percentage  ', np.round(pca2.explained_variance_ratio_ * 100, decimals =2))





# With 2 variables we can explain over 95% of the variation in the original data of group2 columns

pca2 = PCA(n_components=0.95, whiten=True)

X_pca2 = pca2.fit_transform(X2)



print('Original number of features:', len(cols_hc2))

print('Reduced number of features:', X_pca2.shape[1])

print(X_pca2.shape)

sns.pairplot(pd.DataFrame(X_pca2))
#Reduced group1 of 11 columns to 4 columns with 95% variance

x_pca1_95_df = pd.DataFrame(data = X_pca1_95)



#Reduced group1 of 3 columns to 2 columns 

x_pca2_df = pd.DataFrame(data = X_pca2)



#Combind the 3 data frames a) Group1 PCA columns (95% variance), b) Group2 PCA columns, c) No correlation columns

X_new = pd.merge(x_pca1_95_df,x_pca2_df,right_index=True, left_index=True);

X_new = pd.merge(X_new,XScaled[cols_lc],right_index=True, left_index=True);



print('Final Shape', X_new.shape)

X_new.head(10)
## Split the train and test data into 70:30 ratio

X_train, X_test, y_train, y_test = train_test_split(XScaled, y, test_size = 0.3, random_state = 1)

## build the SVM model on training data

svc_org = SVC()

svc_org.fit(X_train,y_train)

prediction= svc_org.predict(X_test)

print(XScaled.shape)

#print("Class Distribution:\n",y['class'].value_counts())

print("Train Data Score", round(svc_org.score(X_train, y_train), 3))

print("Test Data Score ", round(svc_org.score(X_test,y_test), 3))

print("Confusion Matrix:\n   bus car van\n",metrics.confusion_matrix(prediction,y_test))

target_names = ['bus', 'car', 'van']

print(metrics.classification_report(y_test, prediction, target_names=target_names))
## Split the train and test data into 70:30 ratio

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.3, random_state = 1)

## build the SVM model on training data

svc_pca = SVC()

svc_pca.fit(X_train,y_train)

prediction= svc_pca.predict(X_test)

print(X_new.shape)

print(svc_pca)

print("Train Data Score", round(svc_pca.score(X_train, y_train), 3))

print("Test Data Score", round(svc_pca.score(X_test,y_test),3))

print("Confusion Matrix:\n   bus car van\n",metrics.confusion_matrix(prediction,y_test))

target_names = ['bus', 'car', 'van']

print(metrics.classification_report(y_test, prediction, target_names=target_names))

metrics.classification_report(y_test, prediction, target_names=target_names)
import warnings

warnings.filterwarnings("ignore")



# Parameter Grid

param_grid = [{'kernel': ['linear'], 'C': [0.01, 0.05, 0.5, 1.0, 10, 25, 50]},

              {'kernel': ['rbf'], 'C': [0.01, 0.05, 0.5, 1.0, 10, 25, 50]}

             ] 

# Make grid search classifier

clf_grid = GridSearchCV(SVC(), param_grid, verbose=1)

 

# Train the classifier

clf_grid.fit(X_train, y_train)

 

# clf = grid.best_estimator_()

print("Best Parameters:\n", clf_grid.best_params_)

print("Best Estimators:\n", clf_grid.best_estimator_)
## Split the train and test data into 70:30 ratio

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.3, random_state = 1)

## build the SVM model on training data

svc_pca_tun = SVC(C=10, kernel = "rbf")

svc_pca_tun.fit(X_train,y_train)

prediction= svc_pca_tun.predict(X_test)

print(X_new.shape)

print("Train Data Score", round(svc_pca_tun.score(X_train, y_train), 3))

print("Test Data Score", round(svc_pca_tun.score(X_test,y_test),3))

print("Confusion Matrix:\n   bus car van\n",metrics.confusion_matrix(prediction,y_test))

target_names = ['bus', 'car', 'van']

print(metrics.classification_report(y_test, prediction, target_names=target_names))

pred_kfold = cross_val_score(svc_org, XScaled, y, cv=10) 

print("Accuracy with SVM on original data: %0.2f (+/- %0.2f)" % (pred_kfold.mean(), pred_kfold.std() * 2))



pred_kfold = cross_val_score(svc_pca, X_new, y, cv=10) 

print("Accuracy with SVM on PCA data: %0.2f (+/- %0.2f)" % (pred_kfold.mean(), pred_kfold.std() * 2))



pred_kfold = cross_val_score(svc_pca_tun, X_new, y, cv=10) 

print("Accuracy with SVM with tuned params on PCA data: %0.2f (+/- %0.2f)" % (pred_kfold.mean(), pred_kfold.std() * 2))
