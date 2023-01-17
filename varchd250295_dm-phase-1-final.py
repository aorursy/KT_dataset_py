# import packages

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt



from pandas import read_csv

from pandas import concat

from pandas import DataFrame



from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.tsa.seasonal import seasonal_decompose



import scipy.stats
KaggleInput = "/kaggle/input/"

CGMTimeCSV = KaggleInput + "continuous-blood-glucose-monitor-data/CGMDatenumLunchPat1.csv"

CGMValueCSV = KaggleInput + "continuous-blood-glucose-monitor-data/CGMSeriesLunchPat1.csv"
# display data sample

CGMDatenum = pd.read_csv(CGMTimeCSV)

CGMSeries = pd.read_csv(CGMValueCSV)



CGMDatenum.head()
CGMSeries.head()
# Convert TS datatype and reverse indexing for chronological order

CGMDatenum = CGMDatenum.applymap(lambda i : pd.to_datetime(i - 719529, unit='D'))

CGMDatenum = CGMDatenum.iloc[::-1]

CGMDatenum = CGMDatenum.iloc[:, ::-1]



CGMDatenum_updated = CGMDatenum.copy()

CGMDatenum_updated.head()
# Reverse indexing for chronological order

# Missing values- Linear interpolation

CGMSeries = CGMSeries.iloc[::-1]

CGMSeries = CGMSeries.iloc[:, ::-1]



CGMSeries_updated = CGMSeries.copy()

CGMSeries_updated.interpolate(method='linear', inplace=True)



row, col = CGMSeries_updated.shape



CGMSeries_updated.head()
# Feature Matrix

NewFeatureMatrix = pd.DataFrame()
# Windowed velocity(non-overlapping)- 30 mins intervals

velocityDF = pd.DataFrame()

for i in range(0,26):

     velocityDF['Vel_'+str(i)] = (CGMSeries_updated.iloc[:,i+5]-CGMSeries_updated.iloc[:,i])

NewFeatureMatrix['Window_Velocity_Max']=velocityDF.max(axis = 1, skipna=True)

NewFeatureMatrix.head()
#Plotting

plt.plot(NewFeatureMatrix['Window_Velocity_Max'],'r-')

plt.ylabel('Window_Velocity_Max')

plt.xlabel('Days')
#Plotting

fig = plt.figure(figsize = (12,8))

ax = fig.add_subplot(1,1,1) 

ax.set_ylabel('Mean')

ax.set_xlabel('Days')

ax.set_title('Windowed Means')

ax.plot(NewFeatureMatrix.iloc[:,1:7],'-')

ax.legend(('Mean_0', 'Mean_6', 'Mean_12','Mean_18','Mean_24','Mean_30'),loc='upper right')
# Windowed mean interval - 30 mins(non-overlapping)

for i in range(0,31,6):

    NewFeatureMatrix['Mean_'+str(i)] = CGMSeries_updated.iloc[:,i:i+6].mean(axis = 1)

    

NewFeatureMatrix.head()
# FFT- Finding top 8 values for each row

def get_fft(row):

    cgmFFTValues = abs(scipy.fftpack.fft(row))

    cgmFFTValues.sort()

    return np.flip(cgmFFTValues)[0:8]



FFT = pd.DataFrame()

FFT['FFT_Top2'] = CGMSeries_updated.apply(lambda row: get_fft(row), axis=1)

FFT_updated = pd.DataFrame(FFT.FFT_Top2.tolist(), columns=['FFT_1', 'FFT_2', 'FFT_3', 'FFT_4', 'FFT_5', 'FFT_6', 'FFT_7', 'FFT_8'])



#FFT_updated.head()



NewFeatureMatrix = NewFeatureMatrix.join(FFT_updated)



NewFeatureMatrix.head()
# Calculates entropy(from occurences of each value) of given series

def get_entropy(series):

    series_counts = series.value_counts()

    entropy = scipy.stats.entropy(series_counts)  

    return entropy



NewFeatureMatrix['Entropy'] = CGMSeries_updated.apply(lambda row: get_entropy(row), axis=1) 

NewFeatureMatrix.head()
# Final feature matrix

NewFeatureMatrix.head()
# PCA

rows,cols = NewFeatureMatrix.shape



# Standardizes feature matrix

NewFeatureMatrix = StandardScaler().fit_transform(NewFeatureMatrix)



pca = PCA(n_components=5)

principalComponents = pca.fit(NewFeatureMatrix)

print(principalComponents.components_) # Principal Components vs Original Features
print(principalComponents.explained_variance_ratio_.cumsum())
principalComponentsTrans = pca.fit_transform(NewFeatureMatrix)

PC_TimeSeries=pd.DataFrame(data=principalComponentsTrans,columns = ['principal component 1', 'principal component 2','principal component 3', 'principal component 4','principal component 5'])

PC_TimeSeries.head()
#plotting explained variance versus principle componenets

pcs = ['PC1','PC2','PC3','PC4','PC5']

plt.bar(pcs,principalComponents.explained_variance_ratio_*100)

plt.savefig('')
 #plotting top 5 principle components against each time series

ax = PC_TimeSeries.plot.bar(y='principal component 1', rot=0)

ax = PC_TimeSeries.plot.bar(y='principal component 2', rot=0)

ax = PC_TimeSeries.plot.bar(y='principal component 3', rot=0)

ax = PC_TimeSeries.plot.bar(y='principal component 4', rot=0)

ax = PC_TimeSeries.plot.bar(y='principal component 5', rot=0)
#plotting top 5 principle components against each time series

# plots & prove assumptions
# patient-by-patient analysis
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# Turn the values into an array for feeding the classification algorithms.

X_train = X_train.values

X_test = X_test.values

y_train = y_train.values

y_test = y_test.values
# classifiers

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



classifiers = {

    "LogisiticRegression": LogisticRegression(),

    "KNearest": KNeighborsClassifier(),

    "Support Vector Classifier": SVC(),

    "DecisionTreeClassifier": DecisionTreeClassifier()

}
from sklearn.model_selection import cross_val_score





for key, classifier in classifiers.items():

    classifier.fit(X_train, y_train)

    training_score = cross_val_score(classifier, X_train, y_train, cv=5)

    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

y_pred = log_reg.predict(X_train)



# Overfitting Case

print('---' * 45)

print('Recall Score: {:.2f}'.format(recall_score(y_train, y_pred)))

print('Precision Score: {:.2f}'.format(precision_score(y_train, y_pred)))

print('F1 Score: {:.2f}'.format(f1_score(y_train, y_pred)))

print('Accuracy Score: {:.2f}'.format(accuracy_score(y_train, y_pred)))

print('---' * 45)