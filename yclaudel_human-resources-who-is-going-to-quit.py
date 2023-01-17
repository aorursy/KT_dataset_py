import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

plt.rcParams['figure.figsize'] = (16, 9)
# load data

df = pd.read_csv('../input/human-resources-data-set/HRDataset_v13.csv')
df.dropna(how='all',inplace=True)

df.head()
# Missing Values

missings = df.isnull().sum()

print(missings[missings>0])
# Missing values

df['DaysLateLast30'] = df['DaysLateLast30'].fillna(-1)

# Conversion to datatime

df['DateofHire'] = pd.to_datetime(df['DateofHire'], dayfirst = False)

df['DOB'] = pd.to_datetime(df['DOB'], dayfirst = False)

df['DateofTermination'] = pd.to_datetime(df['DateofTermination'], dayfirst = False)

df['LastPerformanceReview_Date'] = pd.to_datetime(df['LastPerformanceReview_Date'], dayfirst = False)

 

#df['Last_Name'] = df['Employee_Name'].apply(lambda name: name.split(',')[0].strip())

#df['first_Name'] = df['Employee_Name'].apply(lambda name: name.split(',')[1].strip() if ("," in name) else "")



# Cleaning

df["Sex"] = df["Sex"].str.strip()

# bining PayRate

bins = [10,20,30,40,50,60,70,80,90]

labels = [10,20,30,40,50,60,70,80]

df["PayRateBin10"] = pd.cut(df["PayRate"], bins)
df['LastDayWorked'] = df['DateofTermination']

df['LastDayWorked'].fillna(df['LastPerformanceReview_Date'].max(),inplace=True)

df['NbrDayWorked'] = (df['LastDayWorked']-df['DateofHire'])//np.timedelta64(1,'D')

del df['LastDayWorked']

print(df['NbrDayWorked'].max(),df['NbrDayWorked'].min())
df["PayRateMedianPosition"] = df.groupby(['Position'])["PayRate"].transform('median')

df["PayRateEcart"] = df["PayRate"] - df["PayRateMedianPosition"]
# Columns to keep by nature

keepColId = ['MarriedID', 'FromDiversityJobFairID','Termd']

keepColNum = ['NbrDayWorked','PayRateEcart', 'EngagementSurvey', 'EmpSatisfaction','SpecialProjectsCount']

keepColCat = ['Sex', 'MaritalDesc', 'CitizenDesc','Department', 'PerformanceScore']



# Normalization

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaledData = scaler.fit_transform(df[keepColNum])

scaledData=pd.DataFrame(scaledData,columns=keepColNum)



# OneHotEncoding

data = pd.concat([scaledData,df[keepColId], pd.get_dummies(df[keepColCat]),df['EmploymentStatus']], axis=1)



# Features

X=data.loc[data['EmploymentStatus'].isin(['Active','Voluntarily Terminated'])].copy()

# Target

y=X['Termd']



X.drop(["Termd",'EmploymentStatus'], axis=1, inplace=True)
pd.crosstab(df['Termd'], df['EmploymentStatus'])
pd.crosstab(df['EmpStatusID'], df['EmploymentStatus'])
from sklearn import tree

import graphviz 



clf = tree.DecisionTreeClassifier()

clf.fit(X, y)



dot_data = tree.export_graphviz(clf, out_file=None, 

                     feature_names=X.columns,  

                     filled=True, rounded=True,  

                     special_characters=True)  

graph = graphviz.Source(dot_data)  

graph 
columns = X.columns

importances = list(zip(clf.feature_importances_,columns))

importances.sort(reverse=True)

pd.DataFrame(importances, index=[x for (_,x) in importances]).plot(kind = 'bar');
scatter = plt.scatter(X["NbrDayWorked"], X["SpecialProjectsCount"],c=y)

plt.xlabel("NbrDayWorked")

plt.ylabel("SpecialProjectsCount")

plt.legend(*scatter.legend_elements(),title="Classes")

plt.show()



scatter = plt.scatter(X["PayRateEcart"], X["MaritalDesc_Divorced"],c=y)

plt.xlabel("PayRateEcart")

plt.ylabel("MaritalDesc_Divorced")

plt.legend(*scatter.legend_elements(),title="Classes")

plt.show()
pd.crosstab(X['Sex_F'], y)
pd.crosstab(X['Department_Production       '], y)
scatter = plt.scatter(df["NbrDayWorked"], df["PayRate"],c=df['EmpStatusID'])

plt.xlabel("NbrDayWorked")

plt.ylabel("PayRate")

plt.legend(handles=scatter.legend_elements()[0],labels=['Active','Future Start','Leave of Absence','Terminated for Cause','Voluntarily Terminated'])

plt.show()
from sklearn.linear_model import LogisticRegression    

from sklearn.metrics import f1_score

from matplotlib.colors import ListedColormap, colorConverter, LinearSegmentedColormap



def visualize_coefficients(coefficients, feature_names, n_top_features=25):

    coefficients = coefficients.squeeze()

    if coefficients.ndim > 1:

        # this is not a row or column vector

        raise ValueError("coeffients must be 1d array or column vector, got"

                         " shape {}".format(coefficients.shape))

    coefficients = coefficients.ravel()



    if len(coefficients) != len(feature_names):

        raise ValueError("Number of coefficients {} doesn't match number of"

                         "feature names {}.".format(len(coefficients),

                                                    len(feature_names)))

    # get coefficients with large absolute values

    coef = coefficients.ravel()

    positive_coefficients = np.argsort(coef)[-n_top_features:]

    negative_coefficients = np.argsort(coef)[:n_top_features]

    interesting_coefficients = np.hstack([negative_coefficients,

                                          positive_coefficients])

    # plot them

    plt.figure(figsize=(20, 7))

    cm = ListedColormap(['#0000aa', '#ff2020'])

    colors = [cm(1) if c < 0 else cm(0)

              for c in coef[interesting_coefficients]]

    plt.bar(np.arange(2 * n_top_features), coef[interesting_coefficients],

            color=colors)

    feature_names = np.array(feature_names)

    plt.subplots_adjust(bottom=0.3)

    plt.xticks(np.arange(1, 1 + 2 * n_top_features),

               feature_names[interesting_coefficients], rotation=60,

               ha="right")

    plt.ylabel("Coefficient magnitude")

    plt.xlabel("Features")
logreg = LogisticRegression()

logreg.fit(X, y)

y_predict = logreg.predict(X)

print("\n LogisticRegression f1_score: {:.5f}\n".format(f1_score(y, y_predict, average='weighted')))
visualize_coefficients(logreg.coef_, X.columns,n_top_features=10)