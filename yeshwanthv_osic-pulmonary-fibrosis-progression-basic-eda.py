import os
from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

!pip install dicom
import dicom
import os
import numpy
from matplotlib import pyplot, cm

#plotly
!pip install chart_studio
import plotly.express as px
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

import seaborn as sns
sns.set(style="whitegrid")


#pydicom
import pydicom

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')


# Settings for pretty nice plots
plt.style.use('fivethirtyeight')
plt.show()
PathDicom = "../input/osic-pulmonary-fibrosis-progression/train/ID00422637202311677017371/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))
!pip install natsort
import natsort
# print(natsort.natsorted(lstFilesDCM,reverse=False))
lstFilesDCM = natsort.natsorted(lstFilesDCM,reverse=False)
import pydicom as dicom
import PIL # optional
import pandas as pd
import matplotlib.pyplot as plt

# specify your image path
#image_path = 'xray.dcm'
PathDicom = "../input/osic-pulmonary-fibrosis-progression/train/"
list_patients = [x[0] for x in os.walk(PathDicom)]

for patient in list_patients:
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(patient):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName,filename))

    lstFilesDCM = natsort.natsorted(lstFilesDCM,reverse=False)

    for i in range(len(lstFilesDCM)):
        ds = dicom.dcmread(lstFilesDCM[i])
        print(ds)
        plt.imshow(ds.pixel_array)
        plt.show()
        break
    break

# List files available
list(os.listdir("../input/osic-pulmonary-fibrosis-progression"))
# Defining data path
IMAGE_PATH = "../input/osic-pulmonary-fibrosis-progressiont/"

train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')


#Training data
print('Training data shape: ', train_df.shape)
train_df.head(5)
train_df["typical_FVC"] = (train_df["FVC"]*100)/train_df["Percent"]
train_df.head(5)
plt.figure()
plt.plot(train_df["Weeks"], train_df["FVC"], "o")
plt.show()
plt.figure()
plt.plot(train_df["Weeks"], train_df["Percent"], "o")
plt.show()
from sklearn import linear_model
import statsmodels.api as sm
i = 0
for pid,tdf in train_df.groupby("Patient"):
    if i % 10 == 0:
        sns.lmplot(x='Weeks',y='Percent',data=tdf,fit_reg=True)
        regr = linear_model.LinearRegression()
        X = tdf.Weeks.values.reshape(-1,1)
        y = tdf.Percent.values.reshape(-1,1)
        regr.fit(X, y)
        print(regr.coef_[0], regr.intercept_)
    i += 1
print(i)
i = 0
for pid,tdf in train_df.groupby("Patient"):
    if i % 10 == 0:
        sns.lmplot(x='Weeks',y='Percent',data=tdf,fit_reg=True)
        X = tdf.Weeks.values.reshape(-1,1)
        X = sm.add_constant(X)
        y = tdf.Percent.values.reshape(-1,1)
        model = sm.OLS(y,X)
        results = model.fit()
        print(results.params)
        print(results.bse)
    i += 1
print(i)
i = 0
df = {}
df["Patient"] = []
df["slope"] = []
df["bse"] = []
for pid,tdf in train_df.groupby("Patient"):
    X = tdf.Weeks.values.reshape(-1,1)
    X = sm.add_constant(X)
    y = tdf.Percent.values.reshape(-1,1)
    model = sm.OLS(y,X)
    results = model.fit()
    df["Patient"].append(pid)
    df["slope"].append(results.params[1])
    df["bse"].append(results.bse[1])
    i += 1
print(i)
df = pd.DataFrame(df)
df
df.describe()
typ_fvc_df = train_df.groupby(['Age', 'Sex', 'SmokingStatus']).mean()['typical_FVC'].to_frame().reset_index()
typ_fvc_df
typ_fvc_df.groupby(['Sex', 'SmokingStatus']).mean()['typical_FVC'].to_frame()
conditions = [
    (typ_fvc_df['Age'] <= 50),
    (typ_fvc_df['Age'] > 50) & (typ_fvc_df['Age'] <= 60),
    (typ_fvc_df['Age'] > 60) & (typ_fvc_df['Age'] <= 70),
    (typ_fvc_df['Age'] > 70) & (typ_fvc_df['Age'] <= 80)]
choices = [0,1,2,3]
typ_fvc_df['age_group'] = np.select(conditions, choices, default=4)
typ_fvc_df
typ_fvc_df.groupby(['Sex', 'SmokingStatus', 'age_group']).mean()['typical_FVC'].to_frame()
train_df.groupby(['SmokingStatus']).count()['Sex'].to_frame()
# Null values and Data types
print('Train Set !!')
print(train_df.info())
print('-------------')
print('Test Set !!')
print(test_df.info())
# Total number of Patient in the dataset(train+test)
print("Total Patient in Train set: ",train_df['Patient'].count())
print("Total Patient in Test set: ",test_df['Patient'].count())
print(f"The total patient ids are {train_df['Patient'].count()}, from those the unique ids are {train_df['Patient'].value_counts().shape[0]} ")
columns = train_df.keys()
columns = list(columns)
print(columns)
train_df['SmokingStatus'].value_counts()
train_df['SmokingStatus'].value_counts(normalize=True).iplot(kind='bar',
                                                      yTitle='Percentage', 
                                                      linecolor='black', 
                                                      opacity=0.7,
                                                      color='red',
                                                      theme='pearl',
                                                      bargap=0.8,
                                                      gridcolor='white',
                                                     
                                                      title='Distribution of the SmokingStatus column in the training set')
train_df['Weeks'].value_counts()
train_df['Weeks'].value_counts().sort_values().iplot(kind='barh',
                                                      xTitle='Counts(Weeks)', 
                                                      linecolor='black', 
                                                      opacity=0.7,
                                                      color='#FB8072',
                                                      theme='pearl',
                                                      bargap=0.2,
                                                      gridcolor='white',
                                                      title='Distribution of the Weeks in the training set')
z=train_df.groupby(['SmokingStatus','Weeks'])['FVC'].count().to_frame().reset_index()
z.style.background_gradient(cmap='Reds') 
train_df['FVC'].value_counts()
train_df['FVC'].value_counts().iplot(kind='barh',
                                      xTitle='Lung Capacity(ml)', 
                                      linecolor='black', 
                                      opacity=0.7,
                                      color='#FB8072',
                                      #|theme='pearl',
                                      bargap=0.5,
                                      gridcolor='white',
                                      title='Distribution of the FVC in the training set')
train_df['Percent'].value_counts()
train_df['Percent'].iplot(kind='hist',bins=30,color='blue',xTitle='Percent distribution',yTitle='Count')
train_df['Age'].iplot(kind='hist',bins=30,color='red',xTitle='Age distribution',yTitle='Count')
train_df['SmokingStatus'].value_counts()
sns.kdeplot(train_df.loc[train_df['SmokingStatus'] == 'Ex-smoker', 'Age'], label = 'Ex-smoker',shade=True)

sns.kdeplot(train_df.loc[train_df['SmokingStatus'] == 'Never smoked', 'Age'], label = 'Never smoked',shade=True)

sns.kdeplot(train_df.loc[train_df['SmokingStatus'] == 'Currently smokes', 'Age'], label = 'Currently smokes',shade=True)

# Labeling of plot
plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
sns.kdeplot(train_df.loc[train_df['Sex'] == 'Male', 'Age'], label = 'Male',shade=True)

sns.kdeplot(train_df.loc[train_df['Sex'] == 'Female', 'Age'], label = 'Female',shade=True)

plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
train_df['Sex'].value_counts()
train_df['Sex'].value_counts().iplot(kind='bar',
                                          yTitle='Percentage', 
                                          linecolor='black', 
                                          opacity=0.7,
                                          color='blue',
                                          theme='pearl',
                                          bargap=0.8,
                                          gridcolor='white',

                                          title='Distribution of the Sex column in the training set')
plt.figure(figsize=(16, 6))
a = sns.countplot(data=train_df, x='SmokingStatus', hue='Sex')

for p in a.patches:
    a.annotate(format(p.get_height(), ','), 
           (p.get_x() + p.get_width() / 2., 
            p.get_height()), ha = 'center', va = 'center', 
           xytext = (0, 4), textcoords = 'offset points')

plt.title('Gender split by SmokingStatus', fontsize=16)
sns.despine(left=True, bottom=True);
from IPython.display import Image
Image(filename='../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/1.dcm')

