import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib



from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.metrics import confusion_matrix



from scipy.stats import norm

from matplotlib import pyplot as plt

from scipy import stats



%matplotlib inline





plt.style.use('seaborn-whitegrid')

#plt.rcParams['figure.dpi'] = 100
df = pd.read_csv('/kaggle/input/airline-passenger-satisfaction/train.csv')

df.shape
df.head()
df = df.drop('Unnamed: 0', axis=1)

df = df.drop('id', axis=1)
df.isnull().sum()
df['satisfaction'].unique()
sns.countplot(x='satisfaction',data=df, palette="Set1");
sns.catplot("satisfaction", col="Gender", col_wrap=2, data=df, kind="count", height=3.5, aspect=1.0, palette="Set1"); 
sns.catplot("satisfaction", col="Customer Type", col_wrap=2, data=df, kind="count", height=3.5, aspect=1.0, palette="Set1"); 
sns.catplot("satisfaction", col="Class", col_wrap=3, data=df, kind="count", height=3.5, aspect=1.0, palette="Set1"); 
sns.catplot("satisfaction", col="Type of Travel", col_wrap=2, data=df, kind="count", height=3.5, aspect=1.0, palette="Set1"); 
num_features = df.columns.drop(["Gender", "Customer Type", "Class", "Type of Travel", "satisfaction"])

num_features
corr_matrix = df[num_features].corr()

corr_matrix = np.round(corr_matrix, 2)

corr_matrix[np.abs(corr_matrix) < 0.3] = 0

plt.figure(figsize=(15,7))

sns.heatmap(corr_matrix, annot=True, linewidths=.5, cmap='coolwarm');

#sns.pairplot(df['Cleanliness', 'Food and drink', 'Seat comfort' ,'Inflight entertainment'], size = 2.5)

#plt.show();
df = df.drop('Arrival Delay in Minutes', axis=1)
plt.figure(figsize = (15, 7))

sns.distplot(df['Flight Distance'], fit=norm, color='grey');

fig = plt.figure()

res = stats.probplot(df['Flight Distance'], plot=plt)
plt.figure(figsize = (15, 7))

fig = sns.kdeplot(df.loc[df['satisfaction'] == 'neutral or dissatisfied', 'Flight Distance'], label="neutral or dissatisfied", color='red');

fig = sns.kdeplot(df.loc[df['satisfaction'] == 'satisfied', 'Flight Distance'], label="satisfied", color='blue');

fig.figure.suptitle("Satisfaction + Flight Distance", fontsize = 16);

plt.xlabel('Flight Distance', fontsize=14);

plt.ylabel('Distribution', fontsize=14);
plt.figure(figsize = (15, 7))

fig = sns.kdeplot(df.loc[(df['satisfaction'] == 'neutral or dissatisfied') | (df['Class'] == 'Business') , 'Flight Distance'], label="Business - neutral or dissatisfied", color='red', linestyle='--');

fig = sns.kdeplot(df.loc[(df['satisfaction'] == 'satisfied') | (df['Class'] == 'Business'), 'Flight Distance'], label="Business - satisfied", color='blue', linestyle='--');

fig = sns.kdeplot(df.loc[(df['satisfaction'] == 'neutral or dissatisfied') | (df['Class'] != 'Business') , 'Flight Distance'], label="Eco - neutral or dissatisfied", color='red');

fig = sns.kdeplot(df.loc[(df['satisfaction'] == 'satisfied') | (df['Class'] != 'Business'), 'Flight Distance'], label="Eco - satisfied" , color='blue');

fig.figure.suptitle("Satisfaction + Class+ Flight Distance", fontsize = 16);

plt.xlabel('Flight Distance', fontsize=14);

plt.ylabel('Distribution', fontsize=14);
plt.figure(figsize = (15, 7))

fig = sns.kdeplot(df.loc[(df['satisfaction'] == 'neutral or dissatisfied') | (df['Type of Travel'] == 'Personal Travel') , 'Flight Distance'], label="Personal Travel - neutral or dissatisfied", color='red', linestyle='--');

fig = sns.kdeplot(df.loc[(df['satisfaction'] == 'satisfied') | (df['Type of Travel'] == 'Personal Travel'), 'Flight Distance'], label="Personal Travel - satisfied", color='blue', linestyle='--');

fig = sns.kdeplot(df.loc[(df['satisfaction'] == 'neutral or dissatisfied') | (df['Type of Travel'] != 'Personal Travel') , 'Flight Distance'], label="Business Travel - neutral or dissatisfied", color='red');

fig = sns.kdeplot(df.loc[(df['satisfaction'] == 'satisfied') | (df['Type of Travel'] != 'Personal Travel'), 'Flight Distance'], label="Business Travel - satisfied" , color='blue');

fig.figure.suptitle("Satisfaction + Type of travel + Flight Distance", fontsize = 16)

plt.xlabel('Purchase amount', fontsize=14);

plt.ylabel('Distribution', fontsize=14);
plt.figure(figsize = (15, 7))

sns.distplot(df['Age'], fit=norm, color='grey');

fig = plt.figure()

res = stats.probplot(df['Age'], plot=plt)
plt.figure(figsize = (15, 7))

fig = sns.kdeplot(df.loc[df['satisfaction'] == 'neutral or dissatisfied', 'Age'], label="neutral or dissatisfied", color='red');

fig = sns.kdeplot(df.loc[df['satisfaction'] == 'satisfied', 'Age'], label="satisfied", color='blue');

fig.figure.suptitle("Satisfaction + Age", fontsize = 16);

plt.xlabel('Age', fontsize=14);

plt.ylabel('Distribution', fontsize=14);
plt.figure(figsize = (15, 7))

fig = sns.kdeplot(df.loc[(df['satisfaction'] == 'neutral or dissatisfied') | (df['Class'] == 'Business') , 'Age'], label="Business - neutral or dissatisfied", color='red', linestyle='--');

fig = sns.kdeplot(df.loc[(df['satisfaction'] == 'satisfied') | (df['Class'] == 'Business'), 'Age'], label="Business - satisfied", color='blue', linestyle='--');

fig = sns.kdeplot(df.loc[(df['satisfaction'] == 'neutral or dissatisfied') | (df['Class'] != 'Business') , 'Age'], label="Eco - neutral or dissatisfied", color='red');

fig = sns.kdeplot(df.loc[(df['satisfaction'] == 'satisfied') | (df['Class'] != 'Business'), 'Age'], label="Eco - satisfied" , color='blue');

fig.figure.suptitle("Satisfaction + Class+ Age", fontsize = 16);

plt.xlabel('Age', fontsize=14);

plt.ylabel('Distribution', fontsize=14);
plt.figure(figsize = (15, 7))

fig = sns.kdeplot(df.loc[(df['satisfaction'] == 'neutral or dissatisfied') | (df['Type of Travel'] == 'Personal Travel') , 'Age'], label="Personal Travel - neutral or dissatisfied", color='red', linestyle='--');

fig = sns.kdeplot(df.loc[(df['satisfaction'] == 'satisfied') | (df['Type of Travel'] == 'Personal Travel'), 'Age'], label="Personal Travel - satisfied", color='blue', linestyle='--');

fig = sns.kdeplot(df.loc[(df['satisfaction'] == 'neutral or dissatisfied') | (df['Type of Travel'] != 'Personal Travel') , 'Age'], label="Business Travel - neutral or dissatisfied", color='red');

fig = sns.kdeplot(df.loc[(df['satisfaction'] == 'satisfied') | (df['Type of Travel'] != 'Personal Travel'), 'Age'], label="Business Travel - satisfied" , color='blue');

fig.figure.suptitle("Satisfaction + Type of travel + Age", fontsize = 16)

plt.xlabel('Age', fontsize=14);

plt.ylabel('Distribution', fontsize=14);
plt.figure(figsize = (15, 7))

sns.distplot(df['Departure Delay in Minutes'], fit=norm, color='grey');

fig = plt.figure()

res = stats.probplot(df['Departure Delay in Minutes'], plot=plt)
plt.figure(figsize = (15, 7))

fig = sns.kdeplot(df.loc[df['satisfaction'] == 'neutral or dissatisfied', 'Departure Delay in Minutes'], label="neutral or dissatisfied", color='red');

fig = sns.kdeplot(df.loc[df['satisfaction'] == 'satisfied', 'Departure Delay in Minutes'], label="satisfied", color='blue');

fig.figure.suptitle("Satisfaction + Departure Delay in Minutes", fontsize = 16);

plt.xlabel('Departure Delay in Minutes', fontsize=14);

plt.ylabel('Distribution', fontsize=14);
features_0_5 = num_features.drop(["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"])

features_0_5
for feature in features_0_5:

    print(feature, df[feature].unique())
plt.figure(figsize = (20, 10))

for feature in features_0_5:    

    fig = sns.lineplot(data=df[feature].value_counts(sort=False), linewidth=2, label=feature)

fig.figure.suptitle("Count + Feature_0_5", fontsize = 16);

plt.xlabel('Value', fontsize=14);

plt.ylabel('Count', fontsize=14);
plt.figure(figsize = (20, 10))

for feature in features_0_5:    

    fig = sns.lineplot(data=df.loc[df['satisfaction'] == 'neutral or dissatisfied', feature].value_counts(sort=False), linewidth=2, label=feature)

fig.figure.suptitle("Count + Feature_0_5 - neutral or dissatisfied ", fontsize = 16);

plt.xlabel('Value', fontsize=14);

plt.ylabel('Count', fontsize=14);
plt.figure(figsize = (20, 10))

for feature in features_0_5:    

    fig = sns.lineplot(data=df.loc[df['satisfaction'] != 'neutral or dissatisfied', feature].value_counts(sort=False), linewidth=2.5, label=feature)

fig.figure.suptitle("Count + Feature_0_5 - satisfied ", fontsize = 16);

plt.xlabel('Value', fontsize=14);

plt.ylabel('Count', fontsize=14);
df.info()
df['satisfaction'].unique()
df['satisfaction'].replace({'neutral or dissatisfied': 0, 'satisfied': 1},inplace = True)
df['satisfaction'].unique()
df['Gender'].unique()
df['Gender'].replace({'Female': 0, 'Male': 1},inplace = True) # may the feminists forgive me  :)
df['satisfaction'].unique()
df['Customer Type'].unique()
df['Customer Type'].replace({'disloyal Customer': 0, 'Loyal Customer': 1},inplace = True)
df['Customer Type'].unique()
df['Type of Travel'].unique()
df['Type of Travel'].replace({'Personal Travel': 0, 'Business travel': 1},inplace = True)
df['Type of Travel'].unique()
df['Class'].unique()
ClassD = pd.get_dummies(df['Class'])

df = pd.concat([df, ClassD],axis =1)
df = df.drop('Class', axis=1)
df.head()
features_0_5
num_features = df.columns.drop(["Gender", "Customer Type", "Type of Travel", "satisfaction", "Flight Distance", "Departure Delay in Minutes", "Age", "Business", "Eco", "Eco Plus"])

num_features
inflight_features = ['Inflight wifi service', 'Departure/Arrival time convenient', 'Food and drink', 'Seat comfort', 'Inflight entertainment', 'Inflight service', 'Cleanliness']

inflight_features
boardinf_features = num_features.drop(inflight_features)

boardinf_features
def reduce_dims(df, dims=2, method='pca'):

    

    assert method in ['pca', 'tsne'], 'Incorrect method'

    

    if method=='pca':

        dim_reducer = PCA(n_components=dims, random_state=42)

        components = dim_reducer.fit_transform(df)

    elif method == 'tsne':

        dim_reducer = TSNE(n_components=dims, learning_rate=250, random_state=42)

        components = dim_reducer.fit_transform(df)

    else:

        print('Error')

        

    colnames = ['component_' + str(i) for i in range(1, dims+1)]

    return dim_reducer, pd.DataFrame(data = components, columns = colnames) 
dim_reducer2d, inflight_components_2d = reduce_dims(df[inflight_features], dims=2, method='pca')

inflight_components_2d.head(2)
df[inflight_features].shape, inflight_components_2d.shape
variance_before_inflight_features = np.var(df[inflight_features], axis=0, ddof=1).sum()
variance_after_inflight_features = np.var(inflight_components_2d, axis=0, ddof=1).sum()

variance_after_inflight_features
variance_after_inflight_features / variance_before_inflight_features
inflight_components_2d
dim_reducer2d, boardinf_components_2d = reduce_dims(df[boardinf_features], dims=2, method='pca')

boardinf_components_2d.head(2)
df[boardinf_features].shape, boardinf_components_2d.shape
variance_before_boardinf_features = np.var(df[boardinf_features], axis=0, ddof=1).sum()
variance_after_boardinf_features = np.var(boardinf_components_2d, axis=0, ddof=1).sum()

variance_after_boardinf_features
variance_after_boardinf_features / variance_before_boardinf_features
boardinf_components_2d
df_with_pca_components = df
df_with_pca_components['inflight_component_1'] = inflight_components_2d['component_1']
df_with_pca_components['inflight_component_2'] = inflight_components_2d['component_2']
df_with_pca_components['boarding_component_1'] = boardinf_components_2d['component_1']
df_with_pca_components['boarding_component_2'] = boardinf_components_2d['component_2']
df.info()
y = df['satisfaction']

X = df.drop('satisfaction', axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=42)
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)

print(classification_report(y_test, predictions))

confusion_matrix(y_test, predictions)
y = df_with_pca_components['satisfaction']

X = df_with_pca_components.drop('satisfaction', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=42)
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)

print(classification_report(y_test, predictions))

confusion_matrix(y_test, predictions)