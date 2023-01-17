import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotnine
from plotnine import *
%matplotlib inline

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
admission_df = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
admission_df.head()
admission_df.count()
admission_df = admission_df.drop('Serial No.', axis=1)
admission_df = admission_df.rename(columns={'GRE Score': 'gre', 'TOEFL Score': 'toefl', 
                                            'University Rating': 'university rating', 
                                           'SOP': 'sop', 'LOR ': 'lor', 'CGPA': 'cgpa', 
                                           'Research': 'research', 'Chance of Admit ': 'admit'})
admission_df['admit'].hist(bins=10, density=True)
admission_df.head()
fig = plt.figure(figsize=(4,4))
axes = sns.PairGrid(admission_df, hue='admit', diag_sharey=False)
axes.map_diag(sns.kdeplot)
axes.map_upper(plt.scatter)
axes.map_lower(plt.scatter)
facet_df = admission_df[['toefl', 'gre', 'cgpa', 'admit']]
fig = plt.figure(figsize=(12, 6))
fig.suptitle('Histogram of TOEFL, GRE and CGPA', y=1.05, fontsize=24)

axes0 = plt.subplot(1, 3, 1)
axes0 = facet_df['toefl'].hist(bins=10, density=True)
axes0.set_xlabel('TOEFL')
axes0.set_ylabel('Frequency')

axes1 = plt.subplot(1, 3, 2)
axes1 = facet_df['gre'].hist(bins=10, density=True)
axes1.set_xlabel('GRE')

axes2 = plt.subplot(1, 3, 3)
axes2 = facet_df['cgpa'].hist(bins=10, density=True)
axes2.set_xlabel('CGPA')
plt.tight_layout()
facet_df.describe()
toefl_gre = facet_df.copy()
toefl_gre['toefl'] = pd.cut(toefl_gre['toefl'], bins=(91,107,121), labels=['Low', 'High'])
toefl_gre['gre'] = pd.cut(toefl_gre['gre'], bins=(289, 317, 341), labels=['Low', 'High'])
toefl_gre.head()
fig = (
ggplot(aes(x = "cgpa", y = "admit", color='admit'),data = toefl_gre) \
+ geom_point(alpha=0.8) \
+ plotnine.facets.facet_grid(facets = ['gre','toefl']) \
+ labs(title = 'Facetplot of TOEFL and GRE',
         x = 'CGPA',
         y = 'Admit') 
)
fig
toefl_cgpa = facet_df.copy()
toefl_cgpa['toefl'] = pd.cut(toefl_cgpa['toefl'], bins=(91,107,121), labels=['Low', 'High'])
toefl_cgpa['cgpa'] = pd.cut(toefl_cgpa['cgpa'], bins=(6.7, 8.57, 10), labels=['Low', 'High'])
toefl_cgpa.head()
fig = (
ggplot(aes(x = "gre", y = "admit", color='admit'),data = toefl_cgpa) \
+ geom_point(alpha=0.8) \
+ plotnine.facets.facet_grid(facets = ['toefl','cgpa']) \
+ labs(title = 'Facetplot of TOEFL and CGPA',
         x = 'GRE',
         y = 'Admit') 
)
fig
gre_cgpa = facet_df.copy()
gre_cgpa['gre'] = pd.cut(gre_cgpa['gre'], bins=(289, 317, 341), labels=['Low', 'High'])
gre_cgpa['cgpa'] = pd.cut(gre_cgpa['cgpa'], bins=(6.7, 8.57, 10), labels=['Low', 'High'])
gre_cgpa.head()
fig = (
ggplot(aes(x = "toefl", y = "admit", color='admit'),data = gre_cgpa) \
+ geom_point(alpha=0.8) \
+ plotnine.facets.facet_grid(facets=['gre','cgpa']) \
+ labs(title = 'Facetplot of GRE and CGPA',
         x = 'TOEFL',
         y = 'Admit') 
)
fig
axes = sns.FacetGrid(toefl_gre, col='toefl', row='gre', hue='admit')
axes = axes.map(plt.scatter, 'cgpa', 'admit')
axes = sns.FacetGrid(toefl_cgpa, col='toefl', row='cgpa', hue='admit')
axes = axes.map(plt.scatter, 'gre', 'admit')
axes = sns.FacetGrid(gre_cgpa, col='gre', row='cgpa', hue='admit')
axes = axes.map(plt.scatter, 'toefl', 'admit')
admission_cor = admission_df.corr()
fig = plt.figure(figsize=(12,8))
axes = sns.heatmap(admission_cor, linewidth=1, linecolor='white', annot=True)
axes.set_title('The Heatmap of Admission Information')
admit_min = admission_df['admit'].min()
admit_avg = admission_df['admit'].mean()
admit_max = admission_df['admit'].max()
admission_df['admit'] = pd.cut(admission_df['admit'], bins=(admit_min-1, admit_avg, admit_max+1), labels=[0,1])
admission_df.count()
min_max_scaler = preprocessing.MinMaxScaler()
gre = admission_df[['gre']].values.astype(float)
gre_scaled = min_max_scaler.fit_transform(gre)
admission_df['gre'] = pd.DataFrame(gre_scaled)

toefl = admission_df[['toefl']].values.astype(float)
toefl_scaled = min_max_scaler.fit_transform(toefl)
admission_df['toefl'] = pd.DataFrame(toefl_scaled)

university_rating = admission_df[['university rating']].values.astype(float)
rating_scaled = min_max_scaler.fit_transform(university_rating)
admission_df['university rating'] = pd.DataFrame(rating_scaled)

sop = admission_df[['sop']].values.astype(float)
sop_scaled = min_max_scaler.fit_transform(sop)
admission_df['sop'] = pd.DataFrame(sop_scaled)

lor = admission_df[['lor']].values.astype(float)
lor_scaled = min_max_scaler.fit_transform(lor)
admission_df['lor'] = pd.DataFrame(lor_scaled)

cgpa = admission_df[['cgpa']].values.astype(float)
cgpa_scaled = min_max_scaler.fit_transform(cgpa)
admission_df['cgpa'] = pd.DataFrame(cgpa_scaled)
admission_df.head()
x = admission_df.drop('admit', axis=1)
y = admission_df['admit']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
gau = GaussianNB()
gau.fit(x_train, y_train)
predict_gau = gau.predict(x_test)
print(classification_report(y_test, predict_gau))
svc = SVC()
svc.fit(x_train, y_train)
predict_svc = svc.predict(x_test)
print(classification_report(y_test, predict_svc))
per = Perceptron()
per.fit(x_train, y_train)
predict_per = per.predict(x_test)
print(classification_report(y_test, predict_per))
log = LogisticRegression()
log.fit(x_train, y_train)
predict_log = log.predict(x_test)
print(classification_report(y_test, predict_log))
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
predict_rf = rf.predict(x_test)
print(classification_report(y_test, predict_rf))
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
predict_dt = dt.predict(x_test)
print(classification_report(y_test, predict_dt))

