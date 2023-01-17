import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

from colorama import Fore, Back, Style

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

import lightgbm

from mlxtend.plotting import plot_confusion_matrix

import xgboost

from plotly.offline import plot, iplot, init_notebook_mode

import plotly.graph_objs as go

from plotly.subplots import make_subplots

import plotly.express as px

from statsmodels.formula.api import ols



init_notebook_mode(connected=True)

warnings.filterwarnings("ignore")

import plotly.figure_factory as ff



%matplotlib inline
data = pd.read_csv("../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")

data.head()
data.info()
data['sex'] = data['sex'].astype('category')

data['diabetes'] = data['diabetes'].astype('category')

data['anaemia'] = data['anaemia'].astype('category')

data['high_blood_pressure'] = data['high_blood_pressure'].astype('category')

data['smoking'] = data['smoking'].astype('category')

data['DEATH_EVENT'] = data['DEATH_EVENT'].astype('category')
data.info()
data['age'].describe()
data['age'] = data['age'].astype('int64')
hist_data = [data['age'].values]

group_labels = ['age']



fig = ff.create_distplot(hist_data, group_labels)

fig.update_layout(title_text='Age Distribution')

fig.show()
fig = px.box(data, x='sex', y='age', points='all',)

fig.update_layout(

    title='Gender-wise Age Distribution',

    xaxis=dict(

    tickmode = 'array',

    tickvals = [1,0],

    ticktext = ['Male', 'Female']))

fig.show()
sns.set_style("ticks")

g = sns.catplot(x='sex', y='age', data=data, kind='boxen')

g.set_xticklabels(['Female', 'Male'])

#plt.show(ax, dpi = 200)
male = data[data['sex'] == 1]

female = data[data['sex'] == 0]



male_alive = male[male['DEATH_EVENT'] == 0]

male_dead = male[male['DEATH_EVENT'] == 1]

female_alive = female[female['DEATH_EVENT'] == 0]

female_dead = female[female['DEATH_EVENT'] == 1]



labels = ['Male - Survived', 'Male - Didn\'t survive', 'Female - Survived', 'Female - Didn\'t survive']

values = [len(male_alive), len(male_dead), len(female_alive), len(female_dead)]



fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.2)])

fig.update_layout(title="Gender-wise Survival Chart")

fig.show()
f, ax = plt.subplots(dpi=200)

ax = plt.pie(values, labels=labels, autopct='%1.1f%%', shadow=True)

plt.title("Gender-wise Survival Chart")

plt.show()
surv = pd.concat([male_alive, female_alive])

surv = surv['age']

dead = pd.concat([male_dead, female_dead])

dead = dead['age']



hist_data = [surv, dead]

group_labels = ['Survived', 'Didn\'t Survive']

fig = ff.create_distplot(hist_data, group_labels, bin_size=0.5)

fig.update_layout(title_text='Age-wise Survival state')

fig.show()
f, ax = plt.subplots(dpi=200)

ax = sns.countplot('age', data=data, hue='DEATH_EVENT')

plt.xticks(rotation=90)
fig = px.violin(data, y='age', x='sex', color='DEATH_EVENT', box=True, points="all", hover_data=data.columns)

fig.update_layout(title_text="Gender vs Age on survival status")
#f, ax = plt.subplots(dpi=200)

g = sns.catplot(x='sex', y='age', data=data, kind='violin', hue='DEATH_EVENT')

g.set_xticklabels(['Survived', 'Didn\'t Survive'])

#g.set_title("Sex vs Age Survival Status")

fig = px.violin(data, y='age', x='smoking', color='DEATH_EVENT', box=True, points="all", hover_data=data.columns)

fig.update_layout(title_text = 'Analysis in Age, Smoking and Survival status')

fig.show()
fig = px.box(data, y='age', x='diabetes', color='DEATH_EVENT', 

             notched=True, points='all', hover_data=data.columns, 

             title='Analysis in Age, Diabetes and Survival Status')

fig.show()
fig = px.histogram(data, x='creatinine_phosphokinase',

                  color='DEATH_EVENT', marginal='violin',

                  hover_data=data.columns,

                  title='Histogram for creatinine phosphokinase')

fig.show()
fig = px.histogram(data, x='ejection_fraction', color='DEATH_EVENT',

                  marginal='violin', hover_data=data.columns,

                  title='Histogram for Ejection Fraction')

fig.show()

fig = px.histogram(data, x='platelets', color='DEATH_EVENT',

                  marginal='violin', hover_data=data.columns,

                  title='Histogram for Platelets',

                  color_discrete_sequence=px.colors.sequential.Plasma)

fig.show()
fig = px.histogram(data, x='serum_creatinine', color='DEATH_EVENT',

                  marginal='violin', hover_data=data.columns,

                  title='Histogram for Serum Creatinine',

                  color_discrete_sequence=px.colors.diverging.PuOr)

fig.show()
fig = px.histogram(data, x='serum_sodium', color='DEATH_EVENT',

                  marginal='violin', hover_data=data.columns,

                  title='Histogram for Serum Sodium')

fig.show()
data = data.rename(columns={'DEATH_EVENT':'death'})
surv = data[data['death'] == 0]['serum_sodium']

dead = data[data['death'] == 1]['serum_sodium']

hist_data=[surv,dead]

labels = ['Survived', 'Didn\'t survive']



fig = ff.create_distplot(hist_data, labels, bin_size=0.5)

fig.update_layout(title_text='Serum Sodium vs Survival Status Analysis')

fig.show()
surv = data[data['death'] == 0]['serum_creatinine']

dead = data[data['death'] == 1]['serum_creatinine']



hist_data = [surv, dead]

labels = ['Survived', 'Didn\'t survive']



fig = ff.create_distplot(hist_data, labels, bin_size=0.5)

fig.update_layout(

    title_text='Serum Creatinine vs Survival Status Analysis')

fig.show()
surv = data[data['death'] == 0]['ejection_fraction']

dead = data[data['death'] == 1]['ejection_fraction']



hist_data = [surv, dead]

labels = ['Survived', 'Didn\'t survive']



fig = ff.create_distplot(hist_data, labels, bin_size=0.5)

fig.update_layout(

    title_text='Ejection Fraction vs Survival Status Analysis')

fig.show()
fig = px.pie(data, values='diabetes', names='death', 

             title='Diabetes x Death Chart')

fig.show()
dpos = data[data['diabetes'] == 1]

dneg = data[data['diabetes'] == 0]



labels = ['Positive Diabetes, Survived', 'Positive Diabetes, Didn\'t Survive',

         'Negative Diabetes, Survived', 'Negative Diabetes, Didn\'t Survive']

dposdneg = dpos[dpos['death'] == 0]

dposdpos = dpos[dpos['death'] == 1]

dnegdneg = dneg[dneg['death'] == 0]

dnegdpos = dneg[dneg['death'] == 1]



values = [len(dposdneg),

         len(dposdpos),

         len(dnegdneg),

         len(dnegdpos)]



fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])

fig.update_layout(title_text='Survival x Diabetes Chart')

fig.show()
fig = px.pie(data, values='anaemia', names='death',

            title='Anaemia x Death Chart')

fig.show()
data.anaemia.describe()
apos = data[data['anaemia'] == 1]

aneg = data[data['anaemia'] == 0]



aposdneg = apos[apos['death'] == 0]

aposdpos = apos[apos['death'] == 1]

anegdneg = aneg[aneg['death'] == 0]

anegdpos = aneg[aneg['death'] == 1]



labels = ['Positive Anaemia, Survived', 'Positive Anaemia, Didn\'t Survive',

         'Negative Anaemia, Survived', 'Negative Anaemia, Didn\'t Survive'] 



values = [len(aposdneg), len(aposdpos), len(anegdneg), len(anegdpos)]



fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])

fig.update_layout(

    title_text='Survival x Anaemia Analysis'

)

fig.show()
fig = px.pie(data, values='smoking', names='death',

            title='Smoking x Death Chart')

fig.show()
apos = data[data['smoking'] == 1]

aneg = data[data['smoking'] == 0]



aposdneg = apos[apos['death'] == 0]

aposdpos = apos[apos['death'] == 1]

anegdneg = aneg[aneg['death'] == 0]

anegdpos = aneg[aneg['death'] == 1]



labels = ['Smoked, Survived', 'Smoked, Didn\'t Survive',

         'Didn\'t Smoke, Survived', 'Didn\'t Smoke, Didn\'t Survive'] 



values = [len(aposdneg), len(aposdpos), len(anegdneg), len(anegdpos)]



fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])

fig.update_layout(

    title_text='Survival x Smoking Analysis'

)

fig.show()
fig = px.pie(data, values='high_blood_pressure', names='death',

            title='High Blood Pressure x Death Chart')

fig.show()
apos = data[data['high_blood_pressure'] == 1]

aneg = data[data['high_blood_pressure'] == 0]



aposdneg = apos[apos['death'] == 0]

aposdpos = apos[apos['death'] == 1]

anegdneg = aneg[aneg['death'] == 0]

anegdpos = aneg[aneg['death'] == 1]



labels = ['HBP, Survived', 'HBP, Didn\'t Survive',

         'Not HBP, Survived', 'Not HBP, Didn\'t Survive'] 



values = [len(aposdneg), len(aposdpos), len(anegdneg), len(anegdpos)]



fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])

fig.update_layout(

    title_text='Survival x High Blood Pressure Analysis'

)

fig.show()
plt.figure(dpi=200)

sns.heatmap(data.corr(), vmin=-1, cmap=sns.color_palette("plasma_r"), annot=True)
#choice = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']

index_vals = data['death'].cat.codes





fig = go.Figure(data=go.Splom(

    dimensions=[

        dict(label='Age', values=data['age']),

        dict(label='Creatinine Phosphokinase', values=data['creatinine_phosphokinase']),

        dict(label='Ejection Fration', values=data['ejection_fraction']),

        dict(label='Platelets', values=data['platelets']),

        dict(label='Serum Creatinine', values=data['serum_creatinine']),

        dict(label='Serum Sodium', values=data['serum_sodium'])    

    ],

    text=data['death']

))



fig.update_layout(

    title='Heart Failure Data set',

    dragmode='select',

    width=1000,

    height=1000,

    hovermode='closest',

)



fig.show()

data.info()
features = ['diabetes', 'ejection_fraction', 'high_blood_pressure',

           'serum_creatinine', 'age', 'time']



features = ['time', 'age', 'ejection_fraction', 'serum_creatinine', 'platelets']



X = data[features]

y = data.death
X_train, X_test, y_train, y_test = train_test_split(X,y, 

                                                    test_size=0.2,

                                                   random_state=2698)
clf = RandomForestClassifier(max_features=2, max_depth=15,

                            random_state=1)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)



print("Accuracy of RandomForestClassifier is : ",

     clf.score(X_test, y_test))
cm = confusion_matrix(y_test, y_pred)

plt.figure()

plot_confusion_matrix(cm, figsize=(10,6), hide_ticks=True,

                     cmap = plt.cm.Blues)

plt.title("Random Forest Model - Confusion Matrix")

plt.xticks(range(2), ["Heart didn\'t fail", "Heart failed"],

          fontsize=16)

plt.yticks(range(2), ["Heart didn\'t fail", "Heart failed"],

          fontsize=16)

plt.show()
gradientboost_clf = GradientBoostingClassifier(max_depth=2,

                                              random_state=4)

gradientboost_clf.fit(X_train, y_train)

gradientboost_pred = gradientboost_clf.predict(X_test)



print("Accuracy of Gradient Boosting is : ",

     gradientboost_clf.score(X_test, y_test))
cm = confusion_matrix(y_test, gradientboost_pred)

plt.figure()

plot_confusion_matrix(cm, figsize=(10,6), hide_ticks=True,

                    cmap=plt.cm.Blues)

plt.title("Gradient Boost Model - Confusion Matrix")

plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.show()
xgb_clf = xgboost.XGBRFClassifier(max_depth=3, random_state=0)

xgb_clf.fit(X_train, y_train)

xgb_pred = xgb_clf.predict(X_test)



print("Accuracy of XGBRFClassifier is : ", xgb_clf.score(X_test,

                                                        y_test))
cm = confusion_matrix(y_test, xgb_pred)

plt.figure()

plot_confusion_matrix(cm, figsize=(10,6), hide_ticks=True,

                     cmap=plt.cm.Blues)

plt.title("XGBRFClassifier Model - Confusion Matrix")

plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.show()
lgb_clf = lightgbm.LGBMClassifier(max_depth=2, random_state=1)

lgb_clf.fit(X_train, y_train)

lgb_pred = lgb_clf.predict(X_test)



print("Accuracy of LGBMClassifier is : ", 

     lgb_clf.score(X_test, y_test))
cm = confusion_matrix(y_test, lgb_pred)

plt.figure()

plot_confusion_matrix(cm, figsize=(10,6), hide_ticks=True,

                     cmap=plt.cm.Blues)

plt.title("LGBMClassifier Model - Confusion Matrix")

plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.show()