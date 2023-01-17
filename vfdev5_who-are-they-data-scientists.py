# Import Python packages
import numpy as np 
import pandas as pd 
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from IPython.core import display as ICD
import warnings

# Bigger than normal fonts
sns.set(font_scale=1.75)

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 5000)
base_dir = '../input/'

fileName = 'multipleChoiceResponses.csv'
filePath = os.path.join(base_dir,fileName)
mcr = pd.read_csv(filePath)
mcr_data = mcr.loc[1:, :]

fileName = 'freeFormResponses.csv'
filePath = os.path.join(base_dir,fileName)
ffr = pd.read_csv(filePath)
ffr_data = ffr.loc[1:, :]

fileName = 'SurveySchema.csv'
filePath = os.path.join(base_dir,fileName)
schema = pd.read_csv(filePath)
# Prepare data types
mcr_data['Time from Start to Finish (seconds)'] = mcr_data['Time from Start to Finish (seconds)'].astype(int)

# Basic trash filtering
# 1) remove answers with 'Time from Start to Finish (seconds)' < 120 seconds and > 3600 seconds
valid_time_mask =  (mcr_data['Time from Start to Finish (seconds)'] > 120) & \
                    (mcr_data['Time from Start to Finish (seconds)'] < 3600)

mcr_data = mcr_data[valid_time_mask].copy().reindex()

is_student = mcr_data['Q6'] == 'Student'
def get_column_names(data, question):
    cols = data.columns[data.columns.str.contains(question)]
    new_cols = []
    for c in cols:
        n = data[c].dropna().values
        if len(n) == 0:
            n = [c]
        new_cols.append(question.split("_")[0] + "_" + n[0])
    return new_cols    
is_datascientist = mcr_data['Q26'].isin(['Probably yes', 'Definitely yes'])
datascientists = mcr_data[is_datascientist]
nb_datascientists_in_world = [len(datascientists), len(mcr_data)]
nb_cats_datascientists = [len(datascientists[datascientists['Q26'] == 'Definitely yes']), 
                          len(datascientists[datascientists['Q26'] == 'Probably yes'])]
nb_student_datascientists_in_students = [len(datascientists[is_student]), len(mcr_data[is_student])]

plt.figure(figsize=(25, 4))
plt.subplot(131)
sns.barplot(y=nb_datascientists_in_world, x=["Data scientists", "World"])
plt.subplot(132)
sns.barplot(y=nb_cats_datascientists, x=["Sure Data scientists", "Probably Data scientists"])
plt.subplot(133)
_ = sns.barplot(y=nb_student_datascientists_in_students, x=["Student data scientists", "All Students"], )
plt.figure(figsize=(20, 20))
plt.grid(which='both')
residence_count = datascientists['Q3'].value_counts().sort_values(ascending=False)
g = sns.barplot(y=residence_count.index.values, x=residence_count)
g.set_xscale('log')
_ = plt.xlabel("Number of data scientists in log10")
import random

random.seed(1)

def get_mean_age(x):
    if "+" in x:
        return int(x[:-1])
    min_max = [int(v)for v in x.split("-")]
    return random.uniform(min_max[0], min_max[1] + 1)

datascientists['mean_age'] = datascientists['Q2'].apply(get_mean_age)

top5_places_datascientists = datascientists[datascientists['Q3'].isin(residence_count.index[:5].values.tolist() + ['Austria', ])]

plt.figure(figsize=(20, 10))
# Show each observation with a scatterplot
sns.boxplot(y="Q3", x="mean_age", data=top5_places_datascientists)
_ = plt.xlabel("Mean Age")
_ = plt.ylabel("Country")
# And the youngest average (precisely, median) data scientist lives in 
res = datascientists.groupby('Q3')['mean_age'].median()
"{} {} {} {}".format(res.argmin(), "and has age:", int(res.min()), "years")
gender_ff = ffr.loc[~ffr['Q1_OTHER_TEXT'].isnull(), 'Q1_OTHER_TEXT'].values[1:].tolist()
gender_ff = [g.lower() for g in gender_ff]
gender_ff = list(set(gender_ff))

wordcloud = WordCloud(max_font_size=40).generate(" ".join(gender_ff))
plt.figure(figsize=(15, 15))
plt.imshow(wordcloud, interpolation="bilinear")
_ = plt.axis("off")
plt.figure(figsize=(20, 10))
plt.grid(which='both')
q6_count = datascientists['Q6'].value_counts().sort_values(ascending=False)
g = sns.barplot(y=q6_count.index.values, x=q6_count)
g.set_xscale('log')
_ = plt.xlabel("Number of data scientists in log10")
cols = [c for c in datascientists.columns if 'Q11_Part' in c]
list_of_activities = datascientists[cols].values.ravel().tolist()
list_of_activities = pd.Series([a for a in list_of_activities if isinstance(a, str)])
activities_counts = list_of_activities.value_counts()


plt.figure(figsize=(20, 12))
plt.title("Activities")
plt.grid(which='both')
g = sns.barplot(y=activities_counts.index.values, x=activities_counts)
g.set_xscale('log')
_ = plt.xlabel("Number of data scientists in log10")
other_activity_ff = ffr.loc[~ffr['Q11_OTHER_TEXT'].isnull(), 'Q11_OTHER_TEXT'].values[1:].tolist()
other_activity_ff = [g.lower() for g in other_activity_ff]
other_activity_ff = list(set(other_activity_ff))

wordcloud = WordCloud(max_font_size=40).generate(" ".join(other_activity_ff))
plt.figure(figsize=(15, 15))
plt.imshow(wordcloud, interpolation="bilinear")
_ = plt.axis("off")
cols = [c for c in datascientists.columns if 'Q16_Part' in c]
list_of_pl = datascientists[cols].values.ravel().tolist()
list_of_pl = pd.Series([ide for ide in list_of_pl if isinstance(ide, str)])
pl_counts = list_of_pl.value_counts()

plt.figure(figsize=(20, 10))
plt.subplot(131)
plt.title("Regular programming language")
plt.grid(which='both')
g = sns.barplot(y=pl_counts.index.values, x=pl_counts)
g.set_xscale('log')
_ = plt.xlabel("Number of data scientists in log10")

plt.subplot(132)
plt.title("Specific programming language")
plt.grid(which='both')
q17_count = datascientists['Q17'].value_counts().sort_values(ascending=False)
g = sns.barplot(y=q17_count.index.values, x=q17_count)
g.set_xscale('log')
_ = plt.xlabel("Number of data scientists in log10")

plt.subplot(133)
plt.title("Recommended programming language")
plt.grid(which='both')
q18_count = datascientists['Q18'].value_counts().sort_values(ascending=False)
g = sns.barplot(y=q18_count.index.values, x=q18_count)
g.set_xscale('log')
_ = plt.xlabel("Number of data scientists in log10")

cols = [c for c in datascientists.columns if 'Q13_Part' in c]
list_of_ide = datascientists[cols].values.ravel().tolist()
list_of_ide = pd.Series([ide for ide in list_of_ide if isinstance(ide, str)])
ide_counts = list_of_ide.value_counts()


plt.figure(figsize=(20, 10))
plt.title("Prefered IDE")
plt.grid(which='both')
g = sns.barplot(y=ide_counts.index.values, x=ide_counts)
g.set_xscale('log')
_ = plt.xlabel("Number of data scientists in log10")
plt.figure(figsize=(20, 10))
plt.title("ML tool")
plt.grid(which='both')
q20_count = datascientists['Q20'].value_counts().sort_values(ascending=False)
g = sns.barplot(y=q20_count.index.values, x=q20_count)
g.set_xscale('log')
_ = plt.xlabel("Number of data scientists in log10")
other_ide_ff = ffr.loc[~ffr['Q13_OTHER_TEXT'].isnull(), 'Q13_OTHER_TEXT'].values[1:].tolist()
other_ide_ff = [g.lower() for g in other_ide_ff]
other_ide_ff = list(set(other_ide_ff))

wordcloud = WordCloud(max_font_size=40).generate(" ".join(other_ide_ff))
plt.figure(figsize=(15, 15))
plt.imshow(wordcloud, interpolation="bilinear")
_ = plt.axis("off")
rec_pl_ff = ffr.loc[~ffr['Q18_OTHER_TEXT'].isnull(), 'Q18_OTHER_TEXT'].values[1:].tolist()
rec_pl_ff = [g.lower() for g in rec_pl_ff]
rec_pl_ff = list(set(rec_pl_ff))

wordcloud = WordCloud(max_font_size=40).generate(" ".join(rec_pl_ff))
plt.figure(figsize=(15, 15))
plt.imshow(wordcloud, interpolation="bilinear")
_ = plt.axis("off")


def has_phd(x):
    return "Yes" if "Doctoral" in x else "No"

plt.figure(figsize=(20, 10))
plt.title("Data scientist has a PhD ?")
plt.grid(which='both')
q4_count = datascientists.loc[~is_student, 'Q4'].apply(has_phd).value_counts().sort_values(ascending=False)
g = sns.barplot(y=q4_count.index.values, x=q4_count)
_ = plt.xlabel("Number of data scientists")
plt.figure(figsize=(20, 10))
plt.title("Industry hires Data Scientists")
plt.grid(which='both')
q7_count = datascientists['Q7'].value_counts().sort_values(ascending=False)
q7_count = q7_count.drop(index="I am a student")
g = sns.barplot(y=q7_count.index.values, x=q7_count)
g.set_xscale('log')
_ = plt.xlabel("Number of data scientists in log10")
cols = [c for c in datascientists.columns if 'Q11_Part' in c]
industry_gb = datascientists.groupby("Q7")[cols]

industries = list(industry_gb.groups.keys())
industries.remove('I am a student')
industry_activity = pd.DataFrame(index=industries, columns=get_column_names(mcr_data, "Q11_Part"), dtype=np.float32)
for g in industries:
    industry_activity.loc[g, :] = industry_gb.get_group(g).count().values / q7_count[g]
    
plt.figure(figsize=(10, 10))
plt.title("Data scientist activity (%) in industry")
_ = sns.heatmap(industry_activity, linewidths=.7, square=True, cmap='coolwarm')
cols = [c for c in datascientists.columns if 'Q19_Part' in c]
industry_gb = datascientists.groupby("Q7")[cols]

industries = list(industry_gb.groups.keys())
industries.remove('I am a student')
industry_activity = pd.DataFrame(index=industries, columns=get_column_names(mcr_data, "Q19_Part"), dtype=np.float32)
for g in industries:
    industry_activity.loc[g, :] = industry_gb.get_group(g).count().values / q7_count[g]

    
plt.figure(figsize=(10, 10))
plt.title("ML Frameworks used in industry by data scientists")
g = sns.heatmap(industry_activity.T, linewidths=.7, square=True, cmap='coolwarm')

cols = [c for c in datascientists.columns if 'Q38_Part' in c]
list_of_media = datascientists[cols].values.ravel().tolist()
list_of_media = pd.Series([m for m in list_of_media if isinstance(m, str)])
media_counts = list_of_media.value_counts()


plt.figure(figsize=(20, 10))
plt.title("Prefered media sources")
plt.grid(which='both')
g = sns.barplot(y=media_counts.index.values, x=media_counts)
g.set_xscale('log')
_ = plt.xlabel("Number of data scientists in log10")
kept_columns = []
for c in mcr_data.columns:
    if "Q" not in c:
        continue
    if mcr_data[c].isin([-1, "-1"]).any():
        continue
    kept_columns.append(c)

data = mcr_data[kept_columns]
# data.head()

print(schema.loc[:0, 'Q38'].values)
# print(schema.loc[:0, 'Q50'].values)
# print(datascientists['Q38_Part_1'])
schema.loc[:0, :]
# gender_df = pd.get_dummies(data['Q1'], prefix="gender")
# age = pd.Series(data['Q2'].apply(get_mean_age), name='mean_age')
# country_df = pd.get_dummies(data['Q3'], prefix="country")

edu_df = pd.get_dummies(data['Q4'], prefix="edu")
edu_field_df = pd.get_dummies(data['Q5'], prefix="edu_field")
work_field_df = pd.get_dummies(data['Q7'], prefix="work_field")

activity_df = (~data[data.columns[data.columns.str.contains("Q11_Part")]].isnull()).astype(int)
activity_df.columns = get_column_names(data, "Q11_Part")

primary_tool_df = pd.get_dummies(data['Q12_MULTIPLE_CHOICE'], prefix="prim_tool")
ide_df = (~data[data.columns[data.columns.str.contains("Q13_Part")]].isnull()).astype(int)
ide_df.columns = get_column_names(data, "Q13_Part")

prog_lang_df = (~data[data.columns[data.columns.str.contains("Q16_Part")]].isnull()).astype(int)
prog_lang_df.columns = get_column_names(data, "Q16_Part")

ml_tool_df = pd.get_dummies(data['Q20'], prefix="ml_tool")

ml_blackbox_df = pd.get_dummies(data['Q48'], prefix="Q48")

ml_metrics_df = (~data[data.columns[data.columns.str.contains("Q42_Part")]].isnull()).astype(int)
ml_metrics_df.columns = get_column_names(data, "Q42_Part")

media_df = (~data[data.columns[data.columns.str.contains("Q38_Part")]].isnull()).astype(int)
media_df.columns = get_column_names(data, "Q38_Part")
train_test_df = pd.concat([
#     gender_df, age, 
#     country_df,
    edu_df, edu_field_df, 
    work_field_df,
    activity_df,
    primary_tool_df, ide_df,
    prog_lang_df, ml_tool_df,
    ml_blackbox_df, 
    ml_metrics_df,
    media_df
], axis=1)
print(train_test_df.shape)
train_test_df.head()
y = is_datascientist.values.astype(int)
"Nb of Data scientists: ", np.sum(y), "All: ", len(y)
from sklearn.model_selection import StratifiedShuffleSplit 

seed = 10
ssplit = StratifiedShuffleSplit(train_size=0.7, random_state=seed)
train_indices, test_indices = next(ssplit.split(train_test_df.values, y))

X_train = train_test_df.values[train_indices]
y_train = y[train_indices]
X_test = train_test_df.values[test_indices]
y_test = y[test_indices]
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

log_reg_cv = LogisticRegressionCV(scoring='roc_auc', random_state=seed, 
                                  solver='liblinear', penalty='l2', 
                                  cv=5, n_jobs=10)
log_reg_cv.fit(X_train, y_train)

print(np.mean(log_reg_cv.scores_[1], axis=0), log_reg_cv.C_)

log_reg = LogisticRegression(C=log_reg_cv.C_[0], random_state=seed, solver='liblinear', penalty='l2')
log_reg.fit(X_train, y_train)

from sklearn.metrics import roc_auc_score, accuracy_score

y_test_preds = log_reg.predict(X_test)
"AUC:", roc_auc_score(y_test, y_test_preds), "Accuracy:", accuracy_score(y_test, y_test_preds)
important_coeffs = np.argsort(np.abs(log_reg.coef_), axis=1)[0, ::-1]
train_test_df.columns.values[important_coeffs[:10]].tolist()
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


rf = RandomForestClassifier(random_state=seed)

max_depth_values = range(5, 10)
n_estimators_values = range(10, 20, 2)
tree_params = {'max_depth': max_depth_values,
               'n_estimators': n_estimators_values}

gs = GridSearchCV(rf, tree_params, scoring='roc_auc', cv=5, n_jobs=10)
gs.fit(X_train, y_train)

gs.best_params_, gs.best_score_, gs.cv_results_['std_test_score'][gs.best_index_]
rf = RandomForestClassifier(random_state=seed, **gs.best_params_)
rf.fit(X_train, y_train)

y_test_preds = rf.predict(X_test)
"AUC:", roc_auc_score(y_test, y_test_preds), "Accuracy:", accuracy_score(y_test, y_test_preds)
rf.feature_importances_
important_coeffs = np.argsort(np.abs(rf.feature_importances_))[::-1]
train_test_df.columns.values[important_coeffs[:10]].tolist()
# This data is just a random stuff :)
df = pd.DataFrame(index=[0], columns=train_test_df.columns)
df.loc[:, :] = 0
df['edu_Doctoral degree'] = 1
df['edu_field_Physics or astronomy'] = 1
df['work_field_Computers/Technology'] = 1
df['Q11_Build and/or run a machine learning service that operationally improves my product or workflows'] = 1
df['Q11_Build prototypes to explore applying machine learning to new areas'] = 1
df['Q11_Do research that advances the state of the art of machine learning'] = 1
df['prim_tool_Cloud-based data software & APIs (AWS, GCP, Azure, etc.)'] = 1
df['prim_tool_Local or hosted development environments (RStudio, JupyterLab, etc.)'] = 1
df['Q13_Jupyter/IPython'] = 1
df['Q13_PyCharm'] = 1
df['Q16_Python'] = 1
df[['ml_tool_PyTorch', 'ml_tool_Xgboost', 'ml_tool_randomForest', 'ml_tool_Scikit-Learn']] = 1
df['Q48_I am confident that I can understand and explain the outputs of many but not all ML models'] = 1
df['Q42_Metrics that consider accuracy'] = 1 
x_me = df.values
res1 = log_reg.predict_proba(x_me)
res2 = rf.predict_proba(x_me)
"Probability that I'm a part of this band is : ", 0.5 * (res1[0, 1] + res2[0, 1])

print(schema.loc[:0, 'Q7'].values)
print(ffr_data.columns)
schema.loc[:0, :]
from sklearn.feature_extraction.text import TfidfVectorizer

def compute_scores(question):
    q = pd.DataFrame(ffr_data[question].dropna().str.lower())
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(q[question])
    q.loc[:, 'tfidf_score'] =  np.array(X.sum(axis=1)).ravel()
    q = q[q['tfidf_score'] > 1.0]
    q = q.sort_values('tfidf_score', ascending=False)
    return q

q11 = compute_scores('Q11_OTHER_TEXT')
for p, s in q11.head(10).values:
    print("{:.2f} - {} \n".format(s, p))
q11.tail(10)
mask = (q11['tfidf_score'] > 2.0) & (q11['tfidf_score'] < 3.0)
for p, s in q11[mask].head(5).values:
    print("{:.2f} - {} \n".format(s, p))
q11[q11['Q11_OTHER_TEXT'].str.contains("fu")]
q12 = compute_scores('Q12_OTHER_TEXT')
for p, s in q12.head(10).values:
    print("{:.2f} - {} \n".format(s, p))
q12.tail(10)
mask = (q12['tfidf_score'] > 2.0) & (q12['tfidf_score'] < 3.0)
for p, s in q12[mask].head(5).values:
    print("{:.2f} - {} \n".format(s, p))
q12[q12['Q12_OTHER_TEXT'].str.contains("kaggle")]
