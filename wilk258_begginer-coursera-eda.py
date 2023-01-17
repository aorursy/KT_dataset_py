# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt 
import seaborn as sns
import squarify 
df = pd.read_csv('/kaggle/input/coursera-course-dataset/coursea_data.csv')
df.head(3)
fig, (ax1, ax2,ax3) = plt.subplots(3, 1,figsize=(10,20))


sns.countplot(df['course_Certificate_type'], palette = 'seismic', ax=ax1)
plt.title('course_Certificate_type count', fontsize = 20)

sns.countplot(df['course_difficulty'], palette = 'gnuplot', ax=ax2)
plt.title('course_difficulty count', fontsize = 20)


sns.countplot(df['course_rating'], palette = 'PuRd')
plt.title('course_rating count', fontsize = 20)
print(df.course_Certificate_type.value_counts())
pie = df.groupby('course_Certificate_type').size()
plt.figure()
pie.plot(kind='pie', subplots=True, figsize=(8, 8))
plt.title("Pie Chart of course_Certificate_type")
plt.ylabel("")
tree = df.groupby('course_difficulty').size().reset_index(name='counts')
labels = tree.apply(lambda x: str(x[0]) + "\n (" + str(x[1]) + ")", axis=1)
sizes = tree['counts'].values.tolist()
colors = [plt.cm.Spectral(i/float(len(labels))) for i in range(len(labels))]

# Draw Plot
plt.figure(figsize=(12,8), dpi= 80)
squarify.plot(sizes=sizes, label=labels, color=colors, alpha=.8)

# Decorate
plt.title('Treemap of Course Difficulty')
plt.axis('off')
df.course_students_enrolled = df.course_students_enrolled.apply(lambda x : float(str(x).replace('k', '').replace('m',''))*1000)
course_pop = df.groupby('course_organization')['course_students_enrolled'].sum().reset_index()
Top15_popular= course_pop.sort_values(by='course_students_enrolled', ascending=False).head(15)
Top15_unpopular=course_pop.sort_values(by='course_students_enrolled', ascending=True).head(15)
fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(10,15))
sns.barplot(x=Top15_popular["course_students_enrolled"],y=Top15_popular['course_organization'],ax=ax1)
ax1.set_title("Top 15 Popular Organization")

sns.barplot(x=Top15_unpopular["course_students_enrolled"],y=Top15_unpopular['course_organization'],ax=ax2)
ax2.set_title("Top 15 Unpopular Organization")
df2 = df[['course_students_enrolled', 'course_organization']].groupby('course_organization').apply(lambda x: x.mean())
df2.sort_values('course_students_enrolled', inplace=True)
df2.reset_index(inplace=True)

# Draw plot
fig, ax = plt.subplots(figsize=(35,25), dpi= 80)
ax.hlines(y=df2.index, xmin=11, xmax=26, color='gray', alpha=0.7, linewidth=1, linestyles='dashdot')
ax.scatter(y=df2.index, x=df2.course_students_enrolled, s=75, color='firebrick', alpha=0.7)

# Title, Label, Ticks and Ylim
ax.set_title('Dot Plot for Entrollment for Every Organization', fontdict={'size':22})
ax.set_xlabel('Entrollment number')
ax.set_yticks(df2.index)
ax.set_yticklabels(df.course_organization.str.title(), fontdict={'horizontalalignment': 'right'})
ax.set_xlim(0, 250000)
def plot_count(feature, title, df, size=1, show_percents=False):
    f, ax = plt.subplots(1,1, figsize=(4*size,4))
    total = float(len(df))
    g = sns.countplot(df[feature], order = df[feature].value_counts().index[0:20], palette='Set3')
    g.set_title("Number of {}".format(title))
    if(size > 2):
        plt.xticks(rotation=90, size=10)
    if(show_percents):
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x()+p.get_width()/2.,
                    height + 3,
                    '{:1.2f}%'.format(100*height/total),
                    ha="center") 
    ax.set_xticklabels(ax.get_xticklabels());
    plt.show()    
plot_count('course_title', 'Top 20 course_title', df, 3.5)
x_var = 'course_rating'
groupby_var = 'course_difficulty'
df_agg = df.loc[:, [x_var, groupby_var]].groupby(groupby_var)
vals = [df[x_var].values.tolist() for i, df in df_agg]

# Draw
plt.figure(figsize=(20,10), dpi= 80)
colors = [plt.cm.Spectral(i/float(len(vals)-1)) for i in range(len(vals))]
n, bins, patches = plt.hist(vals, 30, stacked=True, density=False, color=colors[:len(vals)])

# Decoration
plt.legend({group:col for group, col in zip(np.unique(df[groupby_var]).tolist(), colors[:len(vals)])})
plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$", fontsize=22)
plt.xlabel(x_var)
plt.ylabel("Frequency")
plt.ylim(0, 265)