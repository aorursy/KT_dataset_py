import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
pdf = pd.read_csv('/kaggle/input/titanic/train.csv',index_col=False)

pdf.columns.values, len(pdf)
total_people = len(pdf)

num_males = len(pdf[pdf['Sex']=='male'])

num_females = len(pdf[pdf['Sex']=='female'])

surv_males = len(pdf[(pdf['Sex']=='male') & (pdf['Survived']==1)])

surv_females = len(pdf[(pdf['Sex']=='female') & (pdf['Survived']==1)])

print("Males: %.2f percent" % (100*num_males/total_people))

print("Females %.2f percent" % (100*num_females/total_people))

print("Males Survived: %.2f percent" % (100*surv_males/num_males))

print("Females Survived %.2f percent" % (100*surv_females/num_females))
surv = pdf[pdf['Survived']==True]

ded = pdf[pdf['Survived']==False]

plt.clf()

plt.scatter(ded['Fare'],ded['Age'],color='r')

plt.show()

plt.clf()

plt.scatter(surv['Fare'],surv['Age'],color='g')

plt.show()
required_column = 'Parch'

x_label = 'Parents and Children'

columns = [required_column]+ ['Survived','PassengerId']

counts_df = pdf[columns].groupby(columns[:-1]).count().reset_index()



temp_dict = {}

missing_data = []

for val in zip(counts_df[required_column],counts_df['Survived']):

    temp_dict[val] = True

for val in counts_df[required_column].unique():

    if (val,0) not in temp_dict:

        missing_data.append([val,0,0])

    elif (val,1) not in temp_dict:

        missing_data.append([val,1,0])

temp_df = pd.DataFrame(missing_data,columns = columns)

counts_df = counts_df.append(temp_df).sort_values(by=required_column,kind='mergesort')



Values = sorted(pdf[columns[0]].unique())



barWidth = 0.25

bar_1_positions = np.arange(len(Values))

bar_2_positions = [position + barWidth for position in bar_1_positions]



plt.clf()

plt.xlabel(x_label)

plt.xticks(bar_1_positions,Values)

plt.bar(bar_1_positions, counts_df[counts_df['Survived']==True]['PassengerId'], color='#2d7f5e', width=barWidth, edgecolor='white', label='Survived')

plt.bar(bar_2_positions, counts_df[counts_df['Survived']==False]['PassengerId'], color='#557f2d', width=barWidth, edgecolor='white', label='Dead')

plt.legend()

plt.show()
required_column = 'SibSp'

x_label = 'Siblings'

columns = [required_column]+ ['Survived','PassengerId']

counts_df = pdf[columns].groupby(columns[:-1]).count().reset_index()



temp_dict = {}

missing_data = []

for val in zip(counts_df[required_column],counts_df['Survived']):

    temp_dict[val] = True

for val in counts_df[required_column].unique():

    if (val,0) not in temp_dict:

        missing_data.append([val,0,0])

    elif (val,1) not in temp_dict:

        missing_data.append([val,1,0])

temp_df = pd.DataFrame(missing_data,columns = columns)

counts_df = counts_df.append(temp_df).sort_values(by=required_column,kind='mergesort')



Values = sorted(pdf[columns[0]].unique())



barWidth = 0.25

bar_1_positions = np.arange(len(Values))

bar_2_positions = [position + barWidth for position in bar_1_positions]



plt.clf()

plt.xlabel(x_label)

plt.xticks(bar_1_positions,Values)

plt.bar(bar_1_positions, counts_df[counts_df['Survived']==False]['PassengerId'], color='#2d7f5e', width=barWidth, edgecolor='white', label='Dead')

plt.bar(bar_2_positions, counts_df[counts_df['Survived']==True]['PassengerId'], color='#557f2d', width=barWidth, edgecolor='white', label='Survived')

plt.legend()

plt.show()
plt.rcParams['figure.figsize'] = [13, 7]

pdf['Familial'] = pdf['SibSp'] + pdf['Parch']



required_column = 'Familial'

x_label = 'Family Size'



columns = [required_column]+ ['Survived','PassengerId']



male_counts_df = pdf[pdf['Sex']=='male'][columns].groupby(columns[:-1]).count().reset_index()

temp_dict = {}

missing_data = []

for val in zip(male_counts_df[required_column],male_counts_df['Survived']):

    temp_dict[val] = True

for val in male_counts_df[required_column].unique():

    if (val,0) not in temp_dict:

        missing_data.append([val,0,0])

    elif (val,1) not in temp_dict:

        missing_data.append([val,1,0])

temp_df = pd.DataFrame(missing_data,columns = columns)

male_counts_df = male_counts_df.append(temp_df).sort_values(by=required_column,kind='mergesort')



female_counts_df = pdf[pdf['Sex']=='female'][columns].groupby(columns[:-1]).count().reset_index()

temp_dict = {}

missing_data = []

for val in zip(female_counts_df[required_column],female_counts_df['Survived']):

    temp_dict[val] = True

for val in female_counts_df[required_column].unique():

    if (val,0) not in temp_dict:

        missing_data.append([val,0,0])

    elif (val,1) not in temp_dict:

        missing_data.append([val,1,0])

temp_df = pd.DataFrame(missing_data,columns = columns)

female_counts_df = female_counts_df.append(temp_df).sort_values(by=required_column,kind='mergesort')



Values = sorted(pdf[columns[0]].unique())



barWidth = 0.2

bar_1_positions = np.arange(len(Values))

bar_2_positions = [position + barWidth for position in bar_1_positions]

bar_3_positions = [position + barWidth for position in bar_2_positions]

bar_4_positions = [position + barWidth for position in bar_3_positions]



plt.clf()

plt.xlabel(x_label)

plt.xticks(bar_1_positions,Values)

plt.bar(bar_1_positions, male_counts_df[male_counts_df['Survived']==True]['PassengerId'], color='#2d7f5e', width=barWidth, edgecolor='white', label='Male Surv')

plt.bar(bar_2_positions, male_counts_df[male_counts_df['Survived']==False]['PassengerId'], color='#8b0000', width=barWidth, edgecolor='white', label='Male Deaths')

plt.bar(bar_3_positions, female_counts_df[female_counts_df['Survived']==True]['PassengerId'], color='#557f2d', width=barWidth, edgecolor='white', label='Female Surv')

plt.bar(bar_4_positions, female_counts_df[female_counts_df['Survived']==False]['PassengerId'], color='#ffcccb', width=barWidth, edgecolor='white', label='Female Dead')

plt.legend()

plt.show()
plt.rcParams['figure.figsize'] = [13, 7]

required_column = 'Pclass'

x_label = 'Social Class'



columns = [required_column]+ ['Survived','PassengerId']



male_counts_df = pdf[pdf['Sex']=='male'][columns].groupby(columns[:-1]).count().reset_index()

temp_dict = {}

missing_data = []

for val in zip(male_counts_df[required_column],male_counts_df['Survived']):

    temp_dict[val] = True

for val in male_counts_df[required_column].unique():

    if (val,0) not in temp_dict:

        missing_data.append([val,0,0])

    elif (val,1) not in temp_dict:

        missing_data.append([val,1,0])

temp_df = pd.DataFrame(missing_data,columns = columns)

male_counts_df = male_counts_df.append(temp_df).sort_values(by=required_column,kind='mergesort')



female_counts_df = pdf[pdf['Sex']=='female'][columns].groupby(columns[:-1]).count().reset_index()

temp_dict = {}

missing_data = []

for val in zip(female_counts_df[required_column],female_counts_df['Survived']):

    temp_dict[val] = True

for val in female_counts_df[required_column].unique():

    if (val,0) not in temp_dict:

        missing_data.append([val,0,0])

    elif (val,1) not in temp_dict:

        missing_data.append([val,1,0])

temp_df = pd.DataFrame(missing_data,columns = columns)

female_counts_df = female_counts_df.append(temp_df).sort_values(by=required_column,kind='mergesort')



Values = sorted(pdf[columns[0]].unique())



barWidth = 0.2

bar_1_positions = np.arange(len(Values))

bar_2_positions = [position + barWidth for position in bar_1_positions]

bar_3_positions = [position + barWidth for position in bar_2_positions]

bar_4_positions = [position + barWidth for position in bar_3_positions]



plt.clf()

plt.xlabel(x_label,fontsize=15,fontweight='bold')

plt.xticks(bar_1_positions,Values)



print(female_counts_df[female_counts_df['Survived']==False]['PassengerId'], (female_counts_df[female_counts_df['Survived']==True]['PassengerId']))



plt.bar(bar_1_positions, male_counts_df[male_counts_df['Survived']==True]['PassengerId'], color='#2d7f5e', width=barWidth, edgecolor='white', label='Male Surv')

plt.bar(bar_2_positions, male_counts_df[male_counts_df['Survived']==False]['PassengerId'], color='#8b0000', width=barWidth, edgecolor='white', label='Male Deaths')

plt.bar(bar_3_positions, female_counts_df[female_counts_df['Survived']==True]['PassengerId'], color='#557f2d', width=barWidth, edgecolor='white', label='Female Surv')

plt.bar(bar_4_positions, female_counts_df[female_counts_df['Survived']==False]['PassengerId'], color='#e75480', width=barWidth, edgecolor='white', label='Female Dead')

plt.legend(prop={'weight':'bold'})

plt.show()
import seaborn as sns

pdf['Sex_Code']=pdf.Sex.astype('category').cat.codes

try:

    sns.heatmap(pdf[['Fare','Age','Familial','Sex_Code','Pclass','Survived']].corr(method='pearson'),annot=True)

except KeyError as e:

    print(e)