import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

from matplotlib import patches as patches
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dat = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv', dtype='object')



dat.drop(dat.index[0], inplace=True)
dat['Q4'] = dat['Q4'].apply(lambda x : x if x != 'Some college/university study without earning a bachelor’s degree' else "Bachelor's degree")

dat['Q4'] = dat['Q4'].apply(lambda x : x if x != 'Bachelor’s degree' else "Bachelor's degree")

dat['Q4'] = dat['Q4'].apply(lambda x : x if x != 'Master’s degree' else "Master's degree")

dat['Q4'] = dat['Q4'].apply(lambda x : x if x != 'No formal education past high school' else "High school")



dat['Q3'] = dat['Q3'].apply(lambda x : x if x != 'United Kingdom of Great Britain and Northern Ireland' else 'UK')

dat['Q3'] = dat['Q3'].apply(lambda x : x if x != 'United States of America' else 'US')

dat['Q3'] = dat['Q3'].apply(lambda x : x if x != 'Viet Nam' else 'Vietnam')

dat['Q3'] = dat['Q3'].apply(lambda x : x if x != 'Hong Kong (S.A.R.)' else 'Hong Kong')



dat['Age'] = dat['Q1'].astype('category').cat.as_ordered()

dat['Gender'] = dat['Q2']

dat['Country'] = dat['Q3']
age = dat['Age'].value_counts().reset_index()



#fig size

fig, ax = plt.subplots(figsize=(8,5))



b = ax.barh(age.iloc[:,0], age.iloc[:,1], color='silver', edgecolor='black', height=0.7)

b[0].set_color('blue')

b[0].set_edgecolor('black')

b[0].set_linewidth(1)

b[1].set_color('blue')

b[1].set_edgecolor('black')

b[1].set_linewidth(1)



# Remove axes spines

for s in ['top','bottom','left','right']:

    ax.spines[s].set_visible(False)

    

# Remove x,y ticks

ax.xaxis.set_ticks_position('none')

ax.yaxis.set_ticks_position('none')



# Add padding between the axes and labels

ax.xaxis.set_tick_params(pad=5)

ax.yaxis.set_tick_params(pad=10)



# Add x,y gridlines

ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.3)



# Show top values

ax.invert_yaxis()



# Add plot title

ax.set_title("Age", loc='center', pad=10, fontsize=13)

ax.set_xlabel('No of repsondents')



# Add annotation to bars

for i in ax.patches:

    ax.text(i.get_width()+15, i.get_y()+0.5, str(round(i.get_width(),2)), 

           fontsize=10, fontweight='bold', color='grey')

    

z = [x for x in range(0,11)]

plt.yticks(z, list(age.iloc[:,0]), weight='bold')

plt.show()
gender = dat['Gender'].value_counts().reset_index()
#fig size

fig, ax = plt.subplots(figsize=(8,5))



b = ax.barh(gender.iloc[:,0], gender.iloc[:,1], color='silver', edgecolor='black', height=0.7)

b[0].set_color('blue')

b[0].set_edgecolor('black')

b[0].set_linewidth(1)



# Remove axes spines

for s in ['top','bottom','left','right']:

    ax.spines[s].set_visible(False)

    

# Remove x,y ticks

ax.xaxis.set_ticks_position('none')

ax.yaxis.set_ticks_position('none')



# Add padding between the axes and labels

ax.xaxis.set_tick_params(pad=5)

ax.yaxis.set_tick_params(pad=10)



# Add x,y gridlines

ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.3)



# Show top values

ax.invert_yaxis()



# Add plot title

ax.set_title("Gender", loc='center', pad=10, fontsize=13)

ax.set_xlabel('No of repsondents')



# Add annotation to bars

for i in ax.patches:

    ax.text(i.get_width()+15, i.get_y()+0.5, str(round(i.get_width(),2)), 

           fontsize=10, fontweight='bold', color='grey')

    

z = [x for x in range(0,4)]

plt.yticks(z, list(gender.iloc[:,0]), weight='bold')

plt.show()
country = dat['Country'].value_counts().reset_index()[:20]
#fig size

fig, ax = plt.subplots(figsize=(8,5))



b = ax.barh(country.iloc[:,0], country.iloc[:,1], color='silver', edgecolor='black', height=0.7)

b[0].set_color('blue')

b[0].set_edgecolor('black')

b[0].set_linewidth(1)

b[1].set_color('blue')

b[1].set_edgecolor('black')

b[1].set_linewidth(1)



# Remove axes spines

for s in ['top','bottom','left','right']:

    ax.spines[s].set_visible(False)

    

# Remove x,y ticks

ax.xaxis.set_ticks_position('none')

ax.yaxis.set_ticks_position('none')



# Add padding between the axes and labels

ax.xaxis.set_tick_params(pad=5)

ax.yaxis.set_tick_params(pad=10)



# Add x,y gridlines

ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.3)



# Show top values

ax.invert_yaxis()



# Add plot title

ax.set_title("The top 20 Countries in ML", loc='center', pad=10, fontsize=13)

ax.set_xlabel('No of repsondents')



# Add annotation to bars

for i in ax.patches:

    ax.text(i.get_width()+15, i.get_y()+0.5, str(round(i.get_width(),2)), 

           fontsize=10, fontweight='bold', color='grey')

    

z = [x for x in range(0,20)]

plt.yticks(z, list(country.iloc[:,0]), weight='bold')

plt.show()
degree = pd.DataFrame(columns=['degree', 'count', 'percentage'])



degree['degree'] = dat['Q4'].value_counts().index

degree['count'] = dat['Q4'].value_counts().values

degree['percentage'] = dat['Q4'].value_counts().values/dat['Q4'].value_counts().sum()



degree.index = degree['degree']

degree = degree.drop(columns='degree')

degree['percentage'] = degree['percentage']*100

# degree_sort = degree[['percentage']].sort_values(by='percentage', ascending=False)

degree_sort = degree[['count']].sort_values(by='count', ascending=False)

degree_rst = degree_sort.reset_index()
#fig size

fig, ax = plt.subplots(figsize=(8,5))



# Horizontal Bar Plot

b = ax.barh(degree_rst.iloc[:,0], degree_rst.iloc[:,1], color='silver', edgecolor='black', height=0.7)

b[0].set_color('blue')

# b[0].set_edgecolor('black')

b[0].set_linewidth(1)

b[4].set_color('gainsboro')

# b[4].set_edgecolor('black')





# Remove axes spines

for s in ['top','bottom','left','right']:

    ax.spines[s].set_visible(False)

    

# Remove x,y ticks

ax.xaxis.set_ticks_position('none')

ax.yaxis.set_ticks_position('none')



# Add padding between the axes and labels

ax.xaxis.set_tick_params(pad=5)

ax.yaxis.set_tick_params(pad=10)



# Add x,y gridlines

ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.3)



# Show top values

ax.invert_yaxis()



# Add plot title

ax.set_title("Student's highest education level", loc='center', pad=10, fontsize=13)

ax.set_xlabel('No of repsondents')



# Add annotation to bars

for i in ax.patches:

    ax.text(i.get_width()+15, i.get_y()+0.5, str(round(i.get_width(),2)), 

           fontsize=10, fontweight='bold', color='grey')

    

z = [x for x in range(0,6)]

plt.yticks(z, list(degree_rst.iloc[:,0]), weight='bold')

plt.show()
dat_ide = pd.DataFrame(columns=['IDE','count','percentage'])



for i in range(1,12):

    dat_ide = dat_ide.append({'IDE':dat['Q13_Part_{}'.format(i)].mode()[0],'count':dat['Q13_Part_{}'.format(i)].count(),'percentage':dat['Q13_Part_{}'.format(i)].count()/len(dat)},ignore_index=True)



dat_ide.index = dat_ide['IDE']

dat_ide = dat_ide.drop(columns='IDE')

dat_ide['percentage'] = dat_ide['percentage']*100

dat_ide_sort = dat_ide[['percentage']].sort_values(by='percentage', ascending=False)

dat_ide_rst = dat_ide_sort.reset_index()
#fig size

fig, ax = plt.subplots(figsize=(8,5))



# Horizontal Bar Plot

b = ax.barh(dat_ide_rst.iloc[:,0], dat_ide_rst.iloc[:,1], color='silver', edgecolor='black', height=0.7)

b[0].set_color('blue')

b[0].set_edgecolor('black')

b[0].set_linewidth(1)

b[1].set_color('blue')

b[1].set_edgecolor('black')

b[1].set_linewidth(1)

b[2].set_color('blue')

b[2].set_edgecolor('black')

b[2].set_linewidth(1)

b[3].set_color('cyan')

b[3].set_edgecolor('black')

b[3].set_linewidth(1)

b[8].set_color('gainsboro')



# Remove axes spines

for s in ['top','bottom','left','right']:

    ax.spines[s].set_visible(False)

    

# Remove x,y ticks

ax.xaxis.set_ticks_position('none')

ax.yaxis.set_ticks_position('none')



# Add padding between the axes and labels

ax.xaxis.set_tick_params(pad=5)

ax.yaxis.set_tick_params(pad=10)



# Add x,y gridlines

ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.3)



# Show top values

ax.invert_yaxis()



# Add plot title

ax.set_title("Student's favorite Data Science platforms", loc='center', pad=10, fontsize=13)

ax.set_xlabel('Percentage')



# Add annotation to bars

for i in ax.patches:

    ax.text(i.get_width()+0.5, i.get_y()+0.5, str(round(i.get_width(),2)), 

           fontsize=10, fontweight='bold', color='grey')

z = [x for x in range(0,11)]

plt.yticks(z, list(dat_ide_rst.iloc[:,0]), weight='bold')

plt.show()
media = pd.DataFrame(columns=['media','count','percentage'])



for i in range(1,12):

    media = media.append({'media':dat['Q12_Part_{}'.format(i)].mode()[0],'count':dat['Q12_Part_{}'.format(i)].count(),'percentage':dat['Q13_Part_{}'.format(i)].count()/len(dat)},ignore_index=True)



media.index = media['media']

media = media.drop(columns='media')

media['percentage'] = media['percentage']*100

media_sort = media[['percentage']].sort_values(by='percentage', ascending=False)

media_rst = media_sort.reset_index()
#fig size

fig, ax = plt.subplots(figsize=(8,5))



# Horizontal Bar Plot

b = ax.barh(media_rst.iloc[:,0], media_rst.iloc[:,1], color='silver', edgecolor='black', height=0.7)

b[0].set_color('blue')

b[0].set_edgecolor('black')

b[0].set_linewidth(1)

b[1].set_color('blue')

b[1].set_edgecolor('black')

b[1].set_linewidth(1)

b[2].set_color('blue')

b[2].set_edgecolor('black')

b[2].set_linewidth(1)

b[3].set_color('cyan')

b[3].set_edgecolor('black')

b[3].set_linewidth(1)

b[8].set_color('gainsboro')



# Remove axes spines

for s in ['top','bottom','left','right']:

    ax.spines[s].set_visible(False)

    

# Remove x,y ticks

ax.xaxis.set_ticks_position('none')

ax.yaxis.set_ticks_position('none')



# Add padding between the axes and labels

ax.xaxis.set_tick_params(pad=5)

ax.yaxis.set_tick_params(pad=10)



# Add x,y gridlines

ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.3)



# Show top values

ax.invert_yaxis()



# Add plot title

ax.set_title("Media sources for ML", loc='center', pad=10, fontsize=13)

ax.set_xlabel('Percentage')



# Add annotation to bars

for i in ax.patches:

    ax.text(i.get_width()+0.5, i.get_y()+0.5, str(round(i.get_width(),2)), 

           fontsize=10, fontweight='bold', color='grey')

z = [x for x in range(0,11)]

plt.yticks(z, list(media_rst.iloc[:,0]), weight='bold')



fig.text(0.7, 0.7, 'Top media sources', fontsize=12, color='blue', ha='center', va='top')

plt.axhline(y=2.5, color='blue', linestyle='-.')



plt.show()
prog_lang = pd.DataFrame(columns=['language', 'count'])



for i in range(1,13):

    prog_lang = prog_lang.append({'language':dat['Q18_Part_{}'.format(i)].mode()[0], 'count':dat['Q18_Part_{}'.format(i)].count()}, ignore_index=True)



prog_lang.index = prog_lang['language']

prog_lang = prog_lang.drop(columns=['language'])

lang_sort = prog_lang.sort_values(by='count', ascending=False)

lang_rst = lang_sort.reset_index()
# fig size

fig, ax = plt.subplots(figsize=(8,5))





# Horizontal Bar Plot

dat_ide_rst = dat_ide_sort.reset_index()

b = ax.barh(lang_rst.iloc[:,0], lang_rst.iloc[:,1], color='silver', edgecolor='black', height=0.7)

b[0].set_color('blue')

b[0].set_edgecolor('black')

b[0].set_linewidth(1)

b[11].set_color('gainsboro')



# Remove axes spines

for s in ['top','bottom','left','right']:

    ax.spines[s].set_visible(False)

    

# Remove x,y ticks

ax.xaxis.set_ticks_position('none')

ax.yaxis.set_ticks_position('none')



# Add padding between the axes and labels

ax.xaxis.set_tick_params(pad=5)

ax.yaxis.set_tick_params(pad=10)



# Add x,y gridlines

ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.3)



# Show top values

ax.invert_yaxis()



# Add plot title

ax.set_title("Programming Language Used Regularly", loc='center', pad=10, fontsize=13)

ax.set_xlabel('# of Users')



# Add annotation to bars

for i in ax.patches:

    ax.text(i.get_width()+20, i.get_y()+0.5, str(round(i.get_width(),2)), 

           fontsize=10, fontweight='bold', color='grey')

z = [x for x in range(0,12)]

plt.yticks(z, list(lang_rst.iloc[:,0]), weight='bold')

plt.show()
lang = dat['Q19'].value_counts().reset_index()
# fig size

fig, ax = plt.subplots(figsize=(8,5))





# Horizontal Bar Plot

b = ax.barh(lang.iloc[:,0], lang.iloc[:,1], color='silver', edgecolor='black', height=0.7)

b[0].set_color('blue')

b[0].set_edgecolor('black')

b[0].set_linewidth(1)

b[11].set_color('gainsboro')



# Remove axes spines

for s in ['top','bottom','left','right']:

    ax.spines[s].set_visible(False)

    

# Remove x,y ticks

ax.xaxis.set_ticks_position('none')

ax.yaxis.set_ticks_position('none')



# Add padding between the axes and labels

ax.xaxis.set_tick_params(pad=5)

ax.yaxis.set_tick_params(pad=10)



# Add x,y gridlines

ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.3)



# Show top values

ax.invert_yaxis()



# Add plot title

ax.set_title("Programming Language Recommendation", loc='center', pad=10, fontsize=13)

ax.set_xlabel('# of Users')



# Add annotation to bars

for i in ax.patches:

    ax.text(i.get_width()+20, i.get_y()+0.5, str(round(i.get_width(),2)), 

           fontsize=10, fontweight='bold', color='grey')

z = [x for x in range(0,12)]

plt.yticks(z, list(lang_rst.iloc[:,0]), weight='bold')

plt.show()
visual = pd.DataFrame(columns=['tool', 'count'])



for i in range(1,13):

    visual = visual.append({'tool':dat['Q20_Part_{}'.format(i)].mode()[0], 'count':dat['Q20_Part_{}'.format(i)].count()}, ignore_index=True)



visual.index = visual['tool']

visual = visual.drop(columns=['tool'])

visual_sort = visual.sort_values(by='count', ascending=False)

visual_rst = visual_sort.reset_index()
# fig size

fig, ax = plt.subplots(figsize=(8,5))



# Horizontal Bar Plot

b = ax.barh(visual_rst.iloc[:,0], visual_rst.iloc[:,1], color='silver', edgecolor='black', height=0.7)

b[0].set_color('blue')

b[0].set_edgecolor('black')

b[0].set_linewidth(1)

b[1].set_color('blue')

b[1].set_edgecolor('black')

b[1].set_linewidth(1)

b[2].set_color('blue')

b[2].set_edgecolor('black')

b[2].set_linewidth(1)

b[3].set_color('blue')

b[3].set_edgecolor('black')

b[3].set_linewidth(1)

b[5].set_color('gainsboro')



# Remove axes spines

for s in ['top','bottom','left','right']:

    ax.spines[s].set_visible(False)

    

# Remove x,y ticks

ax.xaxis.set_ticks_position('none')

ax.yaxis.set_ticks_position('none')



# Add padding between the axes and labels

ax.xaxis.set_tick_params(pad=5)

ax.yaxis.set_tick_params(pad=10)



# Add x,y gridlines

ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.3)



# Show top values

ax.invert_yaxis()



# Add plot title

ax.set_title("Data Visualization tools used regularly", loc='center', pad=10, fontsize=13)

ax.set_xlabel('# of Users')



# Add annotation to bars

for i in ax.patches:

    ax.text(i.get_width()+20, i.get_y()+0.5, str(round(i.get_width(),2)), 

           fontsize=10, fontweight='bold', color='grey')

z = [x for x in range(0,12)]

plt.yticks(z, list(visual_rst.iloc[:,0]), weight='bold')



fig.text(0.70, 0.655, 'Mostly Used Libraries', fontsize=12, color='blue', ha='center', va='top', fontstyle='italic')

plt.axhline(y=3.5, color='blue', linestyle='-.')



plt.show()
code_exp = dat['Q15'].value_counts().reset_index()
#fig size

fig, ax = plt.subplots(figsize=(8,5))



b = ax.barh(code_exp.iloc[:,0], code_exp.iloc[:,1], color='silver', edgecolor='black', height=0.7)

b[0].set_color('blue')

b[0].set_linewidth(1)

b[1].set_color('blue')

b[1].set_linewidth(1)

b[5].set_color('gainsboro')

# b[5].set_edgecolor('black')



# Remove axes spines

for s in ['top','bottom','left','right']:

    ax.spines[s].set_visible(False)

    

# Remove x,y ticks

ax.xaxis.set_ticks_position('none')

ax.yaxis.set_ticks_position('none')



# Add padding between the axes and labels

ax.xaxis.set_tick_params(pad=5)

ax.yaxis.set_tick_params(pad=10)



# Add x,y gridlines

ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.3)



# Show top values

ax.invert_yaxis()



# Add plot title

ax.set_title("Student's coding", loc='center', pad=10, fontsize=13)

ax.set_xlabel('No of repsondents')



# Add annotation to bars

for i in ax.patches:

    ax.text(i.get_width()+15, i.get_y()+0.5, str(round(i.get_width(),2)), 

           fontsize=10, fontweight='bold', color='grey')

    

z = [x for x in range(0,7)]

plt.yticks(z, list(code_exp.iloc[:,0]), weight='bold')

plt.show()
ml_algo = pd.DataFrame(columns=['algo', 'count'])



for i in range(1,13):

    ml_algo = ml_algo.append({'algo':dat['Q24_Part_{}'.format(i)].mode()[0], 'count':dat['Q24_Part_{}'.format(i)].count()}, ignore_index=True)



# ml_algo

ml_algo.index = ml_algo['algo']

ml_algo = ml_algo.drop(columns='algo')

ml_algo_sort = ml_algo.sort_values(by='count', ascending=False)

ml_algo_rst = ml_algo_sort.reset_index()
# fig size

fig, ax = plt.subplots(figsize=(8,5))



# Horizontal Bar Plot

b = ax.barh(ml_algo_rst.iloc[:,0], ml_algo_rst.iloc[:,1], color='silver', edgecolor='black', height=0.7)

b[0].set_color('blue')

b[0].set_edgecolor('black')

b[0].set_linewidth(1)

b[1].set_color('blue')

b[1].set_edgecolor('black')

b[1].set_linewidth(1)

b[2].set_color('blue')

b[2].set_edgecolor('black')

b[2].set_linewidth(1)

b[3].set_color('blue')

b[3].set_edgecolor('black')

b[3].set_linewidth(1)

b[7].set_color('gainsboro')



# Remove axes spines

for s in ['top','bottom','left','right']:

    ax.spines[s].set_visible(False)

    

# Remove x,y ticks

ax.xaxis.set_ticks_position('none')

ax.yaxis.set_ticks_position('none')



# Add padding between the axes and labels

ax.xaxis.set_tick_params(pad=5)

ax.yaxis.set_tick_params(pad=10)



# Add x,y gridlines

ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.3)



# Show top values

ax.invert_yaxis()



# Add plot title

ax.set_title("Machine Learning algos used regularly by students", loc='center', pad=10, fontsize=13)

ax.set_xlabel('# of Users')



# Add annotation to bars

for i in ax.patches:

    ax.text(i.get_width()+20, i.get_y()+0.5, str(round(i.get_width(),2)), 

           fontsize=10, fontweight='bold', color='grey')

z = [x for x in range(0,12)]

plt.yticks(z, list(ml_algo_rst.iloc[:,0]), weight='bold')



fig.text(0.70, 0.655, 'Mostly Used Algos', fontsize=12, color='blue', ha='center', va='top', fontstyle='italic')

plt.axhline(y=3.5, color='blue', linestyle='-.')



plt.show()
job = dat['Q5'].value_counts().reset_index()
#fig size

fig, ax = plt.subplots(figsize=(8,5))



b = ax.barh(job.iloc[:,0], job.iloc[:,1], color='silver', edgecolor='black', height=0.7)

b[0].set_color('blue')

b[0].set_edgecolor('black')

b[0].set_linewidth(1)

b[1].set_color('cyan')

b[1].set_edgecolor('black')

b[1].set_linewidth(1)

b[2].set_color('blue')

b[2].set_edgecolor('black')

b[2].set_linewidth(1)

b[3].set_color('gainsboro')

b[4].set_color('blue')

b[4].set_edgecolor('black')

b[4].set_linewidth(1)

b[5].set_color('blue')

b[5].set_edgecolor('black')

b[5].set_linewidth(1)



# Remove axes spines

for s in ['top','bottom','left','right']:

    ax.spines[s].set_visible(False)

    

# Remove x,y ticks

ax.xaxis.set_ticks_position('none')

ax.yaxis.set_ticks_position('none')



# Add padding between the axes and labels

ax.xaxis.set_tick_params(pad=5)

ax.yaxis.set_tick_params(pad=10)



# Add x,y gridlines

ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.3)



# Show top values

ax.invert_yaxis()



# Add plot title

ax.set_title("Job/Occupation of users", loc='center', pad=10, fontsize=13)

ax.set_xlabel('No of repsondents')



# Add annotation to bars

for i in ax.patches:

    ax.text(i.get_width()+15, i.get_y()+0.5, str(round(i.get_width(),2)), 

           fontsize=10, fontweight='bold', color='grey')

    

z = [x for x in range(0,12)]

plt.yticks(z, list(job.iloc[:,0]), weight='bold')

plt.show()
ml_incorp = dat['Q8'].value_counts().reset_index()
#fig size

fig, ax = plt.subplots(figsize=(7,4))



b = ax.barh(ml_incorp.iloc[:,0], ml_incorp.iloc[:,1], color='silver', edgecolor='black', height=0.7)

b[0].set_color('cyan')

b[0].set_edgecolor('black')

b[0].set_linewidth(1)

b[1].set_color('blue')

b[1].set_edgecolor('black')

b[1].set_linewidth(1)

b[2].set_color('blue')

b[2].set_edgecolor('black')

b[2].set_linewidth(1)

b[5].set_color('gainsboro')



# Remove axes spines

for s in ['top','bottom','left','right']:

    ax.spines[s].set_visible(False)

    

# Remove x,y ticks

ax.xaxis.set_ticks_position('none')

ax.yaxis.set_ticks_position('none')



# Add padding between the axes and labels

ax.xaxis.set_tick_params(pad=5)

ax.yaxis.set_tick_params(pad=10)



# Add x,y gridlines

ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.3)



# Show top values

ax.invert_yaxis()



# Add plot title

ax.set_title("Machine Learning Incorporation", loc='center', pad=10, fontsize=13)

ax.set_xlabel('No of repsondents')



# Add annotation to bars

for i in ax.patches:

    ax.text(i.get_width()+15, i.get_y()+0.5, str(round(i.get_width(),2)), 

           fontsize=10, fontweight='bold', color='grey')

    

z = [x for x in range(0,6)]

plt.yticks(z, list(ml_incorp.iloc[:,0]), weight='bold')

plt.show()
size_coy = dat['Q6'].value_counts().reset_index()
#fig size

fig, ax = plt.subplots(figsize=(8,5))



b = ax.barh(size_coy.iloc[:,0], size_coy.iloc[:,1], color='silver', edgecolor='black', height=0.7)

b[0].set_color('blue')

b[0].set_edgecolor('black')

b[0].set_linewidth(1)

b[1].set_color('blue')

b[1].set_edgecolor('black')

b[1].set_linewidth(1)





# Remove axes spines

for s in ['top','bottom','left','right']:

    ax.spines[s].set_visible(False)

    

# Remove x,y ticks

ax.xaxis.set_ticks_position('none')

ax.yaxis.set_ticks_position('none')



# Add padding between the axes and labels

ax.xaxis.set_tick_params(pad=5)

ax.yaxis.set_tick_params(pad=10)



# Add x,y gridlines

ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.3)



# Show top values

ax.invert_yaxis()



# Add plot title

ax.set_title("Size of companies with established ML", loc='center', pad=10, fontsize=13)

ax.set_xlabel('No of repsondents')



# Add annotation to bars

for i in ax.patches:

    ax.text(i.get_width()+15, i.get_y()+0.5, str(round(i.get_width(),2)), 

           fontsize=10, fontweight='bold', color='grey')

    

z = [x for x in range(0,5)]

plt.yticks(z, list(size_coy.iloc[:,0]), weight='bold')

plt.show()
ml_cost = dat['Q11'].value_counts().reset_index()
#fig size

fig, ax = plt.subplots(figsize=(8,5))



b = ax.barh(ml_cost.iloc[:,0], ml_cost.iloc[:,1], color='silver', edgecolor='black', height=0.7)

b[0].set_color('blue')

b[0].set_edgecolor('black')

b[0].set_linewidth(1)



# Remove axes spines

for s in ['top','bottom','left','right']:

    ax.spines[s].set_visible(False)

    

# Remove x,y ticks

ax.xaxis.set_ticks_position('none')

ax.yaxis.set_ticks_position('none')



# Add padding between the axes and labels

ax.xaxis.set_tick_params(pad=5)

ax.yaxis.set_tick_params(pad=10)



# Add x,y gridlines

ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.3)



# Show top values

ax.invert_yaxis()



# Add plot title

ax.set_title("ML and cloud computing cost in companies", loc='center', pad=10, fontsize=13)

ax.set_xlabel('No of repsondents')



# Add annotation to bars

for i in ax.patches:

    ax.text(i.get_width()+15, i.get_y()+0.5, str(round(i.get_width(),2)), 

           fontsize=10, fontweight='bold', color='grey')

    

z = [x for x in range(0,6)]

plt.yticks(z, list(ml_cost.iloc[:,0]), weight='bold')

plt.show()
cloud_plt = pd.DataFrame(columns=['platform', 'count'])



for i in range(1,13):

    cloud_plt = cloud_plt.append({'platform':dat['Q29_Part_{}'.format(i)].mode()[0], 'count':dat['Q29_Part_{}'.format(i)].count()}, ignore_index=True)



cloud_plt.index = cloud_plt['platform']

cloud_plt = cloud_plt.drop(columns='platform')

cloud_plt_sort = cloud_plt.sort_values(by='count', ascending=False)

cloud_plt_rst = cloud_plt_sort.reset_index()
# fig size

fig, ax = plt.subplots(figsize=(8,5))



# Horizontal Bar Plot

b = ax.barh(cloud_plt_rst.iloc[:,0], cloud_plt_rst.iloc[:,1], color='silver', edgecolor='black', height=0.7)

b[0].set_color('blue')

b[0].set_edgecolor('black')

b[0].set_linewidth(1)

b[1].set_color('gainsboro')

b[2].set_color('blue')

b[2].set_edgecolor('black')

b[2].set_linewidth(1)



# Remove axes spines

for s in ['top','bottom','left','right']:

    ax.spines[s].set_visible(False)

    

# Remove x,y ticks

ax.xaxis.set_ticks_position('none')

ax.yaxis.set_ticks_position('none')



# Add padding between the axes and labels

ax.xaxis.set_tick_params(pad=5)

ax.yaxis.set_tick_params(pad=10)



# Add x,y gridlines

ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.3)



# Show top values

ax.invert_yaxis()



# Add plot title

ax.set_title("Cloud Platforms", loc='center', pad=10, fontsize=13)

ax.set_xlabel('# of Users')



# Add annotation to bars

for i in ax.patches:

    ax.text(i.get_width()+20, i.get_y()+0.5, str(round(i.get_width(),2)), 

           fontsize=10, fontweight='bold', color='grey')

z = [x for x in range(0,12)]

plt.yticks(z, list(cloud_plt_rst.iloc[:,0]), weight='bold')



plt.show()