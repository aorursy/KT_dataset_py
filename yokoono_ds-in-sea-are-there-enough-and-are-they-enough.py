# importing packages

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import textwrap

import warnings

warnings.filterwarnings('ignore')



# making new compensation bins - creating less number of bins so that it's easier to see

def new_compensation_bin(comp_str):

    if comp_str in ['$0-999', '1,000-1,999', '2,000-2,999', '3,000-3,999', '4,000-4,999']:

        return '$0-4,999'

    elif comp_str in ['5,000-7,499', '7,500-9,999', '10,000-14,999', '15,000-19,999', '20,000-24,999']:

        return '$5,000-24,999'

    elif comp_str in ['25,000-29,999', '30,000-39,999', '40,000-49,999', '50,000-59,999', '60,000-69,999']:

        return '$25,000-69,999'

    elif comp_str in ['70,000-79,999', '80,000-89,999', '90,000-99,999', '100,000-124,999', '125,000-149,999']:

        return '$70,000-149,999'

    elif comp_str in ['150,000-199,999', '200,000-249,999', '250,000-299,999', '300,000-500,000', '> $500,000']:

        return '> $150,000'

    else:

        return "Null"



# identifying a tag for the southeast asian countries

def tag_sea_countries(country):

    if country in ["Brunei", "Cambodia", "East Timor", "Indonesia", "Laos", "Malaysia", "Myanmar", "Philippines", "Singapore", "Thailand", "Viet Nam"]:

        return "SEA"

    else:

        return "RoW"



df = pd.read_csv("/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv", skiprows=[0])    

df["sea_tag"] = df.apply(lambda x: tag_sea_countries(x["In which country do you currently reside?"]), axis=1)

df["new_compensation_bin"] = df.apply(lambda x: new_compensation_bin(x["What is your current yearly compensation (approximate $USD)?"]), axis=1)



sea_countries = df[df["sea_tag"]=="SEA"]

nonsea_countries = df[df["sea_tag"]=="RoW"]
### COMPARING THE INCREASE IN NUMBER OF SURVEY PARTICIPANTS FROM 2017 TO 2018 ###



# dataset for 2017

df_2017 = pd.read_csv("/kaggle/input/kaggle-survey-2017/multipleChoiceResponses.csv", encoding = "ISO-8859-1")

df_2017["sea_tag"] = df_2017.apply(lambda x: tag_sea_countries(x["Country"]), axis=1)

df_2017["survey_year"] = 2017

sea_2017 = df_2017[df_2017["sea_tag"]=="SEA"]

sea_2017_ = sea_2017[["survey_year", "Country"]]

sea_2017_.columns = ["survey_year", "country"]

sea_2017_grouped = (sea_2017_.groupby("survey_year")["country"].count()).reset_index(name="count")

sea_2017_grouped["total"] = len(df_2017)

sea_2017_grouped["percent_sea"] = round((sea_2017_grouped["count"]/sea_2017_grouped["total"])*100,2)



# dataset for 2018

df_2018 = pd.read_csv("/kaggle/input/kaggle-survey-2018/multipleChoiceResponses.csv", skiprows=[0])    

df_2018["sea_tag"] = df_2018.apply(lambda x: tag_sea_countries(x["In which country do you currently reside?"]), axis=1)

df_2018["survey_year"] = 2018

sea_2018 = df_2018[df_2018["sea_tag"]=="SEA"]

sea_2018_ = sea_2018[["survey_year", "In which country do you currently reside?"]]

sea_2018_.columns = ["survey_year", "country"]

sea_2018_grouped = (sea_2018_.groupby("survey_year")["country"].count()).reset_index(name="count")

sea_2018_grouped["total"] = len(df_2018)

sea_2018_grouped["percent_sea"] = round((sea_2018_grouped["count"]/sea_2018_grouped["total"])*100,2)



#dataset for 2019

sea_2019 = sea_countries

sea_2019["survey_year"] = 2019

sea_2019_ = sea_2019[["survey_year", "In which country do you currently reside?"]]

sea_2019_.columns = ["survey_year", "country"]

sea_2019_grouped = (sea_2019_.groupby("survey_year")["country"].count()).reset_index(name="count")

sea_2019_grouped["total"] = len(df)

sea_2019_grouped["percent_sea"] = round((sea_2019_grouped["count"]/sea_2019_grouped["total"])*100,2)



# 2017-2019

sea_2017to2019 = (pd.concat([sea_2017_grouped, sea_2018_grouped, sea_2019_grouped])).reset_index(drop=True)



# plotting

gridsize = (10, 3)

fig = plt.figure(figsize=(40, 10))



ax1 = plt.subplot2grid(gridsize, (3, 1), rowspan=6)

ax1 = sns.barplot(x = "survey_year", 

                  y= "percent_sea",

                  data=sea_2017to2019,

                  color = "#529FCD"

                   )

ax1_title = ax1.set_title('Percentage of Respondents of SEA Countries per Year')

ax1_yticks = ax1.set_yticks([])

ax1_yticklabels = ax1.set_yticklabels([])

ax1_ylabel = ax1.set_ylabel("")

ax1_xlabel = ax1.set_xlabel("")

ax1.grid(False)

ax1.set_ylim(0,4)



for p in ax1.patches:

    patch = ax1.annotate(str(p.get_height())+"%", 

                        (p.get_x() + p.get_width() / 2.0, 

                         p.get_height()), 

                        ha = 'center', 

                        va = 'center', 

                        xytext = (0, 5),

                        textcoords = 'offset points')

sns.despine(ax=ax1, left=True, top=True, right=True,bottom=False)
from IPython.display import Image

Image("/kaggle/input/dsinsea/sea_chessboard.png", width=600)
### COMPARING THE 2019 SURVEY PARTICIPANTS PER SEA COUNTRY ###



gridsize = (5, 5)

fig = plt.figure(figsize=(20, 15))



ax1 = plt.subplot2grid(gridsize, (0, 2), colspan=5, rowspan=2)

ax1 = sns.countplot(x = "In which country do you currently reside?", 

                    data = sea_countries,

                    order = sea_countries["In which country do you currently reside?"].value_counts().index,

                    color = "#529FCD"

                   )

ax1_title = ax1.set_title('Number of Respondents per SEA Countries')

ax1_yticklabels = ax1.set_yticklabels([])

ax1_ylabel = ax1.set_ylabel("")

ax1_xlabel = ax1.set_xlabel("")

ax1.grid(False)

sns.despine(ax=ax1, left=True, top=True, right=True,bottom=True)



for p in ax1.patches:

    patch = ax1.annotate(p.get_height(), 

                        (p.get_x() + p.get_width() / 2.0, 

                         p.get_height()), 

                        ha = 'center', 

                        va = 'center', 

                        xytext = (0, 5),

                        textcoords = 'offset points')



### COMPARING THE COUNTRIES OF 2019 SURVEY PARTICIPANTS - SEA, ROW ###    

    

ax2 = plt.subplot2grid(gridsize, (0, 1))

ax2_set_title = ax2.set_title('Countries')

grouped_countries = (df.groupby('sea_tag')["In which country do you currently reside?"].nunique()).reset_index(name="count_country")

ax2 = plt.pie(grouped_countries['count_country'],

              labels=grouped_countries['sea_tag'],

              shadow=False,

              startangle=0,

              autopct='%1.2f%%',

              colors=["#A8A495", "#E3692A"]

             )



### COMPARING THE PARTICIPANTS OF 2019 SURVEY - SEA, ROW ###    



ax3 = plt.subplot2grid(gridsize, (1, 1))

ax3_set_title = ax3.set_title('Respondents')

grouped_respondents = (df.groupby('sea_tag')["In which country do you currently reside?"].count()).reset_index(name="count_respondents")

ax3 = plt.pie(grouped_respondents['count_respondents'],

              labels=grouped_respondents['sea_tag'],

              shadow=False,

              startangle=0,

              autopct='%1.2f%%',

              colors=["#A8A495", "#E3692A"],

             )
### COMPARING THE GENDERS OF 2019 SURVEY PARTICIPANTS - SEA, ROW ###    



gridsize = (10, 3)

fig = plt.figure(figsize=(40, 10))





# RoW

ax1 = plt.subplot2grid(gridsize, (0, 1))

ax1_title = ax1.set_title('Gender Breakdown')

ax1_start = 0

ax1_never = round((len(nonsea_countries[nonsea_countries["What is your gender? - Selected Choice"]=="Female"])/len(nonsea_countries[nonsea_countries["What is your gender? - Selected Choice"].isin(["Female", "Male"])]))*100)

ax1_seldom = round((len(nonsea_countries[nonsea_countries["What is your gender? - Selected Choice"]=="Male"])/len(nonsea_countries[nonsea_countries["What is your gender? - Selected Choice"].isin(["Female", "Male"])]))*100)

ax1.broken_barh([(ax1_start, ax1_never), (ax1_never, ax1_never+ax1_seldom)], [10, 9], facecolors=('#E3692A', '#529FCD'))

ax1.set_xlim(0, 100)

ax1.spines['left'].set_visible(False)

ax1.spines['bottom'].set_visible(False)

ax1.spines['top'].set_visible(False)

ax1.spines['right'].set_visible(False)

ax1.set_yticks([15, 20])

ax1.set_xticks([0, 25, 50, 75, 100])

ax1.set_axisbelow(True) 

ax1.set_xticklabels("")

ax1.set_yticklabels(['RoW'])

ax1.grid(axis='x')

ax1.text(ax1_never-6, 14.5, str(ax1_never)+"%", fontsize=8)

ax1.text((ax1_never+ax1_seldom)-6, 14.5, str(ax1_seldom)+"%", fontsize=8)



# SEA

ax2 = plt.subplot2grid(gridsize, (1, 1))

ax2_start = 0

ax2_never = round((len(sea_countries[sea_countries["What is your gender? - Selected Choice"]=="Female"])/len(sea_countries[sea_countries["What is your gender? - Selected Choice"].isin(["Female", "Male"])]))*100)

ax2_seldom = round((len(sea_countries[sea_countries["What is your gender? - Selected Choice"]=="Male"])/len(sea_countries[sea_countries["What is your gender? - Selected Choice"].isin(["Female", "Male"])]))*100)

ax2.broken_barh([(ax2_start, ax2_never), (ax2_never, ax2_never+ax2_seldom)], [10, 9], facecolors=('#E3692A', '#529FCD'))

ax2.set_xlim(0, 100)

ax2.spines['left'].set_visible(False)

ax2.spines['bottom'].set_visible(False)

ax2.spines['top'].set_visible(False)

ax2.spines['right'].set_visible(False)

ax2.set_yticks([15, 20])

ax2.set_xticks([0, 25, 50, 75, 100])

ax2.set_axisbelow(True) 

ax2.set_yticklabels(['SEA'])

ax2.grid(axis='x')

ax2.text(ax2_never-6, 14.5, str(ax2_never)+"%", fontsize=8)

ax2.text((ax2_never+ax2_seldom)-6, 14.5, str(ax2_seldom)+"%", fontsize=8)



### COMPARING THE GENDERS OF 2019 SURVEY PARTICIPANTS per SEA coutry ###    



ax3 = plt.subplot2grid(gridsize, (3, 1), rowspan=6)

r = [0,1,2,3,4,5]

raw_data = {'Female': 

            [len(sea_countries[(sea_countries["What is your gender? - Selected Choice"]=="Female")&

                               (sea_countries["In which country do you currently reside?"]=="Indonesia")]),

             len(sea_countries[(sea_countries["What is your gender? - Selected Choice"]=="Female")&

                               (sea_countries["In which country do you currently reside?"]=="Malaysia")]),

             len(sea_countries[(sea_countries["What is your gender? - Selected Choice"]=="Female")&

                               (sea_countries["In which country do you currently reside?"]=="Philippines")]),

             len(sea_countries[(sea_countries["What is your gender? - Selected Choice"]=="Female")&

                               (sea_countries["In which country do you currently reside?"]=="Singapore")]),

             len(sea_countries[(sea_countries["What is your gender? - Selected Choice"]=="Female")&

                               (sea_countries["In which country do you currently reside?"]=="Thailand")]),

             len(sea_countries[(sea_countries["What is your gender? - Selected Choice"]=="Female")&

                               (sea_countries["In which country do you currently reside?"]=="Viet Nam")])], 

            'Male': 

            [len(sea_countries[(sea_countries["What is your gender? - Selected Choice"]=="Male")&

                               (sea_countries["In which country do you currently reside?"]=="Indonesia")]),

             len(sea_countries[(sea_countries["What is your gender? - Selected Choice"]=="Male")&

                               (sea_countries["In which country do you currently reside?"]=="Malaysia")]),

             len(sea_countries[(sea_countries["What is your gender? - Selected Choice"]=="Male")&

                               (sea_countries["In which country do you currently reside?"]=="Philippines")]),

             len(sea_countries[(sea_countries["What is your gender? - Selected Choice"]=="Male")&

                               (sea_countries["In which country do you currently reside?"]=="Singapore")]),

             len(sea_countries[(sea_countries["What is your gender? - Selected Choice"]=="Male")&

                               (sea_countries["In which country do you currently reside?"]=="Thailand")]),

             len(sea_countries[(sea_countries["What is your gender? - Selected Choice"]=="Male")&

                               (sea_countries["In which country do you currently reside?"]=="Viet Nam")])]

           }

raw_data_df = pd.DataFrame(raw_data)

totals = [i+j for i,j in zip(raw_data_df['Female'], raw_data_df['Male'])]

greenBars = [round(i / j * 100) for i,j in zip(raw_data_df['Female'], totals)]

orangeBars = [round(i / j * 100) for i,j in zip(raw_data_df['Male'], totals)]

barWidth = 0.85

names = ('Indonesia', 'Malaysia', 'Philippines', 'Singapore', 'Thailand', 'Viet Nam')

ax3.bar(r, greenBars, color='#E3692A', edgecolor='white', width=barWidth, label="Female")

ax3.bar(r, orangeBars, bottom=greenBars, color='#529FCD', edgecolor='white', width=barWidth, label="Male")

ax3_title = ax3.set_title('Gender Breakdown per SEA Countries')

ax3_xticks = plt.xticks(r, names)

ax3_legend = plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)



for r in range(0,6,1):

    p = ax3.patches[r]

    patch = ax3.annotate(str(p.get_height())+"%", 

                        (p.get_x() + p.get_width()/2.0, 

                         p.get_height()/2.0), 

                        ha = 'center', 

                        va = 'center')

    q = ax3.patches[r+6]

    patch = ax3.annotate(str(q.get_height())+"%", 

                        (q.get_x() + q.get_width()/2.0, 

                         p.get_height() + (q.get_height()/2.0)), 

                        ha = 'center', 

                        va = 'center')

ax3.grid(False)

ax3_yticklabels = ax3.set_yticklabels([])



sns.despine(ax=ax3, left=True, top=True, right=True,bottom=True)

sns.despine(ax=ax2, left=True, top=True, right=True,bottom=True)

sns.despine(ax=ax1, left=True, top=True, right=True,bottom=True)

ax1.grid(False)

ax1_xticklabels = ax1.set_xticklabels([])

ax2.grid(False)

ax2_xticklabels = ax2.set_xticklabels([])
### COMPARING THE AGE OF 2019 SURVEY PARTICIPANTS - SEA, ROW ###    



column_tosee = "What is your age (# years)?"

short_column_tosee = "age"



# doing all of this to get a pivot for a barh subplot!!!

sea_nonsea_agedata = (df.groupby(['sea_tag', column_tosee])["sea_tag"].count()).reset_index(name="count")

sea_nonsea_agedata.columns = ["sea_tag", short_column_tosee, "count"]

pivoted_sea_nonsea_agedata = (sea_nonsea_agedata.pivot(index='sea_tag', columns=short_column_tosee, values='count')).reset_index()

pivoted_sea_nonsea_agedata.loc[:,'Total'] = pivoted_sea_nonsea_agedata.sum(axis=1)

answers = df[column_tosee].dropna().unique().tolist()

for a in answers:

    pivoted_sea_nonsea_agedata[a] = round((pivoted_sea_nonsea_agedata[a]/pivoted_sea_nonsea_agedata["Total"])*100, 2)

pivoted_sea_nonsea_agedata.reset_index(drop=True, inplace=True)

pivoted_sea_nonsea_agedata.set_index("sea_tag", inplace=True)

pivoted_sea_nonsea_agedata = pivoted_sea_nonsea_agedata.rename_axis(None, axis=1)

pivoted_sea_nonsea_agedata = pivoted_sea_nonsea_agedata.rename_axis(None, axis=0)

pivoted_sea_nonsea_agedata = pivoted_sea_nonsea_agedata.drop(['Total'], axis=1)

pivoted_sea_nonsea_agedata = pivoted_sea_nonsea_agedata.loc[['SEA', 'RoW'], :]



gridsize = (2, len(answers))

fig = plt.figure(figsize=(20, 7))

fig.suptitle('Age: SEA-RoW', fontsize=14)



sns.set_style("whitegrid")



ax_frames = []

counter = 0

for ans in answers:

    ax = plt.subplot2grid(gridsize, (0, counter))

    counter += 1

    ax_frames.append(ax)



# barh barh barh

sns.set_style("whitegrid")

pivoted_sea_nonsea_agedata.plot(kind='barh', subplots=True, sharey=True, layout=(1,len(ax_frames)), legend=False, 

                                 xticks=[], yticks=[], ax=ax_frames,

                                 grid=False, xlim=(0, 35), edgecolor='none', fontsize=14,

                                 color = sns.light_palette(sns.color_palette("Blues_r")[0], len(ax_frames))[::-1]

                                )



sns.despine(left=False, top=True, right=True, bottom=True)



# labels!!!

for a in ax_frames:

    for p in a.patches:

        patch = a.annotate(str(p.get_width())+"%", 

                            (p.get_width(), 

                             p.get_y() + p.get_height()), 

                            ha = 'center', 

                            va = 'center', 

                            xytext = (20, -12),

                            textcoords = 'offset points'

                            )
### COMPARING THE EDUCATION OF 2019 SURVEY PARTICIPANTS - SEA, ROW ###    



column_tosee = "What is the highest level of formal education that you have attained or plan to attain within the next 2 years?"

short_column_tosee = "formal_education"



# doing all of this to get a pivot for a barh subplot!!!

sea_nonsea_educdata = (df.groupby(['sea_tag', column_tosee])["sea_tag"].count()).reset_index(name="count")

sea_nonsea_educdata.columns = ["sea_tag", short_column_tosee, "count"]

pivoted_sea_nonsea_educdata = (sea_nonsea_educdata.pivot(index='sea_tag', columns=short_column_tosee, values='count')).reset_index()

pivoted_sea_nonsea_educdata.loc[:,'Total'] = pivoted_sea_nonsea_educdata.sum(axis=1)

answers = df[column_tosee].dropna().unique().tolist()

for a in answers:

    pivoted_sea_nonsea_educdata[a] = round((pivoted_sea_nonsea_educdata[a]/pivoted_sea_nonsea_educdata["Total"])*100, 2)

pivoted_sea_nonsea_educdata.reset_index(drop=True, inplace=True)

pivoted_sea_nonsea_educdata.set_index("sea_tag", inplace=True)

pivoted_sea_nonsea_educdata = pivoted_sea_nonsea_educdata.rename_axis(None, axis=1)

pivoted_sea_nonsea_educdata = pivoted_sea_nonsea_educdata.rename_axis(None, axis=0)

pivoted_sea_nonsea_educdata = pivoted_sea_nonsea_educdata.drop(['Total'], axis=1)

pivoted_sea_nonsea_educdata = pivoted_sea_nonsea_educdata[[

    "No formal education past high school", 

    "Professional degree",

    "Some college/university study without earning a bachelor’s degree", 

    "Bachelor’s degree", 

    "Master’s degree", 

    "Doctoral degree", 

#     "I prefer not to answer"

]]

pivoted_sea_nonsea_educdata.columns = [

    "No formal education \npast high school", 

    "Professional degree",

    "Some college/university \nstudy without earning \na bachelor’s degree", 

    "Bachelor’s degree", 

    "Master’s degree", 

    "Doctoral degree", 

#     "I prefer not to answer"

]

pivoted_sea_nonsea_educdata = pivoted_sea_nonsea_educdata.loc[['SEA', 'RoW'], :]



gridsize = (2, len(pivoted_sea_nonsea_educdata.columns.tolist()))

fig = plt.figure(figsize=(20, 10))

sns.set_style("whitegrid")

fig.suptitle('Formal Education: SEA-RoW', fontsize=14)





ax_frames = []

counter = 0

for ans in pivoted_sea_nonsea_educdata.columns.tolist():

    ax = plt.subplot2grid(gridsize, (0, counter))

    counter += 1

    ax_frames.append(ax)



# barh barh barh

sns.set_style("whitegrid")

pivoted_sea_nonsea_educdata.plot(kind='barh', subplots=True, sharey=True, layout=(1,len(ax_frames)), legend=False, 

                                 xticks=[], yticks=[], 

                                 ax=ax_frames,

                                 grid=False, xlim=(0, 55), edgecolor='none', fontsize=14,

                                 color = sns.light_palette(sns.color_palette("Blues_r")[0], len(ax_frames))[::-1]

                                )



sns.despine(left=False, top=True, right=True, bottom=True)



# labels!!!

for a in ax_frames:

    for p in a.patches:

        patch = a.annotate(str(p.get_width())+"%", 

                            (p.get_width(), 

                             p.get_y() + p.get_height()), 

                            ha = 'center', 

                            va = 'center', 

                            xytext = (20, -12),

                            textcoords = 'offset points'

                            )
### COMPARING THE COMPENSATION OF 2019 SURVEY PARTICIPANTS - SEA, ROW ###    



column_tosee = "new_compensation_bin"

short_column_tosee = "compensation"



# doing all of this to get a pivot for a barh subplot!!!

sea_nonsea_compensationdata = (df.groupby(['sea_tag', column_tosee])["sea_tag"].count()).reset_index(name="count")

sea_nonsea_compensationdata.columns = ["sea_tag", short_column_tosee, "count"]

pivoted_sea_nonsea_compensationdata = (sea_nonsea_compensationdata.pivot(index='sea_tag', columns=short_column_tosee, values='count')).reset_index()

pivoted_sea_nonsea_compensationdata.loc[:,'Total'] = pivoted_sea_nonsea_compensationdata.sum(axis=1)

answers = df[column_tosee].dropna().unique().tolist()

for a in answers:

    pivoted_sea_nonsea_compensationdata[a] = round((pivoted_sea_nonsea_compensationdata[a]/pivoted_sea_nonsea_compensationdata["Total"])*100, 2)

pivoted_sea_nonsea_compensationdata.reset_index(drop=True, inplace=True)

pivoted_sea_nonsea_compensationdata.set_index("sea_tag", inplace=True)

pivoted_sea_nonsea_compensationdata = pivoted_sea_nonsea_compensationdata.rename_axis(None, axis=1)

pivoted_sea_nonsea_compensationdata = pivoted_sea_nonsea_compensationdata.rename_axis(None, axis=0)

pivoted_sea_nonsea_compensationdata = pivoted_sea_nonsea_compensationdata.drop(['Total'], axis=1)

pivoted_sea_nonsea_compensationdata = pivoted_sea_nonsea_compensationdata[["$0-4,999", "$5,000-24,999", "$25,000-69,999", "$70,000-149,999", "> $150,000", "Null"]]

pivoted_sea_nonsea_compensationdata = pivoted_sea_nonsea_compensationdata.loc[['SEA', 'RoW'], :]



gridsize = (2, len(answers))

fig = plt.figure(figsize=(20, 7))

sns.set_style("whitegrid")

fig.suptitle('Annual Compensation: SEA-RoW', fontsize=14)



ax_frames = []

counter = 0

for ans in answers:

    ax = plt.subplot2grid(gridsize, (0, counter))

    counter += 1

    ax_frames.append(ax)



# barh barh barh

sns.set_style("whitegrid")

pivoted_sea_nonsea_compensationdata.plot(kind='barh', subplots=True, sharey=True, layout=(1,len(ax_frames)), legend=False, 

                                 xticks=[], yticks=[], ax=ax_frames,

                                 grid=False, xlim=(0, 50), edgecolor='none', fontsize=14,

                                 color = sns.light_palette(sns.color_palette("Blues_r")[0], len(ax_frames))[::-1]

                                )



sns.despine(left=False, top=True, right=True, bottom=True)



# labels!!!

for a in ax_frames:

    for p in a.patches:

        patch = a.annotate(str(p.get_width())+"%", 

                            (p.get_width(), 

                             p.get_y() + p.get_height()), 

                            ha = 'center', 

                            va = 'center', 

                            xytext = (20, -12),

                            textcoords = 'offset points'

                            )
### COMPARING THE COMPENSATION OF 2019 SURVEY PARTICIPANTS - PER SEA COUNTRY ###    



column_tosee = "new_compensation_bin"

short_column_tosee = "compensation"



# doing all of this to get a pivot for a barh subplot!!!

sea_nonsea_compensationdata = (sea_countries.groupby(['In which country do you currently reside?', column_tosee])["In which country do you currently reside?"].count()).reset_index(name="count")

sea_nonsea_compensationdata.columns = ["In which country do you currently reside?", short_column_tosee, "count"]

pivoted_sea_nonsea_compensationdata = (sea_nonsea_compensationdata.pivot(index='In which country do you currently reside?', columns=short_column_tosee, values='count')).reset_index()

pivoted_sea_nonsea_compensationdata.loc[:,'Total'] = pivoted_sea_nonsea_compensationdata.sum(axis=1)

answers = sea_countries[column_tosee].dropna().unique().tolist()

for a in answers:

    pivoted_sea_nonsea_compensationdata[a] = round((pivoted_sea_nonsea_compensationdata[a]/pivoted_sea_nonsea_compensationdata["Total"])*100, 2)

pivoted_sea_nonsea_compensationdata.reset_index(drop=True, inplace=True)

pivoted_sea_nonsea_compensationdata.set_index("In which country do you currently reside?", inplace=True)

pivoted_sea_nonsea_compensationdata = pivoted_sea_nonsea_compensationdata.rename_axis(None, axis=1)

pivoted_sea_nonsea_compensationdata = pivoted_sea_nonsea_compensationdata.rename_axis(None, axis=0)

pivoted_sea_nonsea_compensationdata = pivoted_sea_nonsea_compensationdata.drop(['Total'], axis=1)

pivoted_sea_nonsea_compensationdata = pivoted_sea_nonsea_compensationdata[["$0-4,999", "$5,000-24,999", "$25,000-69,999", "$70,000-149,999", "> $150,000", "Null"]]

pivoted_sea_nonsea_compensationdata = pivoted_sea_nonsea_compensationdata.loc[['Viet Nam', 'Indonesia', 'Philippines', 'Malaysia', 'Thailand', 'Singapore'], :]



gridsize = (2, len(answers))

fig = plt.figure(figsize=(20, 10))

sns.set_style("whitegrid")

fig.suptitle('Annual Compensation per SEA Countries', fontsize=14)





ax_frames = []

counter = 0

for ans in answers:

    ax = plt.subplot2grid(gridsize, (0, counter))

    counter += 1

    ax_frames.append(ax)



# barh barh barh

sns.set_style("whitegrid")

pivoted_sea_nonsea_compensationdata.plot(kind='barh', subplots=True, sharey=True, layout=(1,len(ax_frames)), legend=False, 

                                 xticks=[], yticks=[], ax=ax_frames,

                                 grid=False, xlim=(0, 50), edgecolor='none', fontsize=14,

                                 color = '#E3692A'

                                )



sns.despine(left=False, top=True, right=True, bottom=True)



# labels!!!

for a in ax_frames:

    for p in a.patches:

        patch = a.annotate(str(p.get_width())+"%", 

                            (p.get_width(), 

                             p.get_y() + p.get_height()), 

                            ha = 'center', 

                            va = 'center', 

                            xytext = (20, -12),

                            textcoords = 'offset points'

                            )
### COMPARING THE COMPENSATION OF 2019 SURVEY PARTICIPANTS - SEA (without singapore), ROW ###    



new_df = df[df['In which country do you currently reside?']!="Singapore"]



column_tosee = "new_compensation_bin"

short_column_tosee = "compensation"



# doing all of this to get a pivot for a barh subplot!!!

sea_nonsea_compensationWOsingdata = (new_df.groupby(['sea_tag', column_tosee])["sea_tag"].count()).reset_index(name="count")

sea_nonsea_compensationWOsingdata.columns = ["sea_tag", short_column_tosee, "count"]

pivoted_sea_nonsea_compensationWOsingdata = (sea_nonsea_compensationWOsingdata.pivot(index='sea_tag', columns=short_column_tosee, values='count')).reset_index()

pivoted_sea_nonsea_compensationWOsingdata.loc[:,'Total'] = pivoted_sea_nonsea_compensationWOsingdata.sum(axis=1)

answers = new_df[column_tosee].dropna().unique().tolist()

for a in answers:

    pivoted_sea_nonsea_compensationWOsingdata[a] = round((pivoted_sea_nonsea_compensationWOsingdata[a]/pivoted_sea_nonsea_compensationWOsingdata["Total"])*100, 2)

pivoted_sea_nonsea_compensationWOsingdata.reset_index(drop=True, inplace=True)

pivoted_sea_nonsea_compensationWOsingdata.set_index("sea_tag", inplace=True)

pivoted_sea_nonsea_compensationWOsingdata = pivoted_sea_nonsea_compensationWOsingdata.rename_axis(None, axis=1)

pivoted_sea_nonsea_compensationWOsingdata = pivoted_sea_nonsea_compensationWOsingdata.rename_axis(None, axis=0)

pivoted_sea_nonsea_compensationWOsingdata = pivoted_sea_nonsea_compensationWOsingdata.drop(['Total'], axis=1)

pivoted_sea_nonsea_compensationWOsingdata = pivoted_sea_nonsea_compensationWOsingdata[["$0-4,999", "$5,000-24,999", "$25,000-69,999", "$70,000-149,999", "> $150,000", "Null"]]

pivoted_sea_nonsea_compensationWOsingdata = pivoted_sea_nonsea_compensationWOsingdata.loc[['SEA', 'RoW'], :]

pivoted_sea_nonsea_compensationWOsingdata.index = ['SEA - \nw/o Singapore', 'RoW']



gridsize = (2, len(answers))

fig = plt.figure(figsize=(20, 7))

sns.set_style("whitegrid")

fig.suptitle('Annual Compensation: SEA(w/o Singapore)-RoW', fontsize=14)



ax_frames = []

counter = 0

for ans in answers:

    ax = plt.subplot2grid(gridsize, (0, counter))

    counter += 1

    ax_frames.append(ax)



# barh barh barh

sns.set_style("whitegrid")

pivoted_sea_nonsea_compensationWOsingdata.plot(kind='barh', subplots=True, sharey=True, layout=(1,len(ax_frames)), legend=False, 

                                 xticks=[], yticks=[], ax=ax_frames,

                                 grid=False, xlim=(0, 50), edgecolor='none', fontsize=14,

                                 color = sns.light_palette(sns.color_palette("Blues_r")[0], len(ax_frames))[::-1]

                                )



sns.despine(left=False, top=True, right=True, bottom=True)



# labels!!!

for a in ax_frames:

    for p in a.patches:

        patch = a.annotate(str(p.get_width())+"%", 

                            (p.get_width(), 

                             p.get_y() + p.get_height()), 

                            ha = 'center', 

                            va = 'center', 

                            xytext = (20, -12),

                            textcoords = 'offset points'

                            )
### COMPARING THE NUMBER OF 2019 SURVEY PARTICIPANTS - per ML YEARS ###    



column_tosee = 'For how many years have you used machine learning methods?'

shorter_column_tosee = 'years_used_ml_methods'



sea_nonsea_mlyearsdata = (sea_countries.groupby([column_tosee])["sea_tag"].count()).reset_index(name="count")

sea_nonsea_mlyearsdata.columns = [shorter_column_tosee, "count"]

sea_nonsea_mlyearsdata.sort_values(['count'], ascending=False, inplace=True)

sea_nonsea_mlyearsdata.reset_index(drop=True, inplace=True)

length_answer = len(df[column_tosee].dropna().unique().tolist())



gridsize = (length_answer, 9)

fig = plt.figure(figsize=(30, 8))

sns.set_style("whitegrid")

fig.suptitle('Years using Machine Learning Methods: SEA-RoW', fontsize=14)



ax2 = plt.subplot2grid(gridsize, (0, 4), colspan=3, rowspan=length_answer)

ax2 = sns.barplot(x='count', y=shorter_column_tosee, data=sea_nonsea_mlyearsdata, 

                  color = "#529FCD"

#                   palette=sns.light_palette(sns.color_palette("Blues_r")[0], length_answer)[::-1]

                 )

ax2_title = ax2.set_title('S.E.A. Countries')

new_yticks = [ax.get_text().replace("(", "\n(") for ax in ax2.get_yticklabels()]

ax2_new_yticks = ax2.set_yticklabels(new_yticks, {"horizontalalignment":"center", "x":"-0.2"})

ax2_ylabel = ax2.set_ylabel("")

ax2_xlabel = ax2.set_xlabel("")

ax2.grid(False)

for p in ax2.patches:

    patch = ax2.annotate(str(int(p.get_width())), 

                        (p.get_width(), 

                         p.get_y() + p.get_height()), 

                        ha = 'center', 

                        va = 'center', 

                        xytext = (15, 25),

                        textcoords = 'offset points'

                        )

sns.despine(ax=ax2, left=True, top=True, right=True,bottom=False)



### COMPARING THE ML METHODS YEARS OF 2019 SURVEY PARTICIPANTS - SEA, RoW ###    



grouped_all = df.groupby(["sea_tag", column_tosee])["sea_tag"].count().reset_index(name="count")

grouped_all.columns = ["sea_tag", shorter_column_tosee, "count"]

len_sea = grouped_all.groupby("sea_tag")["count"].sum()["SEA"]

len_nonsea = grouped_all.groupby("sea_tag")["count"].sum()["RoW"]

grouped_all["count_percent"] = grouped_all.apply(lambda row: round((row["count"]/len_sea)*100,2) if row["sea_tag"]=="SEA" else round((row["count"]/len_nonsea)*100,2), axis=1)



yticks_list = [ax.get_text() for ax in ax2.get_yticklabels()]

position = list(range(0,len(yticks_list)))



colors = ["#A8A495", "#E3692A"]



for po, b in zip(position, yticks_list):

    a = plt.subplot2grid(gridsize, (po, 1), colspan=2)

    a = sns.barplot(x = "sea_tag", 

                    y= "count_percent",

                    data=grouped_all[grouped_all[shorter_column_tosee]==b],

                    palette=sns.set_palette(sns.color_palette(colors))                    

                       )



    a.get_yaxis().set_visible(False)

    a.get_xaxis().set_visible(False)

    if po==len(yticks_list)-1:

        a.get_xaxis().set_visible(True)

        a_xlabel = a.set_xlabel("")

    for p in a.patches:

        patch = a.annotate(str(p.get_height())+"%", 

                            (p.get_x() + p.get_width() / 2.0, 

                             p.get_height()), 

                            ha = 'center', 

                            va = 'center', 

                            xytext = (0, 5),

                            textcoords = 'offset points')

    a_ylim = a.set_ylim(0, 60)

    sns.despine(ax=a, left=True, top=True, right=True,bottom=False)
### COMPARING THE NUMBER OF 2019 SURVEY PARTICIPANTS - per PRIMARY TOOLS ###    



column_tosee = 'What is the primary tool that you use at work or school to analyze data? (Include text response) - Selected Choice'

shorter_column_tosee = 'tool_analyze_data'



sea_nonsea_ptooldata = (sea_countries.groupby([column_tosee])["sea_tag"].count()).reset_index(name="count")

sea_nonsea_ptooldata.columns = [shorter_column_tosee, "count"]

sea_nonsea_ptooldata.sort_values(['count'], ascending=False, inplace=True)

sea_nonsea_ptooldata.reset_index(drop=True, inplace=True)

length_answer = len(df[column_tosee].dropna().unique().tolist())



gridsize = (length_answer, 9)

fig = plt.figure(figsize=(30, 8))

sns.set_style("whitegrid")

fig.suptitle('Primary Tool Used at Work/School to Analyze Data: SEA-RoW', fontsize=14)



ax2 = plt.subplot2grid(gridsize, (0, 4), colspan=3, rowspan=length_answer)

ax2 = sns.barplot(x='count', y=shorter_column_tosee, data=sea_nonsea_ptooldata, 

                  color="#529FCD"

#                   palette=sns.color_palette("Blues_r")

                 )

ax2_title = ax2.set_title('S.E.A. Countries')

new_yticks = [ax.get_text().replace("(", "\n(") for ax in ax2.get_yticklabels()]

ax2_new_yticks = ax2.set_yticklabels(new_yticks, {"horizontalalignment":"center", "x":"-0.2"})

ax2_ylabel = ax2.set_ylabel("")

ax2_xlabel = ax2.set_xlabel("")

ax2.grid(False)

for p in ax2.patches:

    patch = ax2.annotate(str(int(p.get_width())), 

                        (p.get_width(), 

                         p.get_y() + p.get_height()), 

                        ha = 'center', 

                        va = 'center', 

                        xytext = (15, 25),

                        textcoords = 'offset points'

                        )

sns.despine(ax=ax2, left=True, top=True, right=True,bottom=False)



### COMPARING THE PRIMARY TOOLS OF 2019 SURVEY PARTICIPANTS - SEA, RoW ###    



grouped_all = df.groupby(["sea_tag", column_tosee])["sea_tag"].count().reset_index(name="count")

grouped_all.columns = ["sea_tag", shorter_column_tosee, "count"]

len_sea = grouped_all.groupby("sea_tag")["count"].sum()["SEA"]

len_nonsea = grouped_all.groupby("sea_tag")["count"].sum()["RoW"]

grouped_all["count_percent"] = grouped_all.apply(lambda row: round((row["count"]/len_sea)*100,2) if row["sea_tag"]=="SEA" else round((row["count"]/len_nonsea)*100,2), axis=1)



yticks_list = [ax.get_text().replace("\n(", "(") for ax in ax2.get_yticklabels()]

position = list(range(0,len(yticks_list)))



colors = ["#A8A495", "#E3692A"]



for po, b in zip(position, yticks_list):

    a = plt.subplot2grid(gridsize, (po, 1), colspan=2)

    a = sns.barplot(x = "sea_tag", 

                    y= "count_percent",

                    data=grouped_all[grouped_all[shorter_column_tosee]==b],

                    palette=sns.set_palette(sns.color_palette(colors))

                       )



    a.get_yaxis().set_visible(False)

    a.get_xaxis().set_visible(False)

    if po==5:

        a.get_xaxis().set_visible(True)

        a_xlabel = a.set_xlabel("")

    for p in a.patches:

        patch = a.annotate(str(p.get_height())+"%", 

                            (p.get_x() + p.get_width() / 2.0, 

                             p.get_height()), 

                            ha = 'center', 

                            va = 'center', 

                            xytext = (0, 5),

                            textcoords = 'offset points')

    a_ylim = a.set_ylim(0, 60)

    sns.despine(ax=a, left=True, top=True, right=True,bottom=False)
### COMPARING THE NUMBER OF 2019 SURVEY PARTICIPANTS - per DS SOURCES for SEA ###    



ds_sources_columns = [

    'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Udacity',

    'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Coursera',

    'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - edX',

    'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - DataCamp',

    'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - DataQuest',

    'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Kaggle Courses (i.e. Kaggle Learn)',

    'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Fast.ai',

    'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Udemy',

    'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - LinkedIn Learning',

    'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - University Courses (resulting in a university degree)',

    'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - None',

    'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Other',

]



sea_dssources = sea_countries[ds_sources_columns]

count_allsea = len(sea_dssources)

sea_dssources_all = pd.DataFrame()

for msc in ds_sources_columns:

    dssources_ = ((sea_dssources[[msc]]).dropna().groupby(msc)[msc].count()).reset_index(name="count")

    dssources_.columns = ["ds_sources", "count"]

    dssources_["count_percent"] = round((dssources_["count"]/count_allsea)*100, 2)

    sea_dssources_all = sea_dssources_all.append(dssources_)

sea_dssources_all["sea_tag"] = "SEA"

sea_dssources_all



nonsea_dssources = nonsea_countries[ds_sources_columns]

count_allnonsea = len(nonsea_dssources)

nonsea_dssources_all = pd.DataFrame()

for msc in ds_sources_columns:

    dssources_ = ((nonsea_dssources[[msc]]).dropna().groupby(msc)[msc].count()).reset_index(name="count")

    dssources_.columns = ["ds_sources", "count"]

    dssources_["count_percent"] = round((dssources_["count"]/count_allnonsea)*100, 2)

    nonsea_dssources_all = nonsea_dssources_all.append(dssources_)

nonsea_dssources_all["sea_tag"] = "RoW"



allcountries_dssources_all = pd.concat([sea_dssources_all, nonsea_dssources_all])



column_tosee = 'ds_sources'

shorter_column_tosee = 'ds_sources'



sea_nonsea_tooldata = (sea_dssources_all.groupby([column_tosee])["count"].sum()).reset_index(name="count")

sea_nonsea_tooldata.columns = [shorter_column_tosee, "count"]

sea_nonsea_tooldata.sort_values(['count'], ascending=False, inplace=True)

sea_nonsea_tooldata.reset_index(drop=True, inplace=True)

length_answer = len(sea_dssources_all[column_tosee].dropna().unique().tolist())



gridsize = (length_answer, 9)

fig = plt.figure(figsize=(30, 10))

sns.set_style("whitegrid")

fig.suptitle('Data Science Courses: SEA-RoW', fontsize=14)



ax2 = plt.subplot2grid(gridsize, (0, 4), colspan=3, rowspan=length_answer)

ax2 = sns.barplot(x='count', y=shorter_column_tosee, data=sea_nonsea_tooldata, 

                  color = "#529FCD"

#                   palette=sns.light_palette(sns.color_palette("Blues_r")[0], length_answer)[::-1]

                 )

ax2_title = ax2.set_title('S.E.A. Countries')

new_yticks = [ax.get_text().replace("(", "\n(") for ax in ax2.get_yticklabels()]

ax2_new_yticks = ax2.set_yticklabels(new_yticks, {"horizontalalignment":"center", "x":"-0.2"})

ax2_ylabel = ax2.set_ylabel("")

ax2_xlabel = ax2.set_xlabel("")

ax2.grid(False)

for p in ax2.patches:

    patch = ax2.annotate(str(int(p.get_width())), 

                        (p.get_width(), 

                         p.get_y() + p.get_height()), 

                        ha = 'center', 

                        va = 'center', 

                        xytext = (15, 18),

                        textcoords = 'offset points'

                        )

sns.despine(ax=ax2, left=True, top=True, right=True,bottom=False)



### COMPARING THE NUMBER OF 2019 SURVEY PARTICIPANTS - per DS SOURCES for SEA, RoW ###    



grouped_all = allcountries_dssources_all



yticks_list = [ax.get_text().replace("\n(", "(") for ax in ax2.get_yticklabels()]

position = list(range(0,len(yticks_list)))



palette=sns.set_palette(sns.color_palette(colors))



colors = ["#A8A495", "#E3692A"]



for po, b in zip(position, yticks_list):

    a = plt.subplot2grid(gridsize, (po, 1), colspan=2)

    a = sns.barplot(x = "sea_tag", 

                    y= "count_percent",

                    data=grouped_all[grouped_all[shorter_column_tosee]==b],

                    palette=sns.set_palette(sns.color_palette(colors))

                       )



    a.get_yaxis().set_visible(False)

    a.get_xaxis().set_visible(False)

    if po==len(yticks_list)-1:

        a.get_xaxis().set_visible(True)

        a_xlabel = a.set_xlabel("")

    for p in a.patches:

        patch = a.annotate(str(p.get_height())+"%", 

                            (p.get_x() + p.get_width() / 2.0, 

                             p.get_height()), 

                            ha = 'center', 

                            va = 'center', 

                            xytext = (0, 5),

                            textcoords = 'offset points')

    a_ylim = a.set_ylim(0, 60)

    sns.despine(ax=a, left=True, top=True, right=True,bottom=False)
### COMPARING THE NUMBER OF 2019 SURVEY PARTICIPANTS - per MEDIA SOURCES for SEA, RoW ###    



media_sources_columns = [

    'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Twitter (data science influencers)',

    'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Hacker News (https://news.ycombinator.com/)',

    'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Reddit (r/machinelearning, r/datascience, etc)',

    'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Kaggle (forums, blog, social media, etc)',

    'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Course Forums (forums.fast.ai, etc)',

    'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - YouTube (Cloud AI Adventures, Siraj Raval, etc)',

    'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Podcasts (Chai Time Data Science, Linear Digressions, etc)',

    'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Blogs (Towards Data Science, Medium, Analytics Vidhya, KDnuggets etc)',

    'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Journal Publications (traditional publications, preprint journals, etc)',

    'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Slack Communities (ods.ai, kagglenoobs, etc)',

    'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - None',

    'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Other',

]



sea_mediasources = sea_countries[media_sources_columns]

count_allsea = len(sea_mediasources)

sea_mediasources_all = pd.DataFrame()

for msc in media_sources_columns:

    mediasources_ = ((sea_mediasources[[msc]]).dropna().groupby(msc)[msc].count()).reset_index(name="count")

    mediasources_.columns = ["media_sources", "count"]

    mediasources_["count_percent"] = round((mediasources_["count"]/count_allsea)*100, 2)

    sea_mediasources_all = sea_mediasources_all.append(mediasources_)

sea_mediasources_all["sea_tag"] = "SEA"

sea_mediasources_all



nonsea_mediasources = nonsea_countries[media_sources_columns]

count_allnonsea = len(nonsea_mediasources)

nonsea_mediasources_all = pd.DataFrame()

for msc in media_sources_columns:

    mediasources_ = ((nonsea_mediasources[[msc]]).dropna().groupby(msc)[msc].count()).reset_index(name="count")

    mediasources_.columns = ["media_sources", "count"]

    mediasources_["count_percent"] = round((mediasources_["count"]/count_allnonsea)*100, 2)

    nonsea_mediasources_all = nonsea_mediasources_all.append(mediasources_)

nonsea_mediasources_all["sea_tag"] = "NON-SEA"



allcountries_mediasources_all = pd.concat([sea_mediasources_all, nonsea_mediasources_all])



column_tosee = 'media_sources'

shorter_column_tosee = 'media_sources'



sea_nonsea_tooldata = (sea_mediasources_all.groupby([column_tosee])["count"].sum()).reset_index(name="count")

sea_nonsea_tooldata.columns = [shorter_column_tosee, "count"]

sea_nonsea_tooldata.sort_values(['count'], ascending=False, inplace=True)

sea_nonsea_tooldata.reset_index(drop=True, inplace=True)



length_answer = len(sea_mediasources_all[column_tosee].dropna().unique().tolist())



gridsize = (length_answer, 9)

fig = plt.figure(figsize=(30, 10))

sns.set_style("whitegrid")

fig.suptitle('Media Sources: SEA-RoW', fontsize=14)



ax2 = plt.subplot2grid(gridsize, (0, 4), colspan=3, rowspan=length_answer)

ax2 = sns.barplot(x='count', y=shorter_column_tosee, data=sea_nonsea_tooldata,

                  color = "#529FCD"

#                   palette=sns.light_palette(sns.color_palette("Blues_r")[0], length_answer)[::-1]

                 )

ax2_title = ax2.set_title('S.E.A. Countries')

new_yticks = [ax.get_text().replace("(", "\n(").replace("Time Data Science,", "Time Data Science,\n").replace("Medium,", "Medium,\n").replace("publications,", "publications,\n") for ax in ax2.get_yticklabels()]

ax2_new_yticks = ax2.set_yticklabels(new_yticks, {"horizontalalignment":"center", "x":"-0.2"})

ax2_ylabel = ax2.set_ylabel("")

ax2_xlabel = ax2.set_xlabel("")

ax2.grid(False)

for p in ax2.patches:

    patch = ax2.annotate(str(int(p.get_width())), 

                        (p.get_width(), 

                         p.get_y() + p.get_height()), 

                        ha = 'center', 

                        va = 'center', 

                        xytext = (15, 17),

                        textcoords = 'offset points'

                        )

sns.despine(ax=ax2, left=True, top=True, right=True,bottom=False)



### COMPARING THE NUMBER OF 2019 SURVEY PARTICIPANTS - per MEDIA SOURCES for SEA, RoW ###    



grouped_all = allcountries_mediasources_all



yticks_list = [ax.get_text().replace("\n", "") for ax in ax2.get_yticklabels()]

position = list(range(0,len(yticks_list)))



colors = ["#A8A495", "#E3692A"]



for po, b in zip(position, yticks_list):

    a = plt.subplot2grid(gridsize, (po, 1), colspan=2)

    a = sns.barplot(x = "sea_tag", 

                    y= "count_percent",

                    data=grouped_all[grouped_all[shorter_column_tosee]==b],

                    palette=sns.set_palette(sns.color_palette(colors))

                       )



    a.get_yaxis().set_visible(False)

    a.get_xaxis().set_visible(False)

    if po==len(yticks_list)-1:

        a.get_xaxis().set_visible(True)

        a_xlabel = a.set_xlabel("")

    for p in a.patches:

        patch = a.annotate(str(p.get_height())+"%", 

                            (p.get_x() + p.get_width() / 2.0, 

                             p.get_height()), 

                            ha = 'center', 

                            va = 'center', 

                            xytext = (0, 5),

                            textcoords = 'offset points')

    a_ylim = a.set_ylim(0, 70)

    sns.despine(ax=a, left=True, top=True, right=True,bottom=False)