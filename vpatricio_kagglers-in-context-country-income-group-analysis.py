import matplotlib.pylab as plt
import seaborn as sns
sns.set_style('darkgrid')

import numpy as np 
import pandas as pd 

import warnings
warnings.filterwarnings('ignore')
kaggle_data_raw = pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv')
kaggle_data_raw = kaggle_data_raw[['Q3','Q1','Q2','Q4','Q5','Q7','Q8','Q9']]

kaggle_data = (kaggle_data_raw
               .replace({'United States of America': 'United States'})
               .replace({'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom'})
               .replace({'Hong Kong (S.A.R.)': 'Hong Kong'})
               .replace({'Iran, Islamic Republic of...': 'Iran'})
               .replace({'Viet Nam': 'Vietnam'})
               .replace({'Republic of Korea':'South Korea'}) # TO BE CHECKED!
                )

country = (kaggle_data.Q3
           .value_counts()
           .drop('I do not wish to disclose my location')
           .drop('In which country do you currently reside?')
           .drop('Other')
            )
fig, ax = plt.subplots(1,1,figsize=(20,5))
country.sort_values(inplace=True,ascending=False)
p = sns.barplot(x=country.index,y=country, dodge=False, ax = ax)
dummy = p.set_xticklabels(country.index,rotation=90)
dummy = p.set_xlabel('')
dummy = p.set_ylabel('Kagglers')
wdi_data_raw = pd.read_csv('../input/world-development-indicators/Indicators.csv')

#Make a pre-selection of stuff that interest us
wdi_data_raw.drop_duplicates(subset=['CountryName','IndicatorCode'], keep='last',inplace=True,)
wdi_data_raw = wdi_data_raw[['CountryName','IndicatorCode','Value']]

# Fix the country names to match the ones from the kaggle survey
wdi_data_renamed = (wdi_data_raw
                    .replace({'Russian Federation' : 'Russia'})
                    .replace({'Iran, Islamic Rep.' : 'Iran'})
                    .replace({'Korea, Rep.':'South Korea'})
                    .replace({'Egypt, Arab Rep.' : 'Egypt'})
                    .replace({'Hong Kong SAR, China' : 'Hong Kong'})
                    .set_index('CountryName')
                   )
wdi_data_raw = wdi_data_raw[['CountryName','IndicatorCode','Value']]

# Check if all the data we want  is there
country_names_in_wdi = wdi_data_renamed.index.unique()
for name in country.keys():
    if name not in country_names_in_wdi:
        print(name,'not found')
        
# Make a smaller sub-set with only the countries we are interested in and check if it is what we want
wdi_data = wdi_data_renamed.loc[country.keys()]
wdi_country_info = pd.read_csv('../input/world-development-indicators/Country.csv')

#Make a pre-selection of stuff that interest us
wdi_country_info = wdi_country_info[['ShortName','Region','IncomeGroup']]

# Fix the country names to match the ones from the kaggle survey
wdi_country_info = (wdi_country_info
                    .replace({'Korea':'South Korea'})
                    .replace({'Hong Kong SAR, China' : 'Hong Kong'})
                    .set_index('ShortName')
                   )

# Check if all the data we want  is there
country_names_in_wdi = wdi_country_info.index.unique()
for name in country.keys():
    if name not in country_names_in_wdi:
        print(name,'not found')
        
wdi_country_info = wdi_country_info.loc[country.keys()]
wdi_pop_totl = wdi_data[wdi_data.IndicatorCode=='SP.POP.TOTL'].drop('IndicatorCode',axis=1)
wdi_urb_ptc = wdi_data[wdi_data.IndicatorCode=='SP.URB.TOTL.IN.ZS'].drop('IndicatorCode',axis=1)
wdi_kids_lit  = wdi_data[wdi_data.IndicatorCode=='SE.ADT.1524.LT.ZS'].drop('IndicatorCode',axis=1)
wdi_bigs_lit  = wdi_data[wdi_data.IndicatorCode=='SE.ADT.LITR.ZS'].drop('IndicatorCode',axis=1)
wdi_gini      = wdi_data[wdi_data.IndicatorCode=='SI.POV.GINI'].drop('IndicatorCode',axis=1)
wdi_wom_lit   = wdi_data[wdi_data.IndicatorCode=='SE.ADT.LITR.FE.ZS'].drop('IndicatorCode',axis=1)
wdi_man_lit   = wdi_data[wdi_data.IndicatorCode=='SE.ADT.LITR.MA.ZS'].drop('IndicatorCode',axis=1)
wdi_internet = wdi_data[wdi_data.IndicatorCode=='IT.NET.USER.P2'].drop('IndicatorCode',axis=1)
wdi_income = wdi_data[wdi_data.IndicatorCode=='NY.ADJ.NNTY.PC.CD'].drop('IndicatorCode',axis=1)
wdi_gdp = wdi_data[wdi_data.IndicatorCode=='NY.GDP.PCAP.CD'].drop('IndicatorCode',axis=1)
wdi_ter_labor = wdi_data[wdi_data.IndicatorCode=='SL.TLF.TERT.ZS'].drop('IndicatorCode',axis=1)
wdi_wom_sec_edu = wdi_data[wdi_data.IndicatorCode=='SE.SEC.ENRL.GC.FE.ZS'].drop('IndicatorCode',axis=1)
wdi_wom_industry = wdi_data[wdi_data.IndicatorCode=='SL.IND.EMPL.FE.ZS'].drop('IndicatorCode',axis=1)
wdi_wom_man_labor = wdi_data[wdi_data.IndicatorCode=='SL.TLF.CACT.FM.NE.ZS'].drop('IndicatorCode',axis=1)
wdi_wom_labor_ter_edu = wdi_data[wdi_data.IndicatorCode=='SL.TLF.TERT.FE.ZS'].drop('IndicatorCode',axis=1)
wdi_young_pop = wdi_data[wdi_data.IndicatorCode=='SP.POP.0014.TO.ZS'].drop('IndicatorCode',axis=1)
wdi_middle_pop = wdi_data[wdi_data.IndicatorCode=='SP.POP.1564.TO.ZS'].drop('IndicatorCode',axis=1)
wdi_older_pop = wdi_data[wdi_data.IndicatorCode=='SP.POP.65UP.TO.ZS'].drop('IndicatorCode',axis=1)

pop = pd.concat([country,wdi_pop_totl/1e6,wdi_urb_ptc,wdi_kids_lit,wdi_bigs_lit,wdi_wom_lit/wdi_man_lit,wdi_gini,
                 wdi_internet,wdi_income,wdi_gdp,wdi_ter_labor,
                 wdi_wom_sec_edu,wdi_wom_industry,wdi_wom_man_labor,wdi_wom_labor_ter_edu,  
                 wdi_young_pop,wdi_middle_pop,wdi_older_pop,
                 wdi_income,
                 wdi_country_info.Region,wdi_country_info.IncomeGroup],
                axis=1)
pop.columns =['Kagglers','Pop','UrbanPop','YoungLit','AdultLit','FemMaleLit','Gini','InternetUsers','AdjustedIncome','GDPCapita','LaborTertiaryEducation',
              'WomanSecEducation','WomanIndustry','RatioWMLabor','WomanLaborAndTerEdu',
              'LessThen14','Between','MoreThen65',
              'IncomePerCapita',
              'Region','IncomeGroup']

pop = (pop
        .replace({'High income: OECD': 'High\n(OECD)'})
        .replace({'High income: nonOECD': 'High\n(nonOECD)'})
        .replace({'Lower middle income': 'Lower\nMiddle'})
        .replace({'Upper middle income': 'Upper\nMiddle'})
        )       

# Transform stuff
pop['CountryName'] = pop.index
pop['KagglersCapita'] = pop.apply(lambda row: row.Kagglers / row.Pop , axis=1)
c = ['#F3715A','#FFAD59','#F8FC98','#C5E17A','#00BCD4','#03A9F4','#AA55AA']
c2 = [c[3],c[5],c[0],c[1],c[4],c[6],c[2]]

fig, ax = plt.subplots(2,1,figsize=(20,10))
fig.subplots_adjust(hspace=0.6)

pop.sort_values('Kagglers',inplace=True,ascending=False)
p = sns.barplot(x='CountryName',y='Kagglers',hue='Region',data=pop, dodge=False, palette=c2, ax = ax[0])
dummy = p.set_xticklabels(pop.index,rotation=90)
dummy = p.set_xlabel('')

pop.sort_values('KagglersCapita',inplace=True,ascending=False)
p = sns.barplot(x='CountryName',y='KagglersCapita',hue='Region',data=pop, dodge=False, palette=c, ax = ax[1])
dummy = p.set_xticklabels(pop.index,rotation=90)
dummy = p.set_xlabel('')
incomegroups = (pop.groupby('IncomeGroup')
                .mean()
                .sort_values('AdjustedIncome',ascending=False))

fig,ax = plt.subplots(1,3,figsize=(15,5))
fig.subplots_adjust(right=0.99,hspace=0.4,bottom=0.2)
ax = ax.ravel()

features =['Pop','Kagglers','KagglersCapita']
for x,var in enumerate(features):
    p = sns.barplot(x=incomegroups.index,y=var,data=incomegroups,ax=ax[x],palette='Oranges_r')
    dummy = p.set_xticklabels(incomegroups.index,rotation=60)
    dummy = p.set_xlabel('')
order = ['High\n(OECD)','High\n(nonOECD)', 'Upper\nMiddle', 'Lower\nMiddle']
def plot_and_scatter(ind,i):
    if i == 0:
        p = sns.scatterplot(y='KagglersCapita',x=ind,hue='IncomeGroup',hue_order=order,data=pop,ax=ax[i],palette='Oranges_r')
        p.legend(loc=2)
    else:
        sns.scatterplot(y='KagglersCapita',x=ind,hue='IncomeGroup',hue_order=order,data=pop,ax=ax[i],legend=False,palette='Oranges_r')
    ax[i].annotate('$\\rho$ = %0.4f'%pop['KagglersCapita'].corr(pop[ind],'spearman'),xy=(0.7,0.85),xycoords='axes fraction')
    ax[i].annotate('r = %0.4f'%pop['KagglersCapita'].corr(pop[ind]),xy=(0.7,0.8),xycoords='axes fraction')

fig,ax = plt.subplots(2,4,figsize=(20,10))
fig.subplots_adjust(right=0.99)
ax = ax.ravel()

features = ['GDPCapita','InternetUsers','AdjustedIncome','LaborTertiaryEducation','YoungLit','UrbanPop','AdultLit','Gini']

for x,var in enumerate(features):
    plot_and_scatter(var,x)
kaggle_by_incomegrp = kaggle_data.copy()

for i,k in pop['IncomeGroup'].to_dict().items():
    kaggle_by_incomegrp.replace(to_replace={i:k},inplace=True)
kaggle_by_incomegrp.replace({'I do not wish to disclose my location' : 'Not telling you'},inplace=True)
    
kaggle_by_incomegrp = kaggle_by_incomegrp[['Q3','Q1','Q2','Q4','Q8','Q9']]
kaggle_by_incomegrp.columns = ['IncomeGroup','Gender','Age','Education','Experience','Compensation']
kaggle_by_incomegrp.drop(index=0,inplace=True)
fix_index = ['High\n(OECD)', 'High\n(nonOECD)', 'Upper\nMiddle','Lower\nMiddle', 'Other','Not telling you']

#kaggle_by_incomegrp.describe()
def massage_data(feature):
    feat = kaggle_by_incomegrp.groupby(['IncomeGroup',feature]).size().reset_index()
    feat.columns = ['IncomeGroup',feature,'Counts']
    feat.set_index('IncomeGroup')

    # Change to percentage
    feat['Percentage'] = np.nan
    feature_nb = len(feat[feature].unique())
    for i,grp in enumerate(feat['IncomeGroup'].unique()):
        total = feat[feat['IncomeGroup'] == grp]['Counts'].sum()
        idx = np.where(feat['IncomeGroup'] == grp)[0]
        feat[idx[0]:idx[-1]+1]['Percentage'] = feat[idx[0]:idx[-1]+1]['Counts'] / total*100

    return feat
gender = massage_data('Gender')
# pivot from here: https://pstblog.com/2016/10/04/stacked-charts
pivot_gender = gender.pivot(index='IncomeGroup', columns='Gender', values='Percentage').loc[fix_index]
p = pivot_gender.plot.bar(stacked=True, figsize=(10,7),cmap='Accent')
dummy = p.set_ylabel('Percentage of total answers per income group')
dummy = p.set_xlabel('')
dummy = p.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig,ax = plt.subplots(1,5,figsize=(20,5))
fig.subplots_adjust(right=0.99,bottom=0.1)

p = pivot_gender['Female'].plot.bar(stacked=True,cmap='Accent',ax=ax[0])
dummy = p.set_ylabel('Percentage of total answers per income group')
dummy = p.set_xlabel('')

features =['WomanSecEducation','WomanIndustry','RatioWMLabor','WomanLaborAndTerEdu']
for x,var in enumerate(features):
    p = sns.barplot(x=incomegroups.index,y=var,data=incomegroups,ax=ax[x+1],palette='Oranges_r')
    dummy = p.set_xticklabels(incomegroups.index,rotation=60)
    dummy = p.set_xlabel('')

print('Variance:')
print('Kagglers %0.2f'%pivot_gender['Female'].drop('Other').drop('Not telling you').std())
print('Female in secondary education %0.2f'%incomegroups['WomanSecEducation'].std())
print('Female in Industry %0.2f'%incomegroups['WomanIndustry'].std())
print('Female in the labor force %0.2f'%incomegroups['RatioWMLabor'].std())
print('Female in labor force with terciary education %0.2f'%incomegroups['WomanLaborAndTerEdu'].std())
age = massage_data('Age')
pivot_age = age.pivot(index='IncomeGroup', columns='Age', values='Percentage').loc[fix_index]
p = pivot_age.plot.bar(stacked=True, figsize=(10,7),cmap='Blues_r',)
dummy = p.set_ylabel('Percentage of total answers per income group')
dummy = p.set_xlabel('')
dummy = p.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig,ax = plt.subplots(1,3,figsize=(20,5))
fig.subplots_adjust(right=0.99,bottom=0.1)

features =['LessThen14', 'Between','MoreThen65']
for x,var in enumerate(features):
    p = sns.barplot(x=incomegroups.index,y=var,data=incomegroups,ax=ax[x],palette='Oranges_r')
    dummy = p.set_xticklabels(incomegroups.index,rotation=60)
    dummy = p.set_xlabel('')
compensation = massage_data('Compensation')

# Ordering
pivot_compensation = compensation.pivot(index='IncomeGroup', columns='Compensation', values='Percentage').loc[fix_index]
pivot_compensation = pivot_compensation[['0-10,000', '10-20,000', '20-30,000', '30-40,000','40-50,000','50-60,000','60-70,000','70-80,000', '80-90,000', '90-100,000',
                                          '100-125,000', '125-150,000', '150-200,000', '200-250,000', '250-300,000',  '300-400,000','400-500,000',  '500,000+', 
                                         'I do not wish to disclose my approximate yearly compensation']] # yes, it's hugly. 

# Plotting
p = pivot_compensation.plot.barh(stacked=True, figsize=(15,7),cmap='viridis_r')
dummy = p.set_xlabel('Percentage of total answers per income group')
dummy = p.set_ylabel('')
dummy = p.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# If you're reading this STOP! Below there are four quite embarassing lines, but I didn't know how to display this information and I was too tired to figure it out
# As we say it back home: those who do not have a dog, hunt with a cat
d = p.annotate('__________',xy=(0.201,0.135),color='r',weight='bold',xycoords='axes fraction')
d = p.annotate('_______________________________',xy=(0.203,0.30),color='r',weight='bold',xycoords='axes fraction')
d = p.annotate('____________________________________________________',xy=(0.001,0.47),color='r',weight='bold',xycoords='axes fraction')
d = p.annotate('_______________________________________________________________',xy=(0.001,0.635),color='r',weight='bold',xycoords='axes fraction')
incomegroups.IncomePerCapita
education = massage_data('Education')

# Ordering 
pivot_education = education.pivot(index='IncomeGroup', columns='Education', values='Percentage').loc[fix_index]
pivot_education = pivot_education[['No formal education past high school','Some college/university study without earning a bachelor’s degree','Professional degree',
                                 'Bachelor’s degree','Master’s degree','Doctoral degree','I prefer not to answer']]

# Plotting
p = pivot_education.plot.bar(stacked=True, figsize=(10,7),cmap='PRGn')
dummy = p.set_ylabel('Percentage of total answers per income group')
dummy = p.set_xlabel('')
dummy = p.legend(loc='center left', bbox_to_anchor=(1, 0.5))

experience = massage_data('Experience')

# Ordering
pivot_experience = experience.pivot(index='IncomeGroup', columns='Experience', values='Percentage').loc[fix_index]
pivot_experience = pivot_experience[['0-1', '1-2','2-3', '3-4','4-5','5-10','10-15', '15-20', '20-25', '25-30',  '30 +']]

# Plotting
p = pivot_experience.plot.bar(stacked=True, figsize=(10,7),cmap='Oranges')
dummy = p.set_ylabel('Percentage of total answers per income group')
dummy = p.set_xlabel('')
dummy = p.legend(loc='center left', bbox_to_anchor=(1, 0.5))