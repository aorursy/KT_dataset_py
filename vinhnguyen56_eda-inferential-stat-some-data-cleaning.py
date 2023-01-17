#import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
train = pd.read_csv('../input/train.csv')
train.head()
train.columns.to_frame()
def data_clean(data):
    #fill in missing values
    data['rez_esc']=data['rez_esc'].fillna(0)
    data['v18q1'] = data['v18q1'].fillna(0)
    v2a1 = data['v2a1'].sort_values()
    med = v2a1.median()
    data.loc[(data['tipovivi1']==1), 'v2a1'] = 0
    data.loc[(data['tipovivi4']==1), 'v2a1'] = med
    data.loc[(data['tipovivi5']==1), 'v2a1'] = med
    meaneduc_nan=data[data['meaneduc'].isnull()][['Id','idhogar','escolari']]
    me=meaneduc_nan.groupby('idhogar')['escolari'].mean().reset_index()
    me
    for row in meaneduc_nan.iterrows():
        idx=row[0]
        idhogar=row[1]['idhogar']
        m=me[me['idhogar']==idhogar]['escolari'].tolist()[0]
        data.at[idx, 'meaneduc']=m
        data.at[idx, 'SQBmeaned']=m*m
        
    #binary columns
    housesitu = ['tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5']
    educlevels = ['instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7',
             'instlevel8', 'instlevel9']
    regions = ['lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6']
    relations = ['parentesco1', 'parentesco2', 'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6',
            'parentesco7', 'parentesco8', 'parentesco9', 'parentesco10', 'parentesco11', 'parentesco12']
    marital = ['estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7']
    rubbish = ['elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6']
    energy = ['energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4']
    toilets = ['sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6']
    floormat = ['pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera']
    wallmat = ['paredblolad', 'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc', 'paredfibras', 'paredother']
    roofmat = ['techozinc', 'techoentrepiso', 'techocane', 'techootro']
    floorqual = ['eviv1', 'eviv2', 'eviv3']
    wallqual = ['epared1', 'epared2', 'epared3']
    roofqual = ['etecho1', 'etecho2', 'etecho3']
    waterprov = ['abastaguadentro', 'abastaguafuera', 'abastaguano']
    electric = ['public', 'planpri', 'noelec', 'coopele']
    
    
    #make a dictionary
    binaries = {'housesitu':housesitu,
                'educlevels':educlevels,
                'regions':regions,
                'relations':relations,
                'marital':marital,
                'rubbish':rubbish,
                'energy':energy,
                'toilets':toilets,
                'floormat':floormat,
                'wallmat':wallmat,
                'roofmat':roofmat,
                'floorqual':floorqual,
                'wallqual':wallqual,
                'roofqual':roofqual,
                'waterprov':waterprov,
                'electric':electric
               }
    
    #Replacing the binaries with categorical
    for i in binaries.keys():
        data[i] = data[binaries[i]].idxmax(axis=1)
        data.drop(data[binaries[i]], axis=1, inplace=True)
    
    #recoding values
    hs = {'tipovivi1':'Own', 
      'tipovivi2':'Own/Paying Instllmnts', 
      'tipovivi3':'Rented', 
      'tipovivi4':'Precarious', 
      'tipovivi5':'Other'}
    el = {'instlevel1':'None', 
      'instlevel2':'Incomplete Primary', 
      'instlevel3':'Complete Primary', 
      'instlevel4':'Incomplete Acad. Secondary', 
      'instlevel5':'Complete Acad. Secondary', 
      'instlevel6':'Incomplete Techn. Secondary', 
      'instlevel7':'Complete Techn. Secondary',
      'instlevel8':'Undergrad.', 
      'instlevel9':'Postgrad.'}
    rgn = {'lugar1':'Central', 
       'lugar2':'Chorotega', 
       'lugar3':'Pacafafico Central', 
       'lugar4':'Brunca', 
       'lugar5':'Huetar Atlantica', 
       'lugar6':'Huetar Norte'}
    rltn = {'parentesco1':'Household Head', 
        'parentesco2':'Spouse/Partner', 
        'parentesco3':'Son/Daughter', 
        'parentesco4':'Stepson/Daughter', 
        'parentesco5':'Son/daughter in law', 
        'parentesco6':'Grandson/daughter',
        'parentesco7':'Mother/Father', 
        'parentesco8':'Mother/father in law', 
        'parentesco9':'Brother/sister', 
        'parentesco10':'Brother/sister in law', 
        'parentesco11':'Other family member', 
        'parentesco12':'Other non-family member'}
    mrtl = {'estadocivil1':'< 10 y/o', 
        'estadocivil2':'Free or coupled union', 
        'estadocivil3':'Married', 
        'estadocivil4':'Divorced', 
        'estadocivil5':'Separated', 
        'estadocivil6':'Widow/er', 
        'estadocivil7':'Single'}
    rb = {'elimbasu1':'Tanker Truck', 
      'elimbasu2':'Botan Hollow or Buried', 
      'elimbasu3':'Burning', 
      'elimbasu4':'Thrown in unoccupied space', 
      'elimbasu5':'Thrown in river, creek, or sea', 
      'elimbasu6':'Other'}
    eng = {'energcocinar1':'None', 
       'energcocinar2':'Electricity', 
       'energcocinar3':'Gas', 
       'energcocinar4':'Wood Charcoal'}
    tlt = {'sanitario1':'None', 
       'sanitario2':'Sewer or Cesspool', 
       'sanitario3':'Septic Tank', 
       'sanitario5':'Black hole or letrine', 
       'sanitario6':'Other'}
    flmt = {'pisomoscer':'Mosaic, Ceramic', 
        'pisocemento':'Cement', 
        'pisoother':'Other', 
        'pisonatur':'Natural', 
        'pisonotiene':'None', 
        'pisomadera':'Wood'}
    wlmt = {'paredblolad':'Block/Brick', 
        'paredzocalo':'Socket (wood, zinc, absbesto)', 
        'paredpreb':'Prefabricated/cement', 
        'pareddes':'Waste', 
        'paredmad':'Wood', 
        'paredzinc':'Zinc', 
        'paredfibras':'Natural Fibers', 
        'paredother':'Other'}
    rfmt = {'techozinc':'Metal foil/Zinc', 
        'techoentrepiso':'Fiber cement', 
        'techocane':'Natural fibers', 
        'techootro':'Other'}
    flql = {'eviv1':'Bad', 
        'eviv2':'Regular', 
        'eviv3':'Good'}
    wlql = {'epared1':'Bad',
        'epared2':'Regular', 
        'epared3':'Good'}
    rfqu = {'etecho1':'Bad', 
        'etecho2':'Regular', 
        'etecho3':'Good'}
    wtrpr = {'abastaguadentro':'Inside', 
         'abastaguafuera':'Outside', 
         'abastaguano':'None'}
    elct = {'public':'Public', 
        'planpri':'Private Plant', 
        'noelec':'None', 
        'coopele':'Cooperative'}
    
    #replacing
    data.replace(dict(housesitu=hs, 
                  educlevels=el,
                  regions=rgn,
                  relations=rltn,
                  marital=mrtl,
                  rubbish=rb,
                  energy=eng,
                  toilets=tlt,
                  floormat=flmt,
                  wallmat=wlmt,
                  roofmat=rfmt,
                  floorqual=flql,
                  wallqual=wlql,
                  roofqual=rfqu,
                  waterprov=wtrpr,
                  electric=elct), inplace=True)
train = pd.read_csv('../input/train.csv')
data_clean(train)
train.to_csv('trainclean.csv')
train.head()
train.columns
test = pd.read_csv('../input/test.csv')
data_clean(test)
test.to_csv('testclean.csv')
test.head()
corr = train.corr()
corr.style.background_gradient()
train[['r4h1','r4h2','r4h3','r4m1','r4m2','r4m3','r4t1','r4t2','r4t3']].describe()

sns.countplot('Target',data=train)
plt.xlabel('Poverty Level')
plt.ylabel('Frequency')
plt.title('Household Poverty Levels')
plt.show()
train.floorqual.value_counts()
sns.boxplot(x='Target', y='v2a1', data=train)
plt.xlabel('Poverty Level')
plt.ylabel('Monthly Rent Payment ($)')
plt.show()
train = train[train['v2a1'] < 400000]
trainrented = train[train['housesitu']=='Rented']
sns.boxplot(x='Target', y=trainrented['v2a1'], data=train)
plt.xlabel('Poverty Level')
plt.ylabel('Monthly Rent Payment ($)')
plt.show()
#Monthly rent summary for each poverty level
for i in train['Target'].unique():
    print(i)
    print(trainrented[(trainrented['Target'] == i)]['v2a1'].describe())
    print()

levels = [1,2,3,4]
rentmeans = []
for x in levels:
    mean = np.mean(trainrented[trainrented['Target']==x]['v2a1'])
    rentmeans.append(mean)

plt.plot(levels, rentmeans, marker='o')
plt.xlabel('Poverty Level')
plt.title('Mean Monthly Rent by Poverty Level')
plt.xticks(levels,rotation=30)
plt.ylabel('Mean Monthly Rent ($)')
#non vulnerable
meanV = np.mean(trainrented[trainrented['Target'] == 4]['v2a1'])
print('Non Vulnerable Mean Rent: ', meanV)

#vulnerable
meanV = np.mean(trainrented[trainrented['Target'] == 3]['v2a1'])
print('Vulnerable Mean Rent: ', meanV)

#moderate
meanM = np.mean(trainrented[trainrented['Target'] == 2]['v2a1'])
print('Moderate Mean Rent: ', meanM)

#extreme
meanE = np.mean(trainrented[trainrented['Target'] == 1]['v2a1'])
print('Extreme Mean Rent: ', meanE)

#total
meanTot = np.mean(trainrented['v2a1'])
print('Mean Rent of Total: ', meanTot)
#non vulnerable and vulnerable
from statsmodels.stats.weightstats import ztest
tstat, p = ztest(trainrented[trainrented['Target'] == 4]['v2a1'],
                           trainrented[trainrented['Target'] == 3]['v2a1'])
print('T Stat: ', tstat)
print('P-Value: ', p)
#vulnerable and moderate
tstat, p = ztest(trainrented[trainrented['Target'] == 3]['v2a1'],
                           trainrented[trainrented['Target'] == 2]['v2a1'])
print('T Stat: ', tstat)
print('P-Value: ', p)
#moderate and extreme
tstat, p = ztest(trainrented[trainrented['Target'] == 2]['v2a1'],
                           trainrented[trainrented['Target'] == 1]['v2a1'])
print('T Stat: ', tstat)
print('P-Value: ', p)
#proportion chart to compare normalized data among target levels for each feature.
def percent_table(x):
    return x/float(x[-1])

def prop_chart(column, title):
    df = pd.crosstab(train['Target'], train[column], margins=True).apply(percent_table, axis=1)
    df.iloc[:-1,:-1].plot(kind='bar')
    plt.legend(loc=0, fontsize='x-small')
    plt.title(title)
prop_chart('wallmat', 'Wall Materials')

prop_chart('floormat', 'Floor Materials')

prop_chart('roofmat', 'Roof Materials')
#quality
#create crosstab dataframes 
prop_chart('wallqual', 'Wall Quality')
prop_chart('floorqual', 'Floor Quality')
prop_chart('roofqual', 'Roof Quality')
#ztest proportion
from statsmodels.stats.proportion import proportions_ztest
import warnings
warnings.filterwarnings("ignore")

def propztest_poverty(data, column, val): 
    
    nonvuln = data[data.Target==4]
    vuln = data[data.Target==3]
    moder = data[data.Target==2]
    extreme = data[data.Target==1]
    
    n1 = len(extreme)
    n2 = len(moder)
    n3 = len(vuln)
    n4 = len(nonvuln)
    s1 = len(extreme[data[column]==val])
    s2 = len(moder[data[column]==val])
    s3 = len(vuln[data[column]==val])
    s4 = len(nonvuln[data[column]==val])
    
    #nonvuln and vuln
    z1, pval1 = proportions_ztest([s4, s3], [n4, n3])
    print('Nonvuln proportion:', s4/n4)
    print('Vuln proportion:', s3/n3)
    print('Non Vulnerable and Vulnerable: [zscore, P-Value]', 
          ['{:.12f}'.format(b) for b in (z1, pval1)])
    if pval1 < 0.05:
        print('Significant')
    else:
        print('Not significant')
    
    #vuln and moder
    z2, pval2 = proportions_ztest([s3, s2], [n3, n2])
    print('Vuln proportion:', s3/n3)
    print('Moderate proportion:', s2/n2)
    print('Vulnerable and Moderate: [zscore, P-Value]', 
          ['{:.12f}'.format(b) for b in (z2, pval2)])
    if pval2 < 0.05:
        print('Significant')
    else:
        print('Not significant')
        
    #moder and extreme
    z3, pval3 = proportions_ztest([s2, s1], [n2, n1])
    print('Moderate proportion', s2/n2)
    print('Extreme proportion', s1/n1)
    print('Moderate and Extreme: [zscore, P-Value]', 
          ['{:.12f}'.format(b) for b in (z3, pval3)])
    if pval3 < 0.05:
        print('Significant')
    else:
        print('Not significant')

#Floors
print('Floor Quality')
for x in train['floorqual'].unique():
    
    print(x)
    propztest_poverty(train, 'floorqual', x)
    print()
#Wall Quality
print('Wall Quality')
for x in train['wallqual'].unique():
    print(x)
    propztest_poverty(train, 'wallqual', x)
    print()
#Roof Quality
print('Roof Quality')
for x in train['roofqual'].unique():
    print(x)
    propztest_poverty(train, 'roofqual', x)
    print()
educdf = pd.crosstab(index=train['Target'], columns=train['educlevels'], margins=True).apply(percent_table,axis=1)
educdf
primary = educdf[['Complete Primary', 'Incomplete Primary']]
secondary = educdf[['Complete Acad. Secondary', 'Incomplete Acad. Secondary',
                   'Complete Techn. Secondary', 'Incomplete Techn. Secondary']]
none = educdf['None']
undergrad = educdf['Undergrad.']
postgrad = educdf['Postgrad.']

primary.iloc[:-1].plot(kind='bar')
plt.title('Primary School')
plt.show()

secondary.iloc[:-1].plot(kind='bar')
plt.title('Secondary School')
plt.legend(fontsize='x-small')
plt.show()

none.iloc[:-1].plot(kind='bar')
plt.title('No Schooling')
plt.show()

undergrad.iloc[:-1].plot(kind='bar')
plt.title('Undergrad')
plt.show()

postgrad.iloc[:-1].plot(kind='bar')
plt.title('Postgrad')
plt.show()
print('Education Levels')
for x in train.educlevels.unique():
    print(x)
    propztest_poverty(train, 'educlevels', x)
    print()
train.hacapo.describe()
overcrowdf = pd.crosstab(train['Target'], train['hacapo'], margins=True).apply(percent_table, axis=1)
overcrowdf.iloc[:-1, 1].plot(kind='bar', stacked=True)
plt.xticks(rotation=30)
plt.title('Proportion of Overcrowding per Poverty Level')
plt.ylabel('Proportion')
overcrowdf
#overcrowding by room 
train.hacapo.head()
#overcrowding by bedroom
train.hacdor.head()
overcrowdf = pd.crosstab(train['Target'], train['hacdor'], margins=True).apply(percent_table, axis=1)
overcrowdf.iloc[:-1, 1].plot(kind='bar', stacked=True)
plt.xticks(rotation=30)
plt.title('Proportion of Overcrowding by Bedroom per Poverty Level')
plt.ylabel('Proportion')
print('overcrowding by room')
propztest_poverty(train, 'hacapo', 1)
print()

print('overcrowding by bedroom')
propztest_poverty(train, 'hacdor', 1)
from scipy.stats import pearsonr
corr1 = pearsonr(train.hacapo, train.hacdor)
print('hacapo x hacdor: ', corr1)

corr2 = pearsonr(train.Target, train.hacapo)
print('Target x hacapo: ', corr2)

corr3 = pearsonr(train.Target, train.hacdor)
print('Target x hacdor: ', corr3)
#inside
waterprovdf = pd.crosstab(train['Target'], train['waterprov'], margins=True).apply(percent_table, axis=1)
waterprovdf.iloc[:-1, 0].plot(kind='bar', stacked=True)
plt.legend()
plt.ylabel('Proportion')
#none
waterprovdf.iloc[:-1, 1].plot(kind='bar', stacked=True)
plt.legend()
plt.ylabel('Proportion')
#outside
waterprovdf.iloc[:-1, 2].plot(kind='bar', stacked=True)
plt.legend()
plt.ylabel('Proportion')
waterprovdf
for x in train['waterprov'].unique():
    print(x)
    propztest_poverty(train, 'waterprov', x)
    print()
sns.countplot('regions', data=train)
plt.xticks(rotation=45)
train['regions'].value_counts()
regionsdf = pd.crosstab(train['regions'], train['Target'])
regionsdf
prop_chart('regions', 'Regions')
for x in train['regions'].unique():
    print(x)
    propztest_poverty(train, 'regions', x)
    print()
print(train['relations'].value_counts())
sns.countplot('relations', data=train)
plt.xticks(rotation=60)
prop_chart('relations', 'Relations')
for x in train['relations'].unique():
    print(x)
    propztest_poverty(train, 'relations', x)
    print()

print(train['toilets'].value_counts())
sns.countplot('toilets', data=train)
plt.xticks(rotation=45)
prop_chart('toilets','Toilet is Connected To')
for x in train['toilets'].unique():
    print(x)
    propztest_poverty(train, 'toilets', x)
    print()
print(train['housesitu'].value_counts())
sns.countplot('housesitu', data=train)
plt.xticks(rotation=45)
prop_chart('housesitu', 'Housing Situation')
for x in train['housesitu'].unique():
    print(x)
    propztest_poverty(train, 'housesitu', x)
    print()
print(train['energy'].value_counts())
sns.countplot('energy', data=train)
plt.xticks(rotation=45)
prop_chart('energy', 'Sources of Energy for Cooking')
for x in train['energy'].unique():
    print(x)
    propztest_poverty(train, 'energy', x)
    print()
appliances = train[['v18q', 'v18q1', 'computer', 'television', 'mobilephone', 'qmobilephone']]
appliances.sample(10)
appliances.describe()
for x in appliances.columns:
    print(appliances[x].value_counts())
    sns.countplot(x, data=train)
    plt.show()
for x in appliances.columns:
    prop_chart(x, x)
print('Tablets')
propztest_poverty(train, 'v18q', 1)
print()

print('Computer')
propztest_poverty(train, 'computer', 1)
print()

print('Television')
propztest_poverty(train, 'television', 1)
print()

print('Mobile Phones')
propztest_poverty(train, 'mobilephone', 1)
print()
# Number of Tablets
def ztestmean_poverty(data, column):
    
    nonvuln = train[train['Target']==4][column]
    vuln = train[train['Target']==3][column]
    moder = train[train['Target']==2][column]
    extreme = train[train['Target']==1][column]
    total = train[column]

    print('Nonvulnerable mean: ', np.mean(nonvuln))
    print('Vulnerable mean: ', np.mean(vuln))
    print('Moderate mean: ', np.mean(moder))
    print('Extreme mean: ', np.mean(extreme))
    print('Total Mean: ', np.mean(total))
    print()
    
    tstat, p= ztest(trainrented[trainrented['Target'] == 4][column],
                           trainrented[trainrented['Target'] == 3][column])
    print('Nonvulnerable and Vulnerable p-val: ', p)
    if p < 0.05: 
        print('Significant')
    else: 
        print('Non Significant')
        
    tstat, p= ztest(trainrented[trainrented['Target'] == 3][column],
                           trainrented[trainrented['Target'] == 2][column])
    print('Vulnerable and Moderate p-val: ', p)
    if p < 0.05: 
        print('Significant')
    else: 
        print('Non Significant')
    
    tstat, p= ztest(trainrented[trainrented['Target'] == 2][column],
                           trainrented[trainrented['Target'] == 1][column])
    print('Moderate and Extreme p-val: ', p)
    if p < 0.05: 
        print('Significant')
    else: 
        print('Non Significant')
        
print('Number of tablets')
ztestmean_poverty(train, 'v18q1')
# Number of Phones
print('Number of phones')
ztestmean_poverty(train, 'qmobilephone')
numeric = {'rooms':['rooms'],
           'males': ['r4h1', 'r4h2', 'r4h3'],
          'females': ['r4m1', 'r4m2', 'r4m3'],
          'persons': ['r4t1', 'r4t2', 'r4t3'],
          'sizeohhold':['tamhog'],
          '#ofpersons':['tamviv'],
          'hholdsize':['hhsize'],
          }
#-unfinished.
