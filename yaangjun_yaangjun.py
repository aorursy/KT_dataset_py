# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



pd.set_option('display.Max_column', None)

pd.set_option('display.Max_row', None)



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.style.use('fivethirtyeight')

plt.rcParams['font.size'] = 18

plt.rcParams['patch.edgecolor'] = 'k'



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

print(train.shape)

print(test.shape)

train.head()
train.select_dtypes(np.int64).nunique().value_counts()
train.select_dtypes(np.int64).nunique().value_counts().sort_index().plot.bar(color='blue', figsize = (8,6), edgecolor='k', linewidth=2)

# data type이 int 인거만 뽑아서 unique value의 갯수를 세서, unique값별로 column의 갯수를 세서, index로 정리해서 plot

plt.xlabel('Number of Unique Values')

plt.ylabel('Count')

plt.title('Count of Unique Values in Integer Columns')
from collections import OrderedDict



plt.figure(figsize = (20,12))

plt.style.use('fivethirtyeight')



#Color mapping

# OrderedDict : value뿐만아니라 순서까지 관리하는 Dict.

colors = OrderedDict({ 1: 'red', 2 : 'orange', 3 : 'blue', 4 : 'green'})

poverty_mapping = OrderedDict({ 1: 'extreme', 2: 'moderate', 3:'vulnerable', 4: 'non vulnerable'})



# Iterate through the float columns 

# enumerate : 반복문의 index와 value를 튜플로 반환함

for i, col in enumerate(train.select_dtypes('float')):

    ax = plt.subplot(4 , 2, i+1)

    # Iterate through the poverty levels

    for poverty_level, color in colors.items():

         # Plot each poverty level as a separate line

            sns.kdeplot(train.loc[train['Target'] == poverty_level, col ]. dropna(),

                       ax = ax, color = color, label = poverty_mapping[poverty_level])

            #kdeplot : 커널 밀도(kernel density)는 커널이라는 함수를 겹치는 방법으로 히스토그램보다 부드러운 형태의 분포 곡선을 보여주는 방법

            

    plt.title(f'{col.capitalize()} Distribution')

    # f''안에 {}가 들어가면 변수를 사용할 수 있음. 

    # .capitalize() -> 첫글자만 대문자

    plt.xlabel(f'{col}')

    plt.ylabel('Density')



plt.subplots_adjust(top=2) # 서브플롯간 간격 조절 가능함
train.select_dtypes('object').head()
mapping = {'yes' : 1, 'no' : 0}



#Apply same operation to both train and test

for df in [train, test]:

    df['dependency'] = df['dependency'].replace(mapping).astype(np.float64)

    df['edjefe'] = df['edjefe'].replace(mapping).astype(np.float64)

    df['edjefa'] = df['edjefa'].replace(mapping).astype(np.float64)

    

train[['dependency', 'edjefe', 'edjefa']].describe()
train[['dependency', 'edjefe', 'edjefa']].describe().transpose()
plt.figure(figsize=(16,10))



for i , col in enumerate(['dependency', 'edjefe', 'edjefa']):

    ax = plt.subplot(3, 1, i+1)

    for poverty_level, color in colors.items():

        sns.kdeplot(train.loc[train['Target'] == poverty_level, col].dropna(),

                   ax = ax, color = color, label = poverty_mapping[poverty_level])

        

    plt.title(f'{col.capitalize()} Distribution')

    plt.xlabel(f'{col}')

    plt.ylabel('Density')

    

plt.subplots_adjust(top = 2)

               

test['Target'] = np.nan

data = train.append(test, ignore_index = True)

# append를 사용하면 concat처럼 아래 붙음

print(data.shape)

data.head()
# Heads of household

heads = data.loc[data['parentesco1'] == 1].copy()



#Labels for training

train_labels = data.loc[(data['Target'].notnull()) & (data['parentesco1'] == 1), ['Target', 'idhogar']]



#value counts of target

label_counts = train_labels['Target'].value_counts().sort_index()



# Bar plot of occurrences of each label

label_counts.plot.bar(figsize=(8,6), color=colors.values(), edgecolor = 'k', linewidth=2)



#Formatting

plt.xlabel('Poverty Level')

plt.ylabel('Count')

plt.xticks([x - 1 for x in poverty_mapping.keys()],

          list(poverty_mapping.values()), rotation = 60)

plt.title('Poverty Level Breakdown')



label_counts
# Groupby the household and figure out the number of unique values

all_equal = train.groupby('idhogar')['Target'].apply(lambda x : x.nunique() == 1)



# households where targets are not all equal

not_equal = all_equal[all_equal != True]

print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))

train[train['idhogar'] == not_equal.index[0]][['idhogar', 'parentesco1', 'Target']]
households_leader = train.groupby('idhogar')['parentesco1'].sum()



# Find households without a head

households_no_head = train.loc[train['idhogar'].isin(households_leader[households_leader == 0].index), :]



print('There are {} households without a head.'.format(households_no_head['idhogar'].nunique()))
# Find households without a head and where labels are different

households_no_head_equal = households_no_head.groupby('idhogar')['Target'].apply(lambda x : x.nunique() == 1)

print('{} Households with no head have different labels'.format(sum(households_no_head_equal == False)))
# Iterate through each household

for household in not_equal.index:

    #Find the correct label (for the head of household)

    true_target = int(train[ (train['idhogar'] == household) & (train['parentesco1'] == 1.0)]['Target'])

    

    #SEt the correct label for all members in the household

    train.loc[train['idhogar'] == household , 'Target'] = true_target

    

#Groupby the household and figure out the number of unique values

all_equal = train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)



# Households where targets are not all equal

not_equal = all_equal[all_equal != True]

print('There are {} households where the family members do not all have the same Target.'.format(len(not_equal)))
# Number of missing in each column

missing = pd.DataFrame(data.isnull().sum()).rename(columns = {0 : 'total'})

#Create a percentage missing

missing['percent'] = missing['total'] / len(data)



missing.sort_values('percent', ascending = False).drop('Target').head(10)
def plot_value_counts(df, col, heads_only = False):

    if heads_only:

        df = df.loc[df['parentesco1'] == 1].copy()

    

    plt.figure(figsize=( 8 ,6 ))

    df[col].value_counts().sort_index().plot.bar(color ='blue', edgecolor='k', linewidth = 2)

    plt.xlabel(f'{col}')

    plt.title(f'{col} Value Counts')

    plt.ylabel('Count')

    plt.show()

plot_value_counts(heads, 'v18q1')

# heads : 위에서 가장의 데이터만 뽑아서 복사해놓음
heads.groupby('v18q')['v18q1'].apply(lambda x: x.isnull().sum())
data['v18q1'] = data['v18q1'].fillna(0)
# Variables indicating home ownership

own_variables = [x for x in data if x.startswith('tipo')]



# Plot of the home ownership variables for home missing rent payments

data.loc[data['v2a1'].isnull(), own_variables].sum().plot.bar(figsize=(10,8), color = 'green', edgecolor = 'k', linewidth=2)

plt.xticks([0,1,2,3,4], ['Own and Paid off', 'Owns and Paying', 'Rented', 'Precarious', 'Other'], rotation=60)

plt.title('Home Ownership Status  for Households Missing Rent Payments', size=18)

# Fill in house holds that own the house with 0 rent payment

data.loc[(data['tipovivi1'] == 1), 'v2a1'] = 0



#Create missing rent payment column

data['v2a1-missing'] = data['v2a1'].isnull()

data['v2a1-missing'].value_counts()
data.loc[data['rez_esc'].notnull()]['age'].describe()
data.loc[data['rez_esc'].isnull()]['age'].describe()
# If individual is over 19 or younger than 7 ans missing years behind, set it to 0

data.loc[ (((data['age']>19) | (data['age']<7)) & (data['rez_esc'].isnull())), 'rez_esc'] = 0



# Add a flag for those between 7 and 19 with a missing value

data['rez_esc-missing'] = data['rez_esc'].isnull()
data.loc[data['rez_esc'] > 5 , 'rez_esc'] = 5
def plot_categoricals(x, y, data, annotate = True):

    """Plot counts of two categoricals.

    Size is raw count for each grouping.

    Percentages are for a given value of y."""

    

    # Raw counts 

    raw_counts = pd.DataFrame(data.groupby(y)[x].value_counts(normalize = False))

    raw_counts = raw_counts.rename(columns = {x: 'raw_count'})

    

    # Calculate counts for each group of x and y

    counts = pd.DataFrame(data.groupby(y)[x].value_counts(normalize = True))

    

    # Rename the column and reset the index

    counts = counts.rename(columns = {x: 'normalized_count'}).reset_index()

    counts['percent'] = 100 * counts['normalized_count']

    

    # Add the raw count

    counts['raw_count'] = list(raw_counts['raw_count'])

    

    plt.figure(figsize = (14, 10))

    # Scatter plot sized by percent

    plt.scatter(counts[x], counts[y], edgecolor = 'k', color = 'lightgreen',s = 100 * np.sqrt(counts['raw_count']), marker = 'o', alpha = 0.6, linewidth = 1.5)

    

    if annotate:

        # Annotate the plot with text

        for i, row in counts.iterrows():

            # Put text with appropriate offsets

            plt.annotate(xy = (row[x] - (1 / counts[x].nunique()),  row[y] - (0.15 / counts[y].nunique())), color = 'navy',text = f"{round(row['percent'], 1)}%")

        

    # Set tick marks

    plt.yticks(counts[y].unique())

    plt.xticks(counts[x].unique())

    

    # Transform min and max to evenly space in square root domain

    sqr_min = int(np.sqrt(raw_counts['raw_count'].min()))

    sqr_max = int(np.sqrt(raw_counts['raw_count'].max()))

        

    # 5 sizes for legend

    msizes = list( range(sqr_min, sqr_max, int((sqr_max - sqr_min) / 5)))

    markers = []

    print(msizes)

    

    # Markers for legend

    for size in msizes:

        markers.append(plt.scatter([], [], s = 100 * size, label = f'{int(round(np.square(size) / 100) * 100)}',  color = 'lightgreen', alpha = 0.6, edgecolor = 'k', linewidth = 1.5))

        

    # Legend and formatting

    plt.legend(handles = markers, title = 'Counts',labelspacing = 3, handletextpad = 2, fontsize = 16, loc = (1.10, 0.19))

    

    plt.annotate(f'* Size represents raw count while % is for a given y value.', xy = (0, 1), xycoords = 'figure points', size = 10)

    

    # Adjust axes limits

    plt.xlim((counts[x].min() - (6 / counts[x].nunique()), counts[x].max() + (6 / counts[x].nunique())))

    plt.ylim((counts[y].min() - (4 / counts[y].nunique()), counts[y].max() + (4 / counts[y].nunique())))

    plt.grid(None)

    plt.xlabel(f"{x}"); plt.ylabel(f"{y}"); plt.title(f"{y} vs {x}");
plot_categoricals('rez_esc', 'Target', data);
plot_categoricals('escolari', 'Target', data, annotate =False)
plot_value_counts(data[(data['rez_esc-missing'] == 1)], 'Target')
plot_value_counts(data[(data['v2a1-missing'] == 1)], 'Target')
id_ = ['Id', 'idhogar', 'Target']
ind_bool = ['v18q', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 

            'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 

            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5', 

            'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10', 

            'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3', 

            'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 

            'instlevel9', 'mobilephone', 'rez_esc-missing']



ind_ordered = ['rez_esc', 'escolari', 'age']
hh_bool = ['hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo', 

           'paredpreb','pisocemento', 'pareddes', 'paredmad',

           'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisoother', 

           'pisonatur', 'pisonotiene', 'pisomadera',

           'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 

           'abastaguadentro', 'abastaguafuera', 'abastaguano',

            'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 

           'sanitario2', 'sanitario3', 'sanitario5',   'sanitario6',

           'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 

           'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 

           'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3',

           'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 

           'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 

           'computer', 'television', 'lugar1', 'lugar2', 'lugar3',

           'lugar4', 'lugar5', 'lugar6', 'area1', 'area2', 'v2a1-missing']



hh_ordered = [ 'rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1','r4m2','r4m3', 'r4t1',  'r4t2', 

              'r4t3', 'v18q1', 'tamhog','tamviv','hhsize','hogar_nin',

              'hogar_adul','hogar_mayor','hogar_total',  'bedrooms', 'qmobilephone']



hh_cont = ['v2a1', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'overcrowding']
sqr_ = ['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 

        'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq']
x= ind_bool + ind_ordered + id_ + hh_bool + hh_ordered + hh_cont + sqr_



from collections import Counter



print('There are no repeats : ', np.all(np.array(list(Counter(x).values())) == 1 ))

# np.all : 모든 원소가 참인지 평가하는 함수

print('We covered every variable : ', len(x) == data.shape[1])

sns.lmplot('age', 'SQBage', data = data, fit_reg=False)

plt.title('Squared Age vs Age');
sns.lmplot('edjefe', 'SQBedjefe', data=data, fit_reg=False)

plt.title('SQBedjefe vs edjefe');
data = data.drop(columns = sqr_)

data.shape
heads = data.loc[data['parentesco1'] == 1, :]

heads = heads[id_ + hh_bool+ hh_cont + hh_ordered]

heads.shape
corr_matrix = heads.corr()

#Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]



to_drop

tamhog_corr = corr_matrix.loc[corr_matrix['tamhog'].abs() > 0.9, corr_matrix['tamhog'].abs() > 0.9]

tamhog_corr
sns.heatmap(tamhog_corr, annot=True, cmap=plt.cm.autumn_r, fmt='.3f')
heads = heads.drop(columns = ['tamhog', 'hogar_total', 'r4t3'])
sns.lmplot('tamviv', 'hhsize', data, fit_reg=False, size = 8);

plt.title('Household size vs number of persons living in the household');
heads['hhsize-diff'] = heads['tamviv'] - heads['hhsize']

plot_categoricals('hhsize-diff', 'Target', heads)
corr_matrix = heads.corr()

#Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]



to_drop
corr_matrix.loc[corr_matrix['coopele'].abs() > 0.9, corr_matrix['coopele'].abs() > 0.9]
elec = []



for i , row in heads.iterrows():

    if row['noelec'] == 1:

        elec.append(0)

    elif row['coopele'] == 1:

        elec.append(1)

    elif row['public'] == 1:

        elec.append(2)

    elif row['planpri'] == 1:

        elec.append(3)

    else:

        elec.append(np.nan)

    

heads['elec'] = elec

heads['elec-missing'] = heads['elec'].isnull()



# heads = heads.drop(columns =['noelec', 'coopele', 'public', 'planpri'])
plot_categoricals('elec', 'Target', heads)
heads = heads.drop(columns = 'area2')

heads.groupby('area1')['Target'].value_counts(normalize = True)
heads.head()
heads['walls'] = np.argmax(np.array(heads[['epared1', 'epared2', 'epared3']]), axis = 1)

# heads = heads.drop(columns = ['epared1', 'epared2', 'epared3'])

plot_categoricals('walls', 'Target', heads)
heads['roof'] = np.argmax(np.array(heads[['etecho1', 'etecho2', 'etecho3']]), axis = 1)

heads = heads.drop(columns = ['etecho1', 'etecho2', 'etecho3'])

heads['floor'] = np.argmax(np.array(heads[['eviv1', 'eviv2', 'eviv3']]), axis = 1)

# heads = heads.drop(columns = ['eviv1', 'eviv2', 'eviv3'])
heads['walls+roof+floor'] = heads['walls'] + heads['roof'] + heads['floor']



plot_categoricals('walls+roof+floor', 'Target', heads, annotate = False)
counts = pd.DataFrame(heads.groupby(['walls+roof+floor'])['Target'].value_counts(normalize=True)).rename(columns={'Target' : 'Normalized Count'}).reset_index()

counts.head()
heads['warning'] = 1 * (heads['sanitario1'] + 

                         (heads['elec'] == 0) + 

                         heads['pisonotiene'] + 

                         heads['abastaguano'] + 

                         (heads['cielorazo'] == 0))
plt.figure(figsize = (10,6))

sns.violinplot(x='warning', y ='Target', data= heads)

plt.title('Target vs Warning Variable');
plot_categoricals('warning', 'Target', data = heads)
# Owns a refrigerator, computer, tablet, and television

heads['bonus'] = 1 * (heads['refrig'] + 

                      heads['computer'] + 

                      (heads['v18q1'] > 0) + 

                      heads['television'])



sns.violinplot('bonus', 'Target', data = heads,

                figsize = (10, 6));

plt.title('Target vs Bonus Variable');
heads['phones-per-capita'] = heads['qmobilephone'] / heads['tamviv']

heads['tablets-per-capita'] = heads['v18q1'] / heads['tamviv']

heads['rooms-per-capita'] = heads['rooms'] / heads['tamviv']

heads['rent-per-capita'] = heads['v2a1'] / heads['tamviv']
from scipy.stats import spearmanr
def plot_corrs(x,y):

    spr = spearmanr(x,y).correlation

    pcr = np.corrcoef(x,y)[0,1]

    

    data = pd.DataFrame({'x': x , 'y' : y})

    plt.figure(figsize = (6,4))

    sns.regplot('x', 'y', data= data, fit_reg = False)

    plt.title(f'Spearman : {round(spr,2)}; Pearson : {round(pcr,2)}')
x = np.array(range(100))

y = x **2



plot_corrs(x,y)
train_heads = heads.loc[heads['Target'].notnull(), :].copy()

pcorrs = pd.DataFrame(train_heads.corr()['Target'].sort_values()).rename(columns = {'Target':'pcorr'}).reset_index()

pcorrs = pcorrs.rename(columns = {'index' : 'feature'})

print('Most negatively correlated variables')

print(pcorrs.head())

print('\nMost positively correlated variables')

print(pcorrs.dropna().tail())
import warnings

warnings.filterwarnings('ignore', category = RuntimeWarning)



feats = []

scorr = []

pvalues = []



for c in heads:

    if heads[c].dtype != 'object':

        feats.append(c)

        

        scorr.append(spearmanr(train_heads[c], train_heads['Target']).correlation)

        pvalues.append(spearmanr(train_heads[c], train_heads['Target']).pvalue)



scorrs = pd.DataFrame({'feature': feats , 'scorr' : scorr, 'pvalue' : pvalues}).sort_values('scorr')



        
print('Most negative Spearman correlations:')

print(scorrs.head())

print('\nMost positive Spearman correlations:')

print(scorrs.dropna().tail())
corrs = pcorrs.merge(scorrs, on='feature')

corrs['diff'] = corrs['pcorr'] - corrs['scorr']

corrs.sort_values('diff').head()
corrs.sort_values('diff').dropna().tail()
sns.lmplot('dependency', 'Target', fit_reg = True, data=train_heads, x_jitter = 0.05, y_jitter=0.05)

plt.title('Target vs Dependency');
sns.lmplot('rooms-per-capita', 'Target', fit_reg = True, data=train_heads, x_jitter = 0.05, y_jitter=0.05)

plt.title('Target vs rooms per captiva');
variables = ['Target', 'dependency', 'warning', 'walls+roof+floor', 'meaneduc',

             'floor', 'r4m1', 'overcrowding']



corr_mat = train_heads[variables].corr().round(2)



plt.rcParams['font.size'] = 18

plt.figure(figsize = (12,12))

sns.heatmap(corr_mat, vmin = -0.5, vmax = 0.8, center = 0, cmap = plt.cm.RdYlGn_r, annot= True);
# plot_data.isnull().sum() -> meaneduc에 null 값이 있어서 임의로 내가 넣음



train_heads['meaneduc'].fillna(train_heads['meaneduc'].mean(), inplace=True)
import warnings

warnings.filterwarnings('ignore')



plot_data = train_heads[['Target', 'dependency', 'walls+roof+floor', 'meaneduc', 'overcrowding']]



grid = sns.PairGrid(data = plot_data, size = 4, diag_sharey=False, hue='Target', hue_order = [4,3,2,1],

                   vars = [x for x in list(plot_data.columns) if x != 'Target'])



grid.map_upper(plt.scatter, alpha=0.8, s=20)



grid.map_diag(sns.kdeplot)



grid.map_lower(sns.kdeplot, cmap=plt.cm.OrRd_r)

grid = grid.add_legend()

plt.suptitle('Feature Plots Colored By Target', size = 32, y = 1.05);
household_feats = list(heads.columns)
ind = data[id_ + ind_bool + ind_ordered]

ind.shape
corr_matrix = ind.corr()



upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]



to_drop
ind = ind.drop(columns = 'male')
ind[[c for c in ind if c.startswith('instl')]].head()
ind['inst'] = np.argmax(np.array(ind[[c for c in ind if c.startswith('instl')]]), axis = 1)

plot_categoricals('inst', 'Target', ind, annotate= False)
plt.figure(figsize = (10,8))

sns.violinplot(x= 'Target', y='inst', data = ind)

plt.title('Education Distribution by Target');
ind.shape
ind['escolari/age'] = ind['escolari'] / ind['age']



plt.figure(figsize = (10,8))

sns.violinplot('Target', 'escolari/age', data= ind);
ind['inst/age'] = ind['inst'] / ind['age']

ind['tech'] = ind['v18q'] + ind['mobilephone']

ind['tech'].describe()
range_ = lambda x : x.max() - x.min()

range_.__name__ = 'range_'



ind_agg = ind.drop(columns = 'Target').groupby('idhogar').agg(['min', 'max', 'sum', 'count', 'std', range_])

ind_agg.head()
new_col = []

for c in ind_agg.columns.levels[0]:

    for stat in ind_agg.columns.levels[1]:

        new_col.append(f'{c}-{stat}')

        

ind_agg.columns = new_col

ind_agg.head()
ind_agg.iloc[:,[0,1,2,3,6,7,8,9]].head()
corr_matrix = ind_agg.corr()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))



to_drop = [column for column in upper.columns if any(abs(upper[column])> 0.95)]



print(f'There ar {len(to_drop)} correlated columns to remove.')
ind_agg = ind_agg.drop(columns = to_drop)

ind_feats = list(ind_agg.columns)



final = heads.merge(ind_agg, on ='idhogar', how = 'left')



print('Rinal features shape : ', final.shape)
final.head()
corrs = final.corr()['Target']
corrs.sort_values().head()
corrs.sort_values().dropna().tail()
plot_categoricals('escolari-max', 'Target', final, annotate=False)
plt.figure(figsize = (10,6))

sns.violinplot(x='Target', y ='escolari-max', data=final)

plt.title('Max Schooling by Target');
plt.figure(figsize=(10,6))

sns.boxplot(x='Target', y = 'escolari-max', data=final)

plt.title('Max Schooling by Target');
plt.figure(figsize = (10, 6))

sns.boxplot(x = 'Target', y = 'meaneduc', data = final);

plt.xticks([0, 1, 2, 3], poverty_mapping.values())

plt.title('Average Schooling by Target');
plt.figure(figsize = (10, 6))

sns.boxplot(x = 'Target', y = 'overcrowding', data = final);

plt.xticks([0, 1, 2, 3], poverty_mapping.values())

plt.title('Overcrowding by Target');
head_gender = ind.loc[ind['parentesco1'] == 1 , ['idhogar', 'female']]

final = final.merge(head_gender, on = 'idhogar', how='left').rename(columns={'female' : 'female-head'})
final.groupby('female-head')['Target'].value_counts(normalize=True)
sns.violinplot(x='female-head', y='Target', data=final)

plt.title('Target by Female head of Household');
plt.figure(figsize= (8,8))

sns.boxplot(x='Target', y='meaneduc', hue='female-head', data=final)

plt.title('Average Education by Target and Female Head of Household', size=16);
final.groupby('female-head')['meaneduc'].agg(['mean', 'count'])
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, make_scorer

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline



scorer = make_scorer(f1_score, greater_is_better=True, average='macro')
train_labels = np.array(list(final[final['Target'].notnull()]['Target'].astype(np.uint8)))



train_set = final[final['Target'].notnull()].drop(columns = ['Id', 'idhogar', 'Target'])

test_set = final[final['Target'].isnull()].drop(columns = ['Id', 'idhogar', 'Target'])



submission_base = test[['Id', 'idhogar']].copy()
features = list(train_set.columns)



#imputer는 각 속성의 중앙값을 계산하고, 그 결과를 statistics_ 인스턴스 변수에 저장합니다. 



pipeline = Pipeline([('imputer', Imputer(strategy = 'median')),

                     ('scaler', MinMaxScaler())])



train_set = pipeline.fit_transform(train_set)

test_set = pipeline.transform(test_set)
model = RandomForestClassifier(n_estimators = 100, random_state=10, n_jobs= -1)



cv_score = cross_val_score(model, train_set, train_labels, cv=10 , scoring = scorer)



print(f'10 Fold Cross Validation F1 Score= {round(cv_score.mean(), 4)} with std = {round(cv_score.std(), 4)}')
model.fit(train_set, train_labels)



feature_importances = pd.DataFrame({'feature' : features, 'importance':model.feature_importances_})

feature_importances.head()
def plot_feature_importances(df, n=10, threshold =None):

    plt.style.use('fivethirtyeight')

    

    df = df.sort_values('importance', ascending=False).reset_index(drop=True)

    # index를 reset하며 원래 index가 새로운 column에 저장되는데 그걸 drop시킴

    

    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])

    # 누적합

    

    plt.rcParams['font.size'] = 12

    df.loc[:n, :].plot.barh(y = 'importance_normalized',

                            x = 'feature', color='darkgreen',

                           edgecolor = 'k', figsize = (12,8),

                           legend = False, linewidth = 2)

    

    plt.xlabel('Normalized Importance', size = 18)

    plt.ylabel('')

    plt.title(f'{n} Most Important Features', size=18)

    plt.gca().invert_yaxis()

    

    if threshold:

        plt.figure(figsize = (8, 6))

        plt.plot(list(range(len(df))), df['cumulative_importance'], 'b-')

        plt.xlabel('Number of Features', size = 16); 

        plt.ylabel('Cumulative Importance', size = 16); 

        plt.title('Cumulative Feature Importance', size = 18);

        

        # Number of features needed for threshold cumulative importance

        # This is the index (will need to add 1 for the actual number)

        importance_index = np.min(np.where(df['cumulative_importance'] > threshold))

        

        # Add vertical line to plot

        plt.vlines(importance_index + 1, ymin = 0, ymax = 1.05, linestyles = '--', colors = 'red')

        plt.show();

        

        print('{} features required for {:.0f}% of cumulative importance.'.format(importance_index + 1, 

                                                                                  100 * threshold))

    

    return df
norm_f1 = plot_feature_importances(feature_importances, threshold=0.95)
def kde_target(df, variable):

    colors = {1:'red', 2:'orange', 3:'blue', 4:'green'}

    

    plt.figure(figsize=(12,8))

    

    df = df[df['Target'].notnull()]

    

    for level in df['Target'].unique():

        subset = df[df['Target'] == level].copy()

        sns.kdeplot(subset[variable].dropna(),

                    label = f'Poverty Level : {level}',

                    color = colors[int(subset['Target'].unique())])

    plt.xlabel(variable)

    plt.ylabel('Density')

    plt.title('{} Distribution'.format(variable.capitalize()));
kde_target(final, 'meaneduc')
kde_target(final, 'escolari/age-range_')
from sklearn.svm import LinearSVC

from sklearn.naive_bayes import GaussianNB #

from sklearn.neural_network import MLPClassifier #

from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV 

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #

from sklearn.neighbors import KNeighborsClassifier
import warnings

from sklearn.exceptions import ConvergenceWarning



warnings.filterwarnings('ignore', category = ConvergenceWarning)

warnings.filterwarnings('ignore', category = DeprecationWarning)

warnings.filterwarnings('ignore', category =UserWarning)



model_results = pd.DataFrame(columns = ['model', 'cv_mean', 'cv_std'])



def cv_model(train, train_labels, model, name, model_results=None):

    cv_scores = cross_val_score(model, train, train_labels, cv = 10, scoring=scorer, n_jobs = -1)

    print(f'10 Fold CV Score : {round(cv_scores.mean() , 5)} with std : {round(cv_scores.std(), 5)}')

    

    if model_results is not None:

        model_results = model_results.append(pd.DataFrame({'model' : name,

                                                           'cv_mean' : cv_scores.mean(),

                                                           'cv_std' : cv_scores.std()},

                                                           index = [0]),

                                            ignore_index = True)

        

        return model_results
model_results = cv_model(train_set, train_labels, LinearSVC(), 'LSVC', model_results)
model_results = cv_model(train_set, train_labels, GaussianNB(), 'GNB', model_results)
model_results = cv_model(train_set, train_labels, MLPClassifier(hidden_layer_sizes = (32,64,128,64,32)), 'MLP', model_results)
model_results = cv_model(train_set, train_labels, LinearDiscriminantAnalysis(), 'LDA', model_results)
model_results = cv_model(train_set, train_labels, RidgeClassifierCV(), 'RIDGE', model_results)
for n in [5,10,20]:

    print(f'\nKNN with {n} neighbors\n')

    model_results = cv_model(train_set, train_labels,

                             KNeighborsClassifier(n_neighbors = n),

                             f'knn-{n}', model_results)
from sklearn.ensemble import ExtraTreesClassifier



model_results = cv_model(train_set, train_labels ,

                         ExtraTreesClassifier(n_estimators = 100 , random_state = 10),

                        'EXT', model_results)
# Comparing Model Performance
model_results = cv_model(train_set, train_labels,

                         RandomForestClassifier(100, random_state = 10),

                        'RF', model_results)
model_results.set_index('model', inplace = True)

model_results['cv_mean'].plot.bar(color = 'orange', figsize = (8,6),

                                 yerr = list(model_results['cv_std']),

                                 edgecolor = 'k', linewidth = 2)

plt.title('Model F1 Score Results')

plt.ylabel('Mean F1 Score (with error bar)');

model_results.reset_index(inplace = True)
test_ids = list(final.loc[final['Target'].isnull(), 'idhogar'])
def submit(model, train, train_labels, test, test_ids):

    model.fit(train, train_labels)

    predictions = model.predict(test)

    predictions = pd.DataFrame({'idhogar' : test_ids,

                                'Target' : predictions})

    

    submission = submission_base.merge(predictions,

                                      on = 'idhogar',

                                      how='left').drop(columns = ['idhogar'])

    

    submission['Target'] = submission['Target'].fillna(4).astype(np.int8)

    

    return submission
rf_submission = submit(RandomForestClassifier(n_estimators = 100,

                                             random_state=10, n_jobs = -1),

                      train_set, train_labels, test_set, test_ids)



rf_submission.to_csv('rf_submission.csv', index=False)
train_set = pd.DataFrame(train_set, columns = features)



corr_matrix = train_set.corr()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]



to_drop
train_set = train_set.drop(columns = to_drop)

train_set.shape
test_set = pd.DataFrame(test_set, columns=features)

train_set, test_set = train_set.align(test_set, axis = 1, join ='inner')

features = list(train_set.columns)
from sklearn.feature_selection import RFECV

estimator = RandomForestClassifier(random_state = 10 , n_estimators = 100, n_jobs= -1)

selector = RFECV(estimator, step = 1, cv =3, scoring=scorer, n_jobs=-1)
selector.fit(train_set, train_labels)
plt.plot(selector.grid_scores_)

plt.xlabel('Number of Features')

plt.ylabel('Macro F1 Score')

plt.title('Feature Selection Scores')

selector.n_features_

rankings = pd.DataFrame({'feature' : list(train_set.columns), 'rank': list(selector.ranking_)}).sort_values('rank')

rankings.head(10)
train_selected = selector.transform(train_set)

test_selected = selector.transform(test_set)
selected_features = train_set.columns[np.where(selector.ranking_==1)]

train_selected = pd.DataFrame(train_selected, columns = selected_features)

test_selected = pd.DataFrame(test_selected, columns = selected_features)
model_results = cv_model(train_selected, train_labels, model, 'RF-SEL', model_results)
model_results.set_index('model', inplace = True)

model_results['cv_mean'].plot.bar(color = 'orange', figsize = (8,6),

                                  yerr = list(model_results['cv_std']),

                                  edgecolor='k', linewidth=2)

plt.title('Model F1 Score Results')

plt.ylabel('Mean F1  Score(with error bar)')

model_results.reset_index(inplace = True)
def macro_f1_score(labels, predictions):

    predictions = predictions.reshape(len(np.unique(labels)), -1).argmax(axis = 0)

    metric_value = f1_score(labels, predictions, average ='macro')

    return 'macro_f1', metric_value, True
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

from IPython.display import display



def model_gbm(features, labels, test_features, test_ids, nfolds = 5, return_preds = False, hyp = None):

    """Model using the GBM and cross validation.

       Trains with early stopping on each fold.

       Hyperparameters probably need to be tuned."""

    

    feature_names = list(features.columns)



    # Option for user specified hyperparameters

    if hyp is not None:

        # Using early stopping so do not need number of esimators

        if 'n_estimators' in hyp:

            del hyp['n_estimators']

        params = hyp

    

    else:

        # Model hyperparameters

        params = {'boosting_type': 'dart', 

                  'colsample_bytree': 0.88, 

                  'learning_rate': 0.028, 

                   'min_child_samples': 10, 

                   'num_leaves': 36, 'reg_alpha': 0.76, 

                   'reg_lambda': 0.43, 

                   'subsample_for_bin': 40000, 

                   'subsample': 0.54, 

                   'class_weight': 'balanced'}

    

    # Build the model

    model = lgb.LGBMClassifier(**params, objective = 'multiclass', 

                               n_jobs = -1, n_estimators = 10000,

                               random_state = 10)

    

    # Using stratified kfold cross validation

    strkfold = StratifiedKFold(n_splits = nfolds, shuffle = True)

    

    # Hold all the predictions from each fold

    predictions = pd.DataFrame()

    importances = np.zeros(len(feature_names))

    

    # Convert to arrays for indexing

    features = np.array(features)

    test_features = np.array(test_features)

    labels = np.array(labels).reshape((-1 ))

    

    valid_scores = []

    

    # Iterate through the folds

    for i, (train_indices, valid_indices) in enumerate(strkfold.split(features, labels)):

        

        # Dataframe for fold predictions

        fold_predictions = pd.DataFrame()

        

        # Training and validation data

        X_train = features[train_indices]

        X_valid = features[valid_indices]

        y_train = labels[train_indices]

        y_valid = labels[valid_indices]

        

        # Train with early stopping

        model.fit(X_train, y_train, early_stopping_rounds = 100, eval_metric = macro_f1_score,

                  eval_set = [(X_train, y_train), (X_valid, y_valid)] , eval_names = ['train', 'valid'], verbose = 200)

        

        # Record the validation fold score

        valid_scores.append(model.best_score_['valid']['macro_f1'])

        

        # Make predictions from the fold as probabilities

        fold_probabilitites = model.predict_proba(test_features)

        

        # Record each prediction for each class as a separate column

        for j in range(4):

            fold_predictions[(j + 1)] = fold_probabilitites[:, j]

            

        # Add needed information for predictions 

        fold_predictions['idhogar'] = test_ids

        fold_predictions['fold'] = (i+1)

        

        # Add the predictions as new rows to the existing predictions

        predictions = predictions.append(fold_predictions)

        

        # Feature importances

        importances += model.feature_importances_ / nfolds   

        

        # Display fold information

        display(f'Fold {i + 1}, Validation Score: {round(valid_scores[i], 5)}, Estimators Trained: {model.best_iteration_}')



    # Feature importances dataframe

    feature_importances = pd.DataFrame({'feature': feature_names,

                                        'importance': importances})

    

    valid_scores = np.array(valid_scores)

    display(f'{nfolds} cross validation score: {round(valid_scores.mean(), 5)} with std: {round(valid_scores.std(), 5)}.')

    

    # If we want to examine predictions don't average over folds

    if return_preds:

        predictions['Target'] = predictions[[1, 2, 3, 4]].idxmax(axis = 1)

        predictions['confidence'] = predictions[[1, 2, 3, 4]].max(axis = 1)

        return predictions, feature_importances

    

    # Average the predictions over folds

    predictions = predictions.groupby('idhogar', as_index = False).mean()

    

    # Find the class and associated probability

    predictions['Target'] = predictions[[1, 2, 3, 4]].idxmax(axis = 1)

    predictions['confidence'] = predictions[[1, 2, 3, 4]].max(axis = 1)

    predictions = predictions.drop(columns = ['fold'])

    

    # Merge with the base to have one prediction for each individual

    submission = submission_base.merge(predictions[['idhogar', 'Target']], on = 'idhogar', how = 'left').drop(columns = ['idhogar'])

        

    # Fill in the individuals that do not have a head of household with 4 since these will not be scored

    submission['Target'] = submission['Target'].fillna(4).astype(np.int8)

    

    # return the submission and feature importances along with validation scores

    return submission, feature_importances, valid_scores
%%capture --no-display

predictions, gbm_fi = model_gbm(train_set , train_labels, test_set, test_ids, return_preds=True)
predictions.head()
%%capture

submission, gbm_fi, valid_scores = model_gbm(train_set, train_labels, test_set, test_ids, return_preds = False)

submission = submission[['id', 'Target']]
submission.to_csv('gbm_baseline.csv', index=False)
'''

%%capture

submission, gbm_fi, valid_scores = model_gbm(train_set, train_labels, test_set, test_ids, nfolds=10, return_preds=False)

'''