#Trying to follow the concept of these 2 article, and more

#https://towardsdatascience.com/predicting-the-survival-of-titanic-passengers-30870ccc7e8

#https://www.kaggle.com/mohamedtimor/titanic-dataset



import numpy as np

import pandas as pd 



# data visualization

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style



import warnings

warnings.filterwarnings('ignore')
#get data

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")



#merge train & test data to perform data exploration and transformation 

df = pd.concat([train_df, test_df] , sort = True)
#Data Exploration/Analysis

df.info()
df.describe()
df.head(3)
#check in details what data actuall missing

total = df.isnull().sum().sort_values(ascending=False)

percent_1 = df.isnull().sum()/df.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

display(missing_data.head(5))

print(df.shape)
# Check the correlation for the current numeric feature set.

print(df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr())

sns.heatmap(df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr(), annot=True, fmt = ".2f", cmap = "coolwarm")
df.columns.values
#Train set evaluate

survived = 'survived'

not_survived = 'not survived'

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))

women = train_df[train_df['Sex']=='female']

men = train_df[train_df['Sex']=='male']

ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)

ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)

ax.legend()

ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)

ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)

ax.legend()

_ = ax.set_title('Male')
FacetGrid = sns.FacetGrid(train_df, row='Embarked', size=4.5, aspect=1.6)

FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )

FacetGrid.add_legend()
sns.barplot(x='Pclass', y='Survived', data=train_df)
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
train_df['relatives'] = train_df['SibSp'] + train_df['Parch']

axes = sns.factorplot('relatives','Survived', 

                      data=train_df, aspect = 2.5, )
titles = set()

for name in df['Name']:

    titles.add(name.split(',')[1].split('.')[0].strip())

print(titles)
Title_Dictionary = {

    "Capt": "Officer",

    "Col": "Officer",

    "Major": "Officer",

    "Jonkheer": "Royalty",

    "Don": "Royalty",

    "Dona": "Royalty",

    "Sir" : "Royalty",

    "Dr": "Officer",

    "Rev": "Officer",

    "the Countess":"Royalty",

    "Mme": "Mrs",

    "Mlle": "Miss",

    "Ms": "Mrs",

    "Mr" : "Mr",

    "Mrs" : "Mrs",

    "Miss" : "Miss",

    "Master" : "Master",

    "Lady" : "Royalty"

}
#sur checking

sur = set()

for name in df['Name']:

    sur.add(name.split(' ')[0].replace(' ','').replace(',','').lower())

print(sur)
def get_titles():

    # we extract the title from each name

    df['Title'] = df['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    

    # a map of more aggregated title

    # we map each title

    df['Title'] = df.Title.map(Title_Dictionary)

    return df



df = get_titles()



df['Title'].value_counts()
def get_sur():

    df['sur'] = df['Name'].map(lambda name:name.split(' ')[0].replace(' ','').replace(',','').lower())

    return df



df = get_sur()

df['sur'].value_counts()
print("Train- Missing Age")

print(train_df.iloc[:891].Age.isnull().sum())



print("Test- Missing Age")

print(test_df.iloc[:891].Age.isnull().sum())
grouped_train = df.groupby(['Sex','Pclass','Title'])

grouped_median_train = grouped_train.median()

grouped_median_train = grouped_median_train.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]

grouped_median_train
def fill_age(row):

    condition = (

        (grouped_median_train['Sex'] == row['Sex']) & 

        (grouped_median_train['Title'] == row['Title']) & 

        (grouped_median_train['Pclass'] == row['Pclass'])

    ) 

    return grouped_median_train[condition]['Age'].values[0]





def process_age():

    global df

    # a function that fills the missing values of the Age variable

    df['Age'] = df.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)

    return df



df = process_age()
#make dummies for title 

def process_names():

    global df

    # we clean the Name variable

    df.drop('Name', axis=1, inplace=True)

    

    # encoding in dummy variable

    titles_dummies = pd.get_dummies(df['Title'], prefix='Title')

    df = pd.concat([df, titles_dummies], axis=1)

    

    # removing the title variable

    df.drop('Title', axis=1, inplace=True)

    

    return df



df = process_names()

df.sample(5)
#make dummies for sur 

def process_sur():

    global df

    

    # encoding in dummy variable

    sur_dummies = pd.get_dummies(df['sur'], prefix='sur')

    df = pd.concat([df, sur_dummies], axis=1)

    

    # removing the title variable

    df.drop('sur', axis=1, inplace=True)

    

    return df



df = process_sur()

df.sample(5)
print("Train- Missing Cabin")

print(train_df.iloc[:891].Cabin.isnull().sum())



print("Test- Missing Cabin")

print(test_df.iloc[:891].Cabin.isnull().sum())
train_cabin, test_cabin = set(), set()



for c in df.iloc[:891]['Cabin']:

    try:

        train_cabin.add(c[0])

    except:

        train_cabin.add('U') # replaces NaN values with U (for Unknow)

        

for c in df.iloc[891:]['Cabin']:

    try:

        test_cabin.add(c[0])

    except:

        test_cabin.add('U') 



print("train_cabin -> " +str(train_cabin))



print("test_cabin -> " +str(test_cabin))
#don't have any cabin letter in the test set that is not present in the train set.
def process_cabin():

    global df    

    # replacing missing cabins with U (for Uknown)

    df.Cabin.fillna('U', inplace=True)

    

    # mapping each Cabin value with the cabin letter

    df['Cabin'] = df['Cabin'].map(lambda c: c[0])

    

    # dummy encoding ...

    cabin_dummies = pd.get_dummies(df['Cabin'], prefix='Cabin')    

    df = pd.concat([df, cabin_dummies], axis=1)



    df.drop('Cabin', axis=1, inplace=True)

    return df



df = process_cabin()

df.head()
df['Embarked'].describe()
common_value = 'S'

df['Embarked'] = df['Embarked'].fillna(common_value)
def process_embarked():

    global df

    # two missing embarked values - filling them with the most frequent one in the train  set(S)

    df.Embarked.fillna('S', inplace=True)

    # dummy encoding 

    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')

    df = pd.concat([df, embarked_dummies], axis=1)

    df.drop('Embarked', axis=1, inplace=True)

    return df



df = process_embarked()
df['Fare'] = df['Fare'].fillna(df.Fare.mean())

df['Fare'] = df['Fare'].astype(int)
def process_sex():

    global df

    # mapping string values to numerical one 

    df['Sex'] = df['Sex'].map({'male':1, 'female':0})

    return df



df = process_sex()
def process_pclass():

    

    global df

    # encoding into 3 categories:

    pclass_dummies = pd.get_dummies(df['Pclass'], prefix="Pclass")

    

    # adding dummy variable

    df = pd.concat([df, pclass_dummies],axis=1)

    

    # removing "Pclass"

    df.drop('Pclass',axis=1,inplace=True)

    return df



combined = process_pclass()
def cleanTicket(ticket):

    ticket = ticket.replace('.', '')

    ticket = ticket.replace('/', '')

    ticket = ticket.split()

    ticket = map(lambda t : t.strip(), ticket)

    ticket = list(filter(lambda t : not t.isdigit(), ticket))

    if len(ticket) > 0:

        return ticket[0]

    else: 

        return 'XXX'



tickets = set()

for t in df['Ticket']:

    tickets.add(cleanTicket(t))

    

print(len(tickets))
def process_ticket():

    

    global df

    

    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)

    def cleanTicket(ticket):

        ticket = ticket.replace('.','')

        ticket = ticket.replace('/','')

        ticket = ticket.split()

        ticket = map(lambda t : t.strip(), ticket)

        ticket = list(filter(lambda t : not t.isdigit(), ticket))

        

        if len(ticket) > 0:

            return ticket[0]

        else: 

            return 'XXX'

    



    # Extracting dummy variables from tickets:



    df['Ticket'] = df['Ticket'].map(cleanTicket)

    tickets_dummies = pd.get_dummies(df['Ticket'], prefix='Ticket')

    df = pd.concat([df, tickets_dummies], axis=1)

    df.drop('Ticket', inplace=True, axis=1)

    return df



df = process_ticket()
df.sample(5)
def process_family():

    

    global df

    # introducing a new feature : the size of families (including the passenger)

    df['FamilySize'] = df['Parch'] + df['SibSp'] + 1



    # introducing other features based on the family size

    df['Singleton'] = df['FamilySize'].map(lambda s: 1 if s == 1 else 0)

    df['SmallFamily'] = df['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)

    df['LargeFamily'] = df['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

    return df



df = process_family()
def process_parch():

    global df

    

    # dummy encoding 

    parch_dummies = pd.get_dummies(df['Parch'], prefix='Parch')

    df = pd.concat([df, parch_dummies], axis=1)

    df.drop('Parch', axis=1, inplace=True)

    return df



df = process_parch()



def process_SibSp():

    global df

    

    # dummy encoding 

    SibSp_dummies = pd.get_dummies(df['SibSp'], prefix='SibSp')

    df = pd.concat([df, SibSp_dummies], axis=1)

    df.drop('SibSp', axis=1, inplace=True)

    return df



df = process_SibSp()
df['Age'] = df['Age'].astype(int)

df.loc[ df['Age'] <= 11, 'Age'] = 0

df.loc[(df['Age'] > 11) & (df['Age'] <= 18), 'Age'] = 1

df.loc[(df['Age'] > 18) & (df['Age'] <= 22), 'Age'] = 2

df.loc[(df['Age'] > 22) & (df['Age'] <= 27), 'Age'] = 3

df.loc[(df['Age'] > 27) & (df['Age'] <= 33), 'Age'] = 4

df.loc[(df['Age'] > 33) & (df['Age'] <= 40), 'Age'] = 5

df.loc[(df['Age'] > 40) & (df['Age'] <= 66), 'Age'] = 6

df.loc[ df['Age'] > 66, 'Age'] = 6



# let's see how it's distributed 

df['Age'].value_counts()
def process_age():

    global df

    

    # dummy encoding 

    embarked_dummies = pd.get_dummies(df['Age'], prefix='Age')

    df = pd.concat([df, embarked_dummies], axis=1)

    df.drop('Age', axis=1, inplace=True)

    return df



df = process_age()

df.sample(3)
df['Fare'].describe()
df.loc[ df['Fare'] <= 7.91, 'Fare'] = 0

df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1

df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2

df.loc[(df['Fare'] > 31) & (df['Fare'] <= 99), 'Fare']   = 3

df.loc[(df['Fare'] > 99) & (df['Fare'] <= 250), 'Fare']   = 4

df.loc[ df['Fare'] > 250, 'Fare'] = 5

df['Fare'] = df['Fare'].astype(int)
def process_fare():

    global df

    

    # dummy encoding 

    embarked_dummies = pd.get_dummies(df['Fare'], prefix='Fare')

    df = pd.concat([df, embarked_dummies], axis=1)

    df.drop('Fare', axis=1, inplace=True)

    return df



df = process_fare()

df.sample(3)
print(df.sample(3))

print(df.shape)
df.drop(['PassengerId'], 1, inplace=True)

feature_train_df = df.iloc[:891].copy()

df.drop(['Survived'], 1, inplace=True)
def recover_train_test_target():

    global df

    

    passengerId = test_df['PassengerId']

    targets = pd.read_csv('../input/train.csv', usecols=['Survived'])['Survived'].values

    train = df.iloc[:891]

    test = df.iloc[891:]

    

    return train, test, targets,passengerId



train, test, targets, passengerId= recover_train_test_target()
param = ['Sex',

 'Title_Master',

 'Title_Miss',

 'Title_Mr',

 'Title_Mrs',

 'Title_Officer',

 'Title_Royalty',

 'sur_abbott',

 'sur_ahlin',

 'sur_aks',

 'sur_albimona',

 'sur_ali',

 'sur_allison',

 'sur_andersen-jensen',

 'sur_anderson',

 'sur_andrew',

 'sur_andrews',

 'sur_angle',

 'sur_appleton',

 'sur_arnold-franchi',

 'sur_artagaveytia',

 'sur_asplund',

 'sur_attalah',

 'sur_ayoub',

 'sur_backstrom',

 'sur_baclini',

 'sur_bailey',

 'sur_banfield',

 'sur_barah',

 'sur_barbara',

 'sur_barber',

 'sur_barkworth',

 'sur_bateman',

 'sur_baumann',

 'sur_baxter',

 'sur_beane',

 'sur_becker',

 'sur_beckwith',

 'sur_beesley',

 'sur_behr',

 'sur_berriman',

 'sur_bidois',

 'sur_bing',

 'sur_bishop',

 'sur_bjornstrom-steffansson',

 'sur_blackwell',

 'sur_blank',

 'sur_bonnell',

 'sur_boulos',

 'sur_bourke',

 'sur_bracken',

 'sur_bradley',

 'sur_braund',

 'sur_brewe',

 'sur_brown',

 'sur_bryhl',

 'sur_buss',

 'sur_butler',

 'sur_butt',

 'sur_byles',

 'sur_bystrom',

 'sur_cacic',

 'sur_cairns',

 'sur_calderhead',

 'sur_caldwell',

 'sur_calic',

 'sur_cameron',

 'sur_campbell',

 'sur_canavan',

 'sur_caram',

 'sur_carbines',

 'sur_cardeza',

 'sur_carlsson',

 'sur_carr',

 'sur_carrau',

 'sur_cavendish',

 'sur_chaffee',

 'sur_chambers',

 'sur_chapman',

 'sur_chip',

 'sur_christy',

 'sur_clarke',

 'sur_cleaver',

 'sur_clifford',

 'sur_cohen',

 'sur_coleff',

 'sur_collander',

 'sur_colley',

 'sur_connolly',

 'sur_coutts',

 'sur_crosby',

 'sur_cunningham',

 'sur_dahl',

 'sur_dahlberg',

 'sur_daly',

 'sur_danbom',

 'sur_daniel',

 'sur_davidson',

 'sur_davis',

 'sur_davison',

 'sur_de',

 'sur_dean',

 'sur_del',

 'sur_devaney',

 'sur_dick',

 'sur_dodge',

 'sur_doling',

 'sur_dorking',

 'sur_douglas',

 'sur_dowdell',

 'sur_downton',

 'sur_drew',

 'sur_duff',

 'sur_duran',

 'sur_eitemiller',

 'sur_elias',

 'sur_emanuel',

 'sur_fahlstrom',

 'sur_farthing',

 'sur_faunthorpe',

 'sur_flynn',

 'sur_foo',

 'sur_ford',

 'sur_foreman',

 'sur_fortune',

 'sur_fox',

 'sur_frauenthal',

 'sur_frolicher-stehli',

 'sur_frost',

 'sur_fry',

 'sur_funk',

 'sur_futrelle',

 'sur_fynney',

 'sur_gale',

 'sur_garside',

 'sur_gaskell',

 'sur_gavey',

 'sur_gee',

 'sur_giglio',

 'sur_giles',

 'sur_gill',

 'sur_gillespie',

 'sur_gilnagh',

 'sur_givard',

 'sur_glynn',

 'sur_goldenberg',

 'sur_goldschmidt',

 'sur_goldsmith',

 'sur_goodwin',

 'sur_greenfield',

 'sur_guggenheim',

 'sur_gustafsson',

 'sur_haas',

 'sur_hagland',

 'sur_hakkarainen',

 'sur_hale',

 'sur_hamalainen',

 'sur_hansen',

 'sur_harder',

 'sur_harknett',

 'sur_harper',

 'sur_harrington',

 'sur_harris',

 'sur_harrison',

 'sur_hart',

 'sur_hassab',

 'sur_hassan',

 'sur_hawksford',

 'sur_hays',

 'sur_healy',

 'sur_hedman',

 'sur_hegarty',

 'sur_heikkinen',

 'sur_heininen',

 'sur_henry',

 'sur_herman',

 'sur_hewlett',

 'sur_hickman',

 'sur_hirvonen',

 'sur_hocking',

 'sur_hold',

 'sur_homer',

 'sur_honkanen',

 'sur_hood',

 'sur_hosono',

 'sur_hoyt',

 'sur_hunt',

 'sur_ilett',

 'sur_ilmakangas',

 'sur_isham',

 'sur_jalsevac',

 'sur_jansson',

 'sur_jarvis',

 'sur_jenkin',

 'sur_jensen',

 'sur_jermyn',

 'sur_johannesen-bratthammer',

 'sur_johansson',

 'sur_johnson',

 'sur_johnston',

 'sur_jonsson',

 'sur_jussila',

 'sur_kallio',

 'sur_karun',

 'sur_kelly',

 'sur_kent',

 'sur_kimball',

 'sur_kink-heilmann',

 'sur_kirkland',

 'sur_klaber',

 'sur_knight',

 'sur_kvillner',

 'sur_lahtinen',

 'sur_laitinen',

 'sur_lam',

 'sur_landergren',

 'sur_lang',

 'sur_larsson',

 'sur_leader',

 'sur_leeni',

 'sur_lefebre',

 'sur_lehmann',

 'sur_leitch',

 'sur_lemore',

 'sur_lesurer',

 'sur_levy',

 'sur_lewy',

 'sur_leyson',

 'sur_lindahl',

 'sur_lindblom',

 'sur_lindqvist',

 'sur_lobb',

 'sur_long',

 'sur_louch',

 'sur_lulic',

 'sur_mack',

 'sur_madigan',

 'sur_madsen',

 'sur_mamee',

 'sur_mangan',

 'sur_mannion',

 'sur_marechal',

 'sur_marvin',

 'sur_masselmani',

 'sur_matthews',

 'sur_mccarthy',

 'sur_mccormack',

 'sur_mccoy',

 'sur_mcdermott',

 'sur_mcevoy',

 'sur_mcgough',

 'sur_mcgovern',

 'sur_mcgowan',

 'sur_mckane',

 'sur_meanwell',

 'sur_meek',

 'sur_mellinger',

 'sur_mellors',

 'sur_meyer',

 'sur_millet',

 'sur_minahan',

 'sur_mockler',

 'sur_moen',

 'sur_molson',

 'sur_montvila',

 'sur_moor',

 'sur_moraweck',

 'sur_morley',

 'sur_moss',

 'sur_moubarek',

 'sur_moussa',

 'sur_mudd',

 'sur_mullens',

 'sur_murphy',

 'sur_najib',

 'sur_nakid',

 'sur_nasser',

 'sur_natsch',

 'sur_newell',

 'sur_nicholls',

 'sur_nicholson',

 'sur_nicola-yarred',

 'sur_nilsson',

 'sur_niskanen',

 'sur_norman',

 'sur_nye',

 'sur_nysten',

 "sur_o'driscoll",

 "sur_o'dwyer",

 "sur_o'leary",

 "sur_o'sullivan",

 'sur_ohman',

 'sur_olsen',

 'sur_olsson',

 'sur_oreskovic',

 'sur_osman',

 'sur_ostby',

 'sur_otter',

 'sur_padro',

 'sur_pain',

 'sur_palsson',

 'sur_panula',

 'sur_parkes',

 'sur_parr',

 'sur_parrish',

 'sur_partner',

 'sur_pears',

 'sur_penasco',

 'sur_pengelly',

 'sur_pernot',

 'sur_persson',

 'sur_peter',

 'sur_peters',

 'sur_petranec',

 'sur_petroff',

 'sur_pettersson',

 'sur_peuchen',

 'sur_phillips',

 'sur_pickard',

 'sur_pinsky',

 'sur_ponesell',

 'sur_porter',

 'sur_quick',

 'sur_reeves',

 'sur_reuchlin',

 'sur_reynaldo',

 'sur_rice',

 'sur_richard',

 'sur_richards',

 'sur_ridsdale',

 'sur_ringhini',

 'sur_robbins',

 'sur_robins',

 'sur_roebling',

 'sur_romaine',

 'sur_rood',

 'sur_rosblom',

 'sur_ross',

 'sur_rothes',

 'sur_rothschild',

 'sur_rugg',

 'sur_ryan',

 'sur_ryerson',

 'sur_saad',

 'sur_saalfeld',

 'sur_sage',

 'sur_salkjelsvik',

 'sur_sandstrom',

 'sur_sedgwick',

 'sur_seward',

 'sur_sharp',

 'sur_sheerlinck',

 'sur_shelley',

 'sur_silven',

 'sur_silverthorne',

 'sur_silvey',

 'sur_simonius-blumer',

 'sur_sinkkonen',

 'sur_sjoblom',

 'sur_skoog',

 'sur_slayter',

 'sur_slemen',

 'sur_sloper',

 'sur_smart',

 'sur_smith',

 'sur_sobey',

 'sur_soholt',

 'sur_stahelin-maeglin',

 'sur_stanley',

 'sur_stead',

 'sur_stewart',

 'sur_strandberg',

 'sur_stranden',

 'sur_strom',

 'sur_sunderland',

 'sur_sundman',

 'sur_sutton',

 'sur_taussig',

 'sur_taylor',

 'sur_thayer',

 'sur_thomas',

 'sur_thorne',

 'sur_tobin',

 'sur_toomey',

 'sur_tornquist',

 'sur_touma',

 'sur_troupiansky',

 'sur_trout',

 'sur_troutt',

 'sur_turja',

 'sur_turkula',

 'sur_turpin',

 'sur_uruchurtu',

 'sur_van',

 'sur_vander',

 'sur_vestrom',

 'sur_walker',

 'sur_watson',

 'sur_watt',

 'sur_weir',

 'sur_weisz',

 'sur_wells',

 'sur_west',

 'sur_white',

 'sur_wick',

 'sur_widener',

 'sur_wilhelms',

 'sur_williams',

 'sur_williams-lambert',

 'sur_woolner',

 'sur_wright',

 'sur_yasbeck',

 'sur_yrois',

 'sur_zabour',

 'Cabin_B',

 'Cabin_D',

 'Cabin_E',

 'Cabin_F',

 'Cabin_G',

 'Cabin_T',

 'Cabin_U',

 'Embarked_C',

 'Embarked_Q',

 'Embarked_S',

 'Pclass_1',

 'Pclass_2',

 'Pclass_3',

 'Ticket_A4',

 'Ticket_A5',

 'Ticket_C',

 'Ticket_CA',

 'Ticket_CASOTON',

 'Ticket_FC',

 'Ticket_FCC',

 'Ticket_LINE',

 'Ticket_PC',

 'Ticket_PP',

 'Ticket_SC',

 'Ticket_SCOW',

 'Ticket_SCParis',

 'Ticket_SOC',

 'Ticket_SOP',

 'Ticket_SOPP',

 'Ticket_STONO',

 'Ticket_SWPP',

 'Ticket_WC',

 'Ticket_WEP',

 'Ticket_XXX',

 'FamilySize',

 'Singleton',

 'SmallFamily',

 'LargeFamily',

 'Parch_0',

 'Parch_1',

 'Parch_2',

 'Parch_3',

 'Parch_4',

 'Parch_5',

 'Parch_6',

 'SibSp_0',

 'SibSp_1',

 'SibSp_2',

 'SibSp_3',

 'SibSp_4',

 'SibSp_5',

 'SibSp_8',

 'Age_0',

 'Age_1',

 'Age_2',

 'Age_4',

 'Age_5',

 'Age_6',

 'Fare_0',

 'Fare_1',

 'Fare_2',

 'Fare_3',

 'Fare_4',

 'Fare_5']



train = train[param]

test = test[param]
from sklearn.preprocessing import StandardScaler



ss = StandardScaler()

train_scale= pd.DataFrame(ss.fit_transform(train), columns=test.columns)

test_scale = pd.DataFrame(ss.fit_transform(test), columns=test.columns)
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC



def svc_param_selection(X, y, nfolds):

    Cs = [0.001, 0.01, 0.1, 1, 10]

    gammas = [0.001, 0.01, 0.1, 1]

    kernel =["linear","rbf"]

    param_grid = {'C': Cs, 'gamma' : gammas,'kernel':kernel}

    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds)

    grid_search.fit(X, y)

    grid_search.best_params_

    return grid_search.best_params_



param = svc_param_selection(train_scale,targets,10)

display(param)



#take times to run, use back local given param
from sklearn import svm

from sklearn.svm import SVC



svc = SVC(C=10,gamma=0.001,kernel='rbf')

svc.fit(train_scale, targets)

pred = svc.predict(test_scale).astype(int)



print(round(svc.score(train_scale, targets) * 100, 2))
#Save result into df, csv

passengerId = test_df['PassengerId']

new_df = pd.DataFrame(columns=['PassengerId','Survived'])

new_df['PassengerId'] = passengerId

new_df['Survived'] = pred



new_df.to_csv("submition.csv", index = False)