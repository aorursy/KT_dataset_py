import numpy as np

import pandas as pd



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



# Set up Seaborn with default theme, scaling, and color palette

sns.set()



#Scikit-learn common imports

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline, make_pipeline
def import_train_test():

    train = pd.read_csv('../input/train.csv')

    test = pd.read_csv('../input/test.csv')

    

    return train, test



train, test = import_train_test()
print(train.shape)

train.head()
print(test.shape)

test.head()
print(f"There are {len(train.idhogar.unique())} unique households in the training set.")

print(f"There are {len(test.idhogar.unique())} unique households in the test set.")
fig, axs = plt.subplots(1,2)

fig.tight_layout()



sns.countplot(x='Target', data=train, ax=axs[0])

axs[0].set_title('Train')



sns.countplot(x='Target', data=train.copy().groupby('idhogar').first(), ax=axs[1])

axs[1].set_title('Train Households')



plt.show()
def is_boolean_column(data, column):

    return set(data[column].value_counts().keys()) == {0,1}

    

numeric_non_boolean_attributes = [column for column in train.columns if not is_boolean_column(train, column)]



train.hist(numeric_non_boolean_attributes, bins=50, figsize=(20,15))

plt.show()
q1s = train[numeric_non_boolean_attributes].quantile(0.25)

q3s = train[numeric_non_boolean_attributes].quantile(0.75)



iqrs = pd.DataFrame([q1s, q3s], index=['q1', 'q3']).transpose().drop('Target')

iqrs['iqr'] = iqrs.q3 - iqrs.q1

iqrs['iqr_amplified'] = iqrs.iqr * 1.5

iqrs['outlier_min'] = iqrs.q1 - iqrs.iqr_amplified

iqrs['outlier_max'] = iqrs.q3 + iqrs.iqr_amplified



def min_outlier_count(iqr_row):

    return len(train[train[iqr_row.name] < iqrs.loc[[iqr_row.name]].outlier_min[0]])



def max_outlier_count(iqr_row):

    return len(train[train[iqr_row.name] > iqrs.loc[[iqr_row.name]].outlier_max[0]])



iqrs['min_outlier_count'] = iqrs.apply(min_outlier_count, axis=1)

iqrs['max_outlier_count'] = iqrs.apply(max_outlier_count, axis=1)

iqrs[(iqrs.min_outlier_count > 0) | (iqrs.max_outlier_count > 0)]



with pd.option_context('display.max_columns', len(iqrs)):

    print(train[iqrs.index].describe())
with pd.option_context('display.max_columns', len(iqrs)):

    print(train[iqrs.index].describe())
correlations = train.copy().corr()



correlations['Target'].where(correlations['Target'].abs() > 0.25).dropna().sort_values(ascending=False)
fig, axs = plt.subplots(1,2)

fig.tight_layout()



sns.regplot(x='meaneduc', y='Target', data=train, ax=axs[0])

sns.regplot(x='hogar_nin', y='Target', data=train, ax=axs[1])



plt.show()
id_features = ['Id', 'idhogar']



features_and_descriptions = [('v2a1', 'Monthly rent payment'),

                             ('hacdor', '=1 Overcrowding by bedrooms'),

                             ('rooms', 'number of all rooms in the house'),

                             ('hacapo', '=1 Overcrowding by rooms'),

                             ('v14a', '=1 has bathroom in the household'),

                             ('refrig', '=1 if the household has refrigerator'),

                             ('v18q', 'owns a tablet'),

                             ('v18q1', 'number of tablets household owns'),

                             ('r4h1', 'Males younger than 12 years of age'),

                             ('r4h2', 'Males 12 years of age and older'),

                             ('r4h3', 'Total males in the household'),

                             ('r4m1', 'Females younger than 12 years of age'),

                             ('r4m2', 'Females 12 years of age and older'),

                             ('r4m3', 'Total females in the household'),

                             ('r4t1', 'persons younger than 12 years of age'),

                             ('r4t2', 'persons 12 years of age and older'),

                             ('r4t3', 'Total persons in the household'),

                             ('tamhog', 'size of the household'),

                             ('tamviv', 'number of persons living in the household'),

                             ('escolari', 'years of schooling'),

                             ('rez_esc', 'Years behind in school'),

                             ('hhsize', 'household size'),

                             ('paredblolad', '=1 if predominant material on the outside wall is block or brick'),

                             ('paredzocalo', '"=1 if predominant material on the outside wall is socket (wood,  zinc or absbesto"'),

                             ('paredpreb', '=1 if predominant material on the outside wall is prefabricated or cement'),

                             ('pareddes', '=1 if predominant material on the outside wall is waste material'),

                             ('paredmad', '=1 if predominant material on the outside wall is wood'),

                             ('paredzinc', '=1 if predominant material on the outside wall is zink'),

                             ('paredfibras', '=1 if predominant material on the outside wall is natural fibers'),

                             ('paredother', '=1 if predominant material on the outside wall is other'),

                             ('pisomoscer', '"=1 if predominant material on the floor is mosaic,  ceramic,  terrazo"'),

                             ('pisocemento', '=1 if predominant material on the floor is cement'),

                             ('pisoother', '=1 if predominant material on the floor is other'),

                             ('pisonatur', '=1 if predominant material on the floor is  natural material'),

                             ('pisonotiene', '=1 if no floor at the household'),

                             ('pisomadera', '=1 if predominant material on the floor is wood'),

                             ('techozinc', '=1 if predominant material on the roof is metal foil or zink'),

                             ('techoentrepiso', '"=1 if predominant material on the roof is fiber cement,  mezzanine "'),

                             ('techocane', '=1 if predominant material on the roof is natural fibers'),

                             ('techootro', '=1 if predominant material on the roof is other'),

                             ('cielorazo', '=1 if the house has ceiling'),

                             ('abastaguadentro', '=1 if water provision inside the dwelling'),

                             ('abastaguafuera', '=1 if water provision outside the dwelling'),

                             ('abastaguano', '=1 if no water provision'),

                             ('public', '"=1 electricity from CNFL,  ICE,  ESPH/JASEC"'),

                             ('planpri', '=1 electricity from private plant'),

                             ('noelec', '=1 no electricity in the dwelling'),

                             ('coopele', '=1 electricity from cooperative'),

                             ('sanitario1', '=1 no toilet in the dwelling'),

                             ('sanitario2', '=1 toilet connected to sewer or cesspool'),

                             ('sanitario3', '=1 toilet connected to  septic tank'),

                             ('sanitario5', '=1 toilet connected to black hole or letrine'),

                             ('sanitario6', '=1 toilet connected to other system'),

                             ('energcocinar1', '=1 no main source of energy used for cooking (no kitchen)'),

                             ('energcocinar2', '=1 main source of energy used for cooking electricity'),

                             ('energcocinar3', '=1 main source of energy used for cooking gas'),

                             ('energcocinar4', '=1 main source of energy used for cooking wood charcoal'),

                             ('elimbasu1', '=1 if rubbish disposal mainly by tanker truck'),

                             ('elimbasu2', '=1 if rubbish disposal mainly by botan hollow or buried'),

                             ('elimbasu3', '=1 if rubbish disposal mainly by burning'),

                             ('elimbasu4', '=1 if rubbish disposal mainly by throwing in an unoccupied space'),

                             ('elimbasu5', '"=1 if rubbish disposal mainly by throwing in river,  creek or sea"'),

                             ('elimbasu6', '=1 if rubbish disposal mainly other'),

                             ('epared1', '=1 if walls are bad'),

                             ('epared2', '=1 if walls are regular'),

                             ('epared3', '=1 if walls are good'),

                             ('etecho1', '=1 if roof are bad'),

                             ('etecho2', '=1 if roof are regular'),

                             ('etecho3', '=1 if roof are good'),

                             ('eviv1', '=1 if floor are bad'),

                             ('eviv2', '=1 if floor are regular'),

                             ('eviv3', '=1 if floor are good'),

                             ('dis', '=1 if disable person'),

                             ('male', '=1 if male'),

                             ('female', '=1 if female'),

                             ('estadocivil1', '=1 if less than 10 years old'),

                             ('estadocivil2', '=1 if free or coupled uunion'),

                             ('estadocivil3', '=1 if married'),

                             ('estadocivil4', '=1 if divorced'),

                             ('estadocivil5', '=1 if separated'),

                             ('estadocivil6', '=1 if widow/er'),

                             ('estadocivil7', '=1 if single'),

                             ('parentesco1', '=1 if household head'),

                             ('parentesco2', '=1 if spouse/partner'),

                             ('parentesco3', '=1 if son/doughter'),

                             ('parentesco4', '=1 if stepson/doughter'),

                             ('parentesco5', '=1 if son/doughter in law'),

                             ('parentesco6', '=1 if grandson/doughter'),

                             ('parentesco7', '=1 if mother/father'),

                             ('parentesco8', '=1 if father/mother in law'),

                             ('parentesco9', '=1 if brother/sister'),

                             ('parentesco10', '=1 if brother/sister in law'),

                             ('parentesco11', '=1 if other family member'),

                             ('parentesco12', '=1 if other non family member'),

                             ('idhogar', 'Household level identifier'),

                             ('hogar_nin', 'Number of children 0 to 19 in household'),

                             ('hogar_adul', 'Number of adults in household'),

                             ('hogar_mayor', '# of individuals 65+ in the household'),

                             ('hogar_total', '# of total individuals in the household'),

                             ('dependency', 'Dependency rate, calculated = (number of members of the household younger than 19 or older than 64)/(number of member of household between 19 and 64)'),

                             ('edjefe', 'years of education of male head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0'),

                             ('edjefa', 'years of education of female head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0'),

                             ('meaneduc', 'average years of education for adults (18+)'),

                             ('instlevel1', '=1 no level of education'),

                             ('instlevel2', '=1 incomplete primary'),

                             ('instlevel3', '=1 complete primary'),

                             ('instlevel4', '=1 incomplete academic secondary level'),

                             ('instlevel5', '=1 complete academic secondary level'),

                             ('instlevel6', '=1 incomplete technical secondary level'),

                             ('instlevel7', '=1 complete technical secondary level'),

                             ('instlevel8', '=1 undergraduate and higher education'),

                             ('instlevel9', '=1 postgraduate higher education'),

                             ('bedrooms', 'number of bedrooms'),

                             ('overcrowding', '# persons per room'),

                             ('tipovivi1', '=1 own and fully paid house'),

                             ('tipovivi2', '"=1 own,  paying in installments"'),

                             ('tipovivi3', '=1 rented'),

                             ('tipovivi4', '=1 precarious'),

                             ('tipovivi5', '"=1 other(assigned,  borrowed)"'),

                             ('computer', '=1 if the household has notebook or desktop computer'),

                             ('television', '=1 if the household has TV'),

                             ('mobilephone', '=1 if mobile phone'),

                             ('qmobilephone', '# of mobile phones'),

                             ('lugar1', '=1 region Central'),

                             ('lugar2', '=1 region Chorotega'),

                             ('lugar3', '=1 region PacÃƒÂ­fico central'),

                             ('lugar4', '=1 region Brunca'),

                             ('lugar5', '=1 region Huetar AtlÃƒÂ¡ntica'),

                             ('lugar6', '=1 region Huetar Norte'),

                             ('area1', '=1 zona urbana'),

                             ('area2', '=2 zona rural'),

                             ('age', 'Age in years'),

                             ('SQBescolari', 'escolari squared'),

                             ('SQBage', 'age squared'),

                             ('SQBhogar_total', 'hogar_total squared'),

                             ('SQBedjefe', 'edjefe squared'),

                             ('SQBhogar_nin', 'hogar_nin squared'),

                             ('SQBovercrowding', 'overcrowding squared'),

                             ('SQBdependency', 'dependency squared'),

                             ('SQBmeaned', 'square of the mean years of education of adults (>=18) in the household'),

                             ('agesq', 'Age squared')]



all_features = [feature for feature, description in features_and_descriptions]





household_features_and_descriptions = [('v2a1', 'Monthly rent payment'),

                             ('hacdor', '=1 Overcrowding by bedrooms'),

                             ('rooms', 'number of all rooms in the house'),

                             ('hacapo', '=1 Overcrowding by rooms'),

                             ('v14a', '=1 has bathroom in the household'),

                             ('refrig', '=1 if the household has refrigerator'),

                             ('v18q1', 'number of tablets household owns'),

                             ('r4h1', 'Males younger than 12 years of age'),

                             ('r4h2', 'Males 12 years of age and older'),

                             ('r4h3', 'Total males in the household'),

                             ('r4m1', 'Females younger than 12 years of age'),

                             ('r4m2', 'Females 12 years of age and older'),

                             ('r4m3', 'Total females in the household'),

                             ('r4t1', 'persons younger than 12 years of age'),

                             ('r4t2', 'persons 12 years of age and older'),

                             ('r4t3', 'Total persons in the household'),

                             ('tamhog', 'size of the household'),

                             ('tamviv', 'number of persons living in the household'),

                             ('hhsize', 'household size'),

                             ('paredblolad', '=1 if predominant material on the outside wall is block or brick'),

                             ('paredzocalo', '"=1 if predominant material on the outside wall is socket (wood,  zinc or absbesto"'),

                             ('paredpreb', '=1 if predominant material on the outside wall is prefabricated or cement'),

                             ('pareddes', '=1 if predominant material on the outside wall is waste material'),

                             ('paredmad', '=1 if predominant material on the outside wall is wood'),

                             ('paredzinc', '=1 if predominant material on the outside wall is zink'),

                             ('paredfibras', '=1 if predominant material on the outside wall is natural fibers'),

                             ('paredother', '=1 if predominant material on the outside wall is other'),

                             ('pisomoscer', '"=1 if predominant material on the floor is mosaic,  ceramic,  terrazo"'),

                             ('pisocemento', '=1 if predominant material on the floor is cement'),

                             ('pisoother', '=1 if predominant material on the floor is other'),

                             ('pisonatur', '=1 if predominant material on the floor is  natural material'),

                             ('pisonotiene', '=1 if no floor at the household'),

                             ('pisomadera', '=1 if predominant material on the floor is wood'),

                             ('techozinc', '=1 if predominant material on the roof is metal foil or zink'),

                             ('techoentrepiso', '"=1 if predominant material on the roof is fiber cement,  mezzanine "'),

                             ('techocane', '=1 if predominant material on the roof is natural fibers'),

                             ('techootro', '=1 if predominant material on the roof is other'),

                             ('cielorazo', '=1 if the house has ceiling'),

                             ('abastaguadentro', '=1 if water provision inside the dwelling'),

                             ('abastaguafuera', '=1 if water provision outside the dwelling'),

                             ('abastaguano', '=1 if no water provision'),

                             ('public', '"=1 electricity from CNFL,  ICE,  ESPH/JASEC"'),

                             ('planpri', '=1 electricity from private plant'),

                             ('noelec', '=1 no electricity in the dwelling'),

                             ('coopele', '=1 electricity from cooperative'),

                             ('sanitario1', '=1 no toilet in the dwelling'),

                             ('sanitario2', '=1 toilet connected to sewer or cesspool'),

                             ('sanitario3', '=1 toilet connected to  septic tank'),

                             ('sanitario5', '=1 toilet connected to black hole or letrine'),

                             ('sanitario6', '=1 toilet connected to other system'),

                             ('energcocinar1', '=1 no main source of energy used for cooking (no kitchen)'),

                             ('energcocinar2', '=1 main source of energy used for cooking electricity'),

                             ('energcocinar3', '=1 main source of energy used for cooking gas'),

                             ('energcocinar4', '=1 main source of energy used for cooking wood charcoal'),

                             ('elimbasu1', '=1 if rubbish disposal mainly by tanker truck'),

                             ('elimbasu2', '=1 if rubbish disposal mainly by botan hollow or buried'),

                             ('elimbasu3', '=1 if rubbish disposal mainly by burning'),

                             ('elimbasu4', '=1 if rubbish disposal mainly by throwing in an unoccupied space'),

                             ('elimbasu5', '"=1 if rubbish disposal mainly by throwing in river,  creek or sea"'),

                             ('elimbasu6', '=1 if rubbish disposal mainly other'),

                             ('epared1', '=1 if walls are bad'),

                             ('epared2', '=1 if walls are regular'),

                             ('epared3', '=1 if walls are good'),

                             ('etecho1', '=1 if roof are bad'),

                             ('etecho2', '=1 if roof are regular'),

                             ('etecho3', '=1 if roof are good'),

                             ('eviv1', '=1 if floor are bad'),

                             ('eviv2', '=1 if floor are regular'),

                             ('eviv3', '=1 if floor are good'),

                             ('hogar_nin', 'Number of children 0 to 19 in household'),

                             ('hogar_adul', 'Number of adults in household'),

                             ('hogar_mayor', '# of individuals 65+ in the household'),

                             ('hogar_total', '# of total individuals in the household'),

                             ('dependency', 'Dependency rate, calculated = (number of members of the household younger than 19 or older than 64)/(number of member of household between 19 and 64)'),

                             ('edjefe', 'years of education of male head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0'),

                             ('edjefa', 'years of education of female head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0'),

                             ('meaneduc', 'average years of education for adults (18+)'),

                             ('bedrooms', 'number of bedrooms'),

                             ('overcrowding', '# persons per room'),

                             ('tipovivi1', '=1 own and fully paid house'),

                             ('tipovivi2', '"=1 own,  paying in installments"'),

                             ('tipovivi3', '=1 rented'),

                             ('tipovivi4', '=1 precarious'),

                             ('tipovivi5', '"=1 other(assigned,  borrowed)"'),

                             ('computer', '=1 if the household has notebook or desktop computer'),

                             ('television', '=1 if the household has TV'),

                             ('mobilephone', '=1 if mobile phone'),

                             ('qmobilephone', '# of mobile phones'),

                             ('lugar1', '=1 region Central'),

                             ('lugar2', '=1 region Chorotega'),

                             ('lugar3', '=1 region PacÃƒÂ­fico central'),

                             ('lugar4', '=1 region Brunca'),

                             ('lugar5', '=1 region Huetar AtlÃƒÂ¡ntica'),

                             ('lugar6', '=1 region Huetar Norte'),

                             ('area1', '=1 zona urbana'),

                             ('area2', '=2 zona rural')]



household_features = [feature for feature, description in household_features_and_descriptions]



individual_features_and_descriptions = [('v18q', 'owns a tablet'),

                             ('escolari', 'years of schooling'),

                             ('rez_esc', 'Years behind in school'),

                             ('dis', '=1 if disable person'),

                             ('male', '=1 if male'),

                             ('female', '=1 if female'),

                             ('estadocivil1', '=1 if less than 10 years old'),

                             ('estadocivil2', '=1 if free or coupled uunion'),

                             ('estadocivil3', '=1 if married'),

                             ('estadocivil4', '=1 if divorced'),

                             ('estadocivil5', '=1 if separated'),

                             ('estadocivil6', '=1 if widow/er'),

                             ('estadocivil7', '=1 if single'),

                             ('parentesco1', '=1 if household head'),

                             ('parentesco2', '=1 if spouse/partner'),

                             ('parentesco3', '=1 if son/doughter'),

                             ('parentesco4', '=1 if stepson/doughter'),

                             ('parentesco5', '=1 if son/doughter in law'),

                             ('parentesco6', '=1 if grandson/doughter'),

                             ('parentesco7', '=1 if mother/father'),

                             ('parentesco8', '=1 if father/mother in law'),

                             ('parentesco9', '=1 if brother/sister'),

                             ('parentesco10', '=1 if brother/sister in law'),

                             ('parentesco11', '=1 if other family member'),

                             ('parentesco12', '=1 if other non family member'),

                             ('instlevel1', '=1 no level of education'),

                             ('instlevel2', '=1 incomplete primary'),

                             ('instlevel3', '=1 complete primary'),

                             ('instlevel4', '=1 incomplete academic secondary level'),

                             ('instlevel5', '=1 complete academic secondary level'),

                             ('instlevel6', '=1 incomplete technical secondary level'),

                             ('instlevel7', '=1 complete technical secondary level'),

                             ('instlevel8', '=1 undergraduate and higher education'),

                             ('instlevel9', '=1 postgraduate higher education'),

                             ('age', 'Age in years')]



individual_features = [feature for feature, description in individual_features_and_descriptions]



squared_features_and_descriptions = [('SQBescolari', 'escolari squared'),

                             ('SQBage', 'age squared'),

                             ('SQBhogar_total', 'hogar_total squared'),

                             ('SQBedjefe', 'edjefe squared'),

                             ('SQBhogar_nin', 'hogar_nin squared'),

                             ('SQBovercrowding', 'overcrowding squared'),

                             ('SQBdependency', 'dependency squared'),

                             ('SQBmeaned', 'square of the mean years of education of adults (>=18) in the household'),

                             ('agesq', 'Age squared')]



squared_features = [feature for feature, description in squared_features_and_descriptions]



squared_household_features_and_descriptions = [('SQBhogar_total', 'hogar_total squared'),

                             ('SQBedjefe', 'edjefe squared'),

                             ('SQBhogar_nin', 'hogar_nin squared'),

                             ('SQBovercrowding', 'overcrowding squared'),

                             ('SQBdependency', 'dependency squared'),

                             ('SQBmeaned', 'square of the mean years of education of adults (>=18) in the household')]



squared_household_features = [feature for feature, description in squared_household_features_and_descriptions]



squared_individual_features_and_descriptions = [('SQBescolari', 'escolari squared'),

                             ('SQBage', 'age squared'),

                             ('agesq', 'Age squared')]



squared_individual_features = [feature for feature, description in squared_individual_features_and_descriptions]
# Verify that seemingly duplicate attributes, SQBage and agesq, 

# are in fact duplicates and contain duplicate values in both train and test sets.



for df, name in [(train, 'train'), (test, 'test')]:

    assert df.agesq.equals(df.SQBage), f"agesq is not equivalent with SQBage in the {name} set"

    

# Remove duplicate column agesq from feature lists

lists_with_agesq = [features_and_descriptions, squared_features_and_descriptions, squared_individual_features_and_descriptions]

for feature_list in lists_with_agesq:

    try:

        feature_list.remove(('agesq', 'Age squared'))

    except:

        continue



# Verify we deleted them all

for feature_list in lists_with_agesq:

    assert 'agesq' not in {x for x, y in feature_list}, 'Duplicated column agesq is still in feature list.'
def print_nan_counts(df):

    nan_counts = df.isna().sum()

    print(nan_counts[nan_counts > 0].sort_values(ascending = False))

    

print_nan_counts(train)
print('Stats for potentially related characteristics for individuals where rez_esc isna.')

print(train[train.rez_esc.isna()][['age', 'escolari', 'meaneduc']].describe())



print('')

print('Stats for potentially related characteristics for individuals where rez_esc is not null.')

print(train[train.rez_esc.notna()][['age', 'escolari', 'meaneduc']].describe())
class ZerofillRezEscOutOfBounds(BaseEstimator, TransformerMixin):

    """Zerofill rez_esc for any row where age is < 7 or age is > 19."""

    def __init__(self):

        pass

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        X.loc[X.rez_esc.isna() & ((X.age < 7) | (X.age > 19)), 'rez_esc'] = 0

        return X

        

    

zerofiller = ZerofillRezEscOutOfBounds()

train_zerofilled = zerofiller.transform(train.copy())



assert not train_zerofilled[(train_zerofilled.age < 7) | (train_zerofilled.age > 19)].rez_esc.isna().any(), 'There are individuals younger than 7 or older than 19 with a na value for rez_esc.'

print(f"There are {train_zerofilled.rez_esc.isna().sum()} individuals with a na value for rez_esc.")

print(f"There are {train_zerofilled[(train_zerofilled.age < 7) | (train_zerofilled.age > 19)].rez_esc.isna().sum()} individuals younger than 7 or older than 19 with a na value for rez_esc.")
train.loc[train.rez_esc > 5, id_features + individual_features].head()
test.loc[test.rez_esc > 5, id_features + individual_features].head()
class ZeroMaxRezEsc(BaseEstimator, TransformerMixin):

    """Zero out rez_esc for any row where rez_esc is > the prescribed max of 5."""

    def __init__(self):

        pass

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        X.loc[X.rez_esc > 5, 'rez_esc'] = 0

        return X

        

    

zero_rez_esc = ZeroMaxRezEsc()

test_zero_rez_esc = zero_rez_esc.transform(test.copy())



assert len(test_zero_rez_esc[test_zero_rez_esc.rez_esc > 5]) == 0, 'There are individuals with greater than 5 rez_esc'
train.loc[train.v18q1 > 1, ['Id', 'idhogar', 'parentesco1', 'v18q', 'v18q1']].head(20)
train.v18q.value_counts()
print(train.v18q1.value_counts())

print(len(train.loc[(train.v18q == 0 & train.v18q1.isna())]))
class ZerofillV18q1ForFalseV18Q(BaseEstimator, TransformerMixin):

    """Zerofill v18q1 for any row where v18q is 0."""

    def __init__(self):

        pass

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        X.loc[X.v18q == 0, 'v18q1'] = 0

        return X

        

    

zerofiller = ZerofillV18q1ForFalseV18Q()

train_zerofilled = zerofiller.transform(train.copy())



assert not train_zerofilled.v18q1.isna().any(), 'There are individuals with na value for v18q1.'
for v in ['tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5']:

    print(f"Value counts for {v} for individuals with missing v2a1:")

    print(train.loc[train.v2a1.isna(), v].value_counts())

    print('')
class ZerofillV2a1(BaseEstimator, TransformerMixin):

    """Zerofill v2a1 if tipovivi1 or tipovivi5 is 1."""

    def __init__(self):

        pass

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        X.loc[(X.v2a1.isna()) & ((X.tipovivi1 == 1) | (X.tipovivi5 == 1)), 'v2a1'] = 0

        return X

        

    

zerofiller = ZerofillV2a1()

train_zerofilled = zerofiller.transform(train.copy())



# Assert that everyone that isn't a tipovivi4 doesn't have a NaN value for v2a1

assert not train_zerofilled[train_zerofilled.tipovivi4 == 0].v2a1.isna().any(), 'There are individuals with na value for v2a1.'
nan_pipeline = make_pipeline(ZerofillRezEscOutOfBounds(), ZerofillV18q1ForFalseV18Q(), ZerofillV2a1())

train_nan = nan_pipeline.fit_transform(train.copy())



print_nan_counts(train_nan)
train.select_dtypes(include='object').head()
class TransformYesNoToNumeric(BaseEstimator, TransformerMixin):

    """Transform edjefe, edjefa, and dependencey yes/no values to numeric values.

    yes=1 and no=0."""

    def __init__(self):

        pass

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        replacements = {'yes': 1, 'no': 0}

        columns = [('edjefe', 'uint8'), 

                   ('edjefa', 'uint8'), 

                   ('dependency', 'float16')]

    

        for column, converted_type in columns:    

            if X[column].dtype == 'object':

                X[column] = X[column].replace(replacements).astype(converted_type)

        

        return X



yes_no_transformer = TransformYesNoToNumeric()

train_yes_no_transformed = yes_no_transformer.transform(train.copy())



# Assert that all columns aside from Id and idhogar are numeric

assert train_yes_no_transformed.select_dtypes(include='object').columns.values.tolist() == ['Id', 'idhogar'], 'There are columns aside from Id and idhogar that are type object.'
class AggregateIndividualFeatures(BaseEstimator, TransformerMixin):

    """Aggregate individual features per household by grouping them by idhogar and

    applying sum, min, max, std.

    

    New features will be added as {feature name}-{aggregation type}, e.g. age-std

    """

    def __init__(self):

        self.excluded_individual_features = ['v18q', 'male', 'female', 'Target']

        self.individual_features_to_aggregate = [feature for feature in individual_features if feature not in self.excluded_individual_features]

        self.aggregations = ['sum', 'min', 'max', 'std', 'mean', 'median']

        self.aggregated_features = []

            

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        if 'instlevel' in X.columns and 'instlevel' not in self.individual_features_to_aggregate:

            self.individual_features_to_aggregate.append('instlevel')

            

        aggregates = X[id_features + self.individual_features_to_aggregate].groupby("idhogar").agg(self.aggregations)

        aggregates.columns = ['-'.join(column).strip() for column in aggregates.columns.values]

        self.aggregated_features = aggregates.columns.values.tolist()

    

        return X.merge(aggregates, on='idhogar', how='left')



    

ind_aggregator = AggregateIndividualFeatures()

train_ind_aggregated = ind_aggregator.transform(train.copy())



# To be exhaustive, we could loop through all households, but this spost check should 

# give us a lot more confidence and be a lot faster.

household_id = '6893e65ca'



for aggregation in ind_aggregator.aggregations:

    manual = train.loc[train.idhogar == household_id].age.apply(aggregation)

    aggregated = train_ind_aggregated.loc[(train_ind_aggregated.idhogar == household_id), [f"age-{aggregation}"]].iloc[0].values[0]

    assert manual == aggregated, f"Calculated {aggregation} for age doesn't match aggregation for household {household_id}"

    print(f"Calculated {aggregation} for age, {aggregated}, matches manual aggregation, {manual}, for household {household_id}")
class Ordinalizer(BaseEstimator, TransformerMixin):

    """Add an ordered numeric attribute for previously broken out boolean attributes"""

    

    def __init__(self):

        pass

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        features = ['instlevel', 'epared', 'etecho', 'eviv']



        for feature in features:

            columns = [column for column in all_features if column.startswith(feature)]

            X[feature] = X.loc[:, columns].idxmax(1).apply(lambda x: columns.index(x) + 1)

            

            # Don't drop instleveli columns since we'll use them in household aggregation

            if feature != 'instlevel':

                X.drop(columns, axis=1, inplace=True)

                

        return X

    

    

# Tests

ordinalizer = Ordinalizer()

train_ordinalized = ordinalizer.transform(train.copy())

    

# Test 10 random rows

for index, individual in train_ordinalized.sample(10).iterrows():

    for feature in ['instlevel', 'epared', 'etecho', 'eviv']:

        assert train.loc[index, f"{feature}{individual[feature]}"] == 1, f"Ordinalized {feature} doesn't match original boolean instlevel"

                      

# Test dropped columns

for feature in ['epared', 'etecho', 'eviv']:

    feature_columns = [column for column in train_ordinalized.columns if column.startswith(feature)]

    assert len(feature_columns) == 1, f"{feature} column is still present."



# Test AggregateIndividualFeatures picks up instlevel for aggregation

household_id = "6893e65ca"



instlevel_ind_aggregator = AggregateIndividualFeatures()

train_instlevel_ind_aggregated = instlevel_ind_aggregator.transform(train_ordinalized)

                      

assert 'instlevel-sum' in instlevel_ind_aggregator.aggregated_features, "instlevel is not in aggregated features."

                      

for aggregation in instlevel_ind_aggregator.aggregations:

    manual = train_ordinalized.loc[train_ordinalized.idhogar == household_id].instlevel.apply(aggregation)

    aggregated = train_instlevel_ind_aggregated.loc[(train_instlevel_ind_aggregated.idhogar == household_id), [f"instlevel-{aggregation}"]].iloc[0].values[0]

    assert manual == aggregated, f"Calculated {aggregation} for instlevel doesn't match aggregation for household {household_id}"

    print(f"Calculated {aggregation} for instlevel, {aggregated}, matches manual aggregation, {manual}, for household {household_id}")
class FeatureCreator(BaseEstimator, TransformerMixin):

    """Adds additional features"""

    

    def __init__(self):

        self.created_features = []

    

    def fit(self, X, y=None):

        self.created_features = []

        

        self._add_feature('rent_per_room', lambda X: X.v2a1 / X.rooms)

        self._add_feature('rent_per_hhsize', lambda X: X.v2a1 / X.hhsize)

        self._add_feature('tablets_per_hhsize', lambda X: X.v18q1 / X.hhsize)

        self._add_feature('tablets_per_adult', lambda X: X.v18q1 / X.hogar_adul)

        self._add_feature('escolari-mean_to_age_mean', lambda X: X['escolari-mean'] / X['age-mean'])

        self._add_feature('rez_esc-mean_to_age_mean', lambda X: X['rez_esc-mean'] / X['age-mean'])

        self._add_feature('males_to_females', lambda X: X.r4h3 / X.r4m3)

        self._add_feature('under12_to_over12', lambda X: X.r4t1 / X.r4t2)

        

        return self

    

    def transform(self, X, y=None):

        for label, calculation in self.created_features:

            X[label] = calculation(X)

        

#         # Address NaNs and infinity probably on a per feature basis

# #         X.replace({'males_to_females': {np.inf: np.nan},

# #                    'under12_to_over12': {np.inf: np.nan}}, inplace=True)

        X.replace(np.inf, np.nan, inplace=True)

        

        return X

        

    def _add_feature(self, label, calculation):

        self.created_features.append((label, calculation))



        

# Tests

ind_aggregator = AggregateIndividualFeatures()

train_ind_aggregated = ind_aggregator.transform(train.copy())

feature_creator = FeatureCreator()

features_created = feature_creator.fit_transform(train_ind_aggregated.copy())



assert features_created.rent_per_room.equals(features_created.v2a1 / features_created.rooms), "rent_per_room wasn't created correctly."

assert len(features_created.columns) - len(train_ind_aggregated.columns) == 8, "Didn't create the expected number of new features"



features_created[['rent_per_room', 'v2a1', 'rooms']].head()

assert len(features_created.columns[features_created[features_created == np.inf].any()]) == 0, "inifinity values present"
household_groups = train.groupby('idhogar')



# Verify that all the rows in a household group have the same value for the household features

assert len(household_groups) == len(train.idhogar.unique()), "Length of household groups is not the same as the number of unique household ids."

assert not household_groups[["idhogar"] + household_features].var().any().any(), "Not all rows for a group have the same values for a household feature"
# Todo: I don't love that the separating of the target variable happens

# here. Seems like it should be an explicit step in the pipeline or that

# We should have a better way of getting the household targets. Probably 

# the latter.



class ExtractHouseholds(BaseEstimator, TransformerMixin):

    """Returns dataframe for households."""

    

    def __init__(self, individual_aggregator=None, feature_creator=None):

        self.individual_aggregator = individual_aggregator

        self.feature_creator = feature_creator

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        self.included_features = household_features + squared_household_features

        

        if self.individual_aggregator:

            self.included_features = self.included_features + self.individual_aggregator.aggregated_features

            

        if self.feature_creator:

            self.included_features = self.included_features + [feature for feature, _ in self.feature_creator.created_features]

            

        # Don't include features that have been dropped earlier in the pipeline

        self.included_features = [feature for feature in self.included_features if feature in X]

    

        return X.groupby('idhogar')[self.included_features].first()

 

# Tests

household_extractor = ExtractHouseholds()

train_households = household_extractor.transform(train.copy())

assert len(train_households) == len(train.idhogar.unique()), f"Extracted households length ({len(train_households)}) doesn't match unique household indentifiers in train set ({len(train.idhogar.unique())})."



household_extractor = ExtractHouseholds(individual_aggregator=instlevel_ind_aggregator)

train_households = household_extractor.transform(train_instlevel_ind_aggregated.copy())

assert len(train_households) == len(train_instlevel_ind_aggregated.idhogar.unique()), f"Extracted households length ({len(train_households)}) doesn't match unique household indentifiers in train set ({len(train_instlevel_ind_aggregated.idhogar.unique())})."



train_households.head()
import scipy



# Transformers

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler, StandardScaler



# Cross validation

from sklearn.model_selection import cross_validate,  RandomizedSearchCV

from sklearn.metrics.scorer import make_scorer

from sklearn.metrics import f1_score



# Models

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.neural_network import MLPClassifier
class FeatureSelector(BaseEstimator, TransformerMixin):

    """Returns NumPy array corresponding to the selected features."""

    

    def __init__(self, selected_features=[], excluded_features=[]):

        self.selected_features = selected_features

        self.excluded_features = excluded_features

        

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        if not self.selected_features:

            self.selected_features = X.columns.tolist()

            

        if self.excluded_features:

            self.selected_features = [feature for feature in self.selected_features if feature not in self.excluded_features]



        return X[self.selected_features]
from sklearn.exceptions import NotFittedError



class ModelEvaluator():

    def __init__(self, pipeline, estimators, features=[], excluded_features=[]):

        self.pipeline = pipeline

        self.estimators = [self._init_estimator(estimator) for estimator in estimators]

        self._preprocessed = False

        self.features = features

        self.excluded_features = excluded_features

        

        # Ensure we're working with clean data and that none of our previous work

        # has leaked into the datasets.

        self.train, self.test = import_train_test()

        self.results = []

        

        

    def _init_estimator(self, estimator):

        return {'estimator': estimator,

                'fit': False,

                'cv_results': {},

                'feature_importances': None,

                'tuned': False,

                'tuned_best_estimator': None,

                'tuned_best_score': None,

                'tuned_best_params': None,

                'tuned_cv_results': {}}

        

    def _preprocess_data(self):

        try:

            self.pipeline.named_steps['featureselector'].selected_features = self.features

            self.pipeline.named_steps['featureselector'].excluded_features = self.excluded_features

        except KeyError:

            pass

        

        self._trainX_preprocessed = pipeline.fit_transform(self.train.copy())



        # Since X is now households, we need the corresponding targets to those households

        self._trainY_preprocessed = self.train.copy().groupby('idhogar')['Target'].first()

        

        self._testX_preprocessed = self.pipeline.transform(self.test.copy())





        self._preprocessed = True

        

        

    def evaluate(self):

        if not self._preprocessed:

            self._preprocess_data()

            

        print(f"Evaluating on {self._trainX_preprocessed.shape[1]} features and {self._trainX_preprocessed.shape[0]} samples.")

        print("")

        

        for estimator in self.estimators:

            estimator['cv_results'] = self._cross_validate(estimator)

            estimator['feature_importances'] = self._calculate_feature_importances(estimator)

            

        self._print_cv_results()



        

    def _cross_validate(self, estimator):

        return cross_validate(estimator['estimator'], 

                              self._trainX_preprocessed, 

                              self._trainY_preprocessed, 

                              cv=5, 

                              scoring='f1_macro',

                              n_jobs=-1,

                              return_train_score=False)

        

        

    def _fit_estimator(self, estimator):

        estimator['estimator'].fit(self._trainX_preprocessed, self._trainY_preprocessed)

        estimator['fit'] = True



        

    def _calculate_feature_importances(self, estimator):

        if not estimator['fit']:

            self._fit_estimator(estimator)

            

        try:

            return self._feature_importances_dataframe(estimator['estimator'].feature_importances_)

        except AttributeError:

            return None

            

        

    def _feature_importances_dataframe(self, feature_importances):

        # Get the features we actually trained on

        # Todo - consider moving this to preprocessing step

        trained_features = self.pipeline.named_steps['featureselector'].selected_features



        return pd.DataFrame(data={'importance': feature_importances}, 

                            index=trained_features).sort_values(by='importance', ascending=False)

    

    def _print_cv_results(self):

        for estimator in self.estimators:

            print(f"Scores for {estimator['estimator']}")

            print(f"Mean Macro F1 Score: {estimator['cv_results']['test_score'].mean()}, SD={estimator['cv_results']['test_score'].std()}")

            print("")

                  

                  

    def tune_hyperparameters(self, estimator, param_dist, n_iter=20, cv=5):

        random_search = RandomizedSearchCV(estimator['estimator'], 

                                           param_distributions=param_dist,

                                           n_iter=n_iter, 

                                           cv=cv, 

                                           scoring='f1_macro',

                                           n_jobs=-1,

                                           verbose=1)

                  

        random_search.fit(self._trainX_preprocessed, self._trainY_preprocessed)

                  

        estimator['tuned'] = True

        estimator['tuned_best_estimator'] = random_search.best_estimator_

        estimator['tuned_best_score'] = random_search.best_score_

        estimator['tuned_cv_results'] = random_search.cv_results_

        estimator['tuned_best_params'] = random_search.best_params_

        estimator['tuned_best_index'] = random_search.best_index_

                  

        return random_search

                  

        

    def prepare_submissions(self):

        """Prepare a submission csv for every tuned estimator"""

        for estimator in self.estimators:

            if not estimator['tuned']:

                continue

            

            self._save_submission(estimator)

    

                  

    def _save_submission(self, estimator):

        if not estimator['tuned_best_estimator']:

            pass

                  

        # Make predictions on test set

        predictions = self.test.copy().groupby('idhogar').first().reset_index()

        predictions['Target'] = estimator['tuned_best_estimator'].predict(self._testX_preprocessed)



        # # Merge household predictions back to individuals in test set

        test_results = test[['Id', 'idhogar']].copy()

        test_results = test_results.merge(predictions[['idhogar', 'Target']].copy(), on="idhogar", how="left").drop("idhogar", axis=1)



        assert test_results.shape[0] == self.test.shape[0], "Number of results don't match number of test samples."



        filename = f"{estimator['tuned_best_estimator'].__class__.__name__}-tuned-{estimator['tuned_best_score']:.3f}.csv"

        self._save_predictions(test_results, filename)

        

                  

    def _save_predictions(self, predictions, filename):

        """Write results to csv file."""

        predictions.to_csv(filename, index=False)

        print(f"Wrote results to {filename}")
# store this so we can pass it into household extractor

individual_aggregator = AggregateIndividualFeatures()

feature_creator = FeatureCreator()



# Create pipeline

pipeline = make_pipeline(ZerofillRezEscOutOfBounds(),

                         ZeroMaxRezEsc(),

                         ZerofillV18q1ForFalseV18Q(), 

                         ZerofillV2a1(),

                         TransformYesNoToNumeric(),

                         Ordinalizer(), # Make sure this happens before individual aggregation

                         individual_aggregator,

                         feature_creator,

                         ExtractHouseholds(individual_aggregator=individual_aggregator, 

                                           feature_creator=feature_creator),

                         FeatureSelector(),

                         SimpleImputer(),

                         MinMaxScaler()

                        )



# Estimators we want to evaluate

estimators = [DecisionTreeClassifier(random_state=42),

              KNeighborsClassifier(),

              LogisticRegression(random_state=42, solver='lbfgs', multi_class='multinomial', max_iter=1000),

              RandomForestClassifier(random_state=42, n_estimators=100),

              AdaBoostClassifier(random_state=42),

              GradientBoostingClassifier(random_state=42),

              SGDClassifier(random_state=42, max_iter=1000, tol=1e-3),

              MLPClassifier(random_state=42, solver='lbfgs')]



evaluator = ModelEvaluator(pipeline, estimators)

evaluator.evaluate()
# Build a pipeline that performs everything except imputation and scaling

pipeline = make_pipeline(ZerofillRezEscOutOfBounds(),

                         ZeroMaxRezEsc(),

                         ZerofillV18q1ForFalseV18Q(), 

                         ZerofillV2a1(),

                         TransformYesNoToNumeric(),

                         Ordinalizer(), # Make sure this happens before individual aggregation

                         individual_aggregator,

                         feature_creator,

                         ExtractHouseholds(individual_aggregator=individual_aggregator, 

                                           feature_creator=feature_creator)

                        )



evaluator_corr = ModelEvaluator(pipeline, estimators)

evaluator_corr._preprocess_data()



corr = evaluator_corr._trainX_preprocessed.corr()
print(evaluator_corr.train.columns)

print(evaluator_corr._trainX_preprocessed.columns)
correlated_features = corr[corr.abs().gt(0.95)].count() > 1

corr.loc[correlated_features, correlated_features]



plt.figure(figsize=(25,25))

sns.heatmap(corr.loc[correlated_features, correlated_features])
for attribute in corr.loc[correlated_features, correlated_features]:

    print(corr.loc[attribute, corr.loc[attribute].abs().gt(0.95)])

    print()
variance = len(train.loc[train[['r4t3', 'tamhog', 'hhsize', 'hogar_total']].var(axis=1) != 0, ['r4t3', 'tamhog', 'hhsize', 'hogar_total']])

print(f"Variance in train between 'r4t3', 'tamhog', 'hhsize', 'hogar_total': {variance}")

      

variance = len(test.loc[test[['r4t3', 'tamhog', 'hhsize', 'hogar_total']].var(axis=1) != 0, ['r4t3', 'tamhog', 'hhsize', 'hogar_total']])

print(f"Variance in test between 'r4t3', 'tamhog', 'hhsize', 'hogar_total': {variance}")



variance = len(train.loc[train[['tamhog', 'hhsize', 'hogar_total']].var(axis=1) != 0, ['tamhog', 'hhsize', 'hogar_total']])

print(f"Variance in train between 'tamhog', 'hhsize', 'hogar_total': {variance}")



variance = len(test.loc[test[['tamhog', 'hhsize', 'hogar_total']].var(axis=1) != 0, ['tamhog', 'hhsize', 'hogar_total']])

print(f"Variance in test between 'tamhog', 'hhsize', 'hogar_total': {variance}")

      

train.loc[train[['r4t3', 'tamhog', 'hhsize', 'hogar_total']].var(axis=1) != 0, ['r4t3', 'tamhog', 'hhsize', 'hogar_total','Target']].head(10)
print(train[['tamviv', 'r4t3', 'tamhog', 'hhsize', 'hogar_total']].corr())

print(test[['tamviv', 'r4t3', 'tamhog', 'hhsize', 'hogar_total']].corr())
print(train.loc[(train.tamviv != train.r4t3), ['tamviv', 'r4t3', 'tamhog', 'hhsize', 'hogar_total', 'Target']].head())

print(train.loc[(train.tamviv > train.r4t3), ['tamviv', 'r4t3', 'tamhog', 'hhsize', 'hogar_total', 'Target']].head())



equality = train.loc[(train.tamviv != train.r4t3)].equals(train.loc[(train.tamviv > train.r4t3)])

print(f"When tamviv isn't equal to r4t3, tamviv is greater than tamviv, train: {equality}")



equality = test.loc[(test.tamviv != test.r4t3)].equals(test.loc[(test.tamviv > test.r4t3)])

print(f"When tamviv isn't equal to r4t3, tamviv is greater than tamviv, test: {equality}")



# Whenever r4t3 is different than the hhsize attributes, is tamviv always equal to r4t3?

print(len(train.loc[(train.r4t3 != train.hhsize) & (train.tamviv != train.r4t3), ['tamviv', 'r4t3', 'tamhog', 'hhsize', 'hogar_total', 'Target']]))
class TamvivR4t3Combined(BaseEstimator, TransformerMixin):

    """Adds a new attribute with the max of tamviv and r4t3"""

    

    def __init__(self):

        self.created_features = []

        

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        X['tamviv_r4t3_combined'] = X[['r4t3', 'tamviv']].max(axis=1)

        

        self.created_features = ['tamviv_r4t3_combined']



        return X

    

    

tamviv_r4t3_combiner = TamvivR4t3Combined()

tamviv_r4t3_preprocessed = tamviv_r4t3_combiner.transform(train.copy())



max_check = tamviv_r4t3_preprocessed.tamviv_r4t3_combined.equals(tamviv_r4t3_preprocessed[['r4t3', 'tamviv']].max(axis=1))

assert max_check, "TamvivR4t3Combiner didn't create the correct max column"
# store this so we can pass it into household extractor

individual_aggregator = AggregateIndividualFeatures()

feature_creator = FeatureCreator()



# Create pipeline

pipeline = make_pipeline(ZerofillRezEscOutOfBounds(),

                         ZeroMaxRezEsc(),

                         ZerofillV18q1ForFalseV18Q(), 

                         ZerofillV2a1(),

                         TransformYesNoToNumeric(),

                         Ordinalizer(), # Make sure this happens before individual aggregation

                         individual_aggregator,

                         feature_creator,

                         ExtractHouseholds(individual_aggregator=individual_aggregator, 

                                           feature_creator=feature_creator),

                         TamvivR4t3Combined(),

                         FeatureSelector(),

                         SimpleImputer(),

                         MinMaxScaler()

                        )



correlated_features_list = correlated_features[correlated_features == True].index



# sum_features = [feature for feature in correlated_features_list if feature[-4:] == '-sum']

# max_features = [feature for feature in correlated_features_list if feature[-4:] == '-max']

# std_features = [feature for feature in correlated_features_list if feature[-4:] == '-std']



excluded_features = ['elimbasu5', 'area2', 'r4t3', 'tamviv', 'coopele', 'SQBdependency', 'hogar_total', 'tamhog'] #+ squared_features + sum_features + max_features

kept_features = ['area1', 'public', 'hhsize', 'dependency']



# Exclude the aggregation features more selectively than excluding them all

for feature in correlated_features_list:

    if feature not in excluded_features + kept_features:

        for excluded_feature in corr.loc[feature, corr.loc[feature].abs().between(0.95, 0.9999999)].index:

            excluded_features.append(excluded_feature)

            

        kept_features.append(feature)

        

# print(f"Excluding features: {excluded_features}")



evaluator_excluded = ModelEvaluator(pipeline, estimators, excluded_features=excluded_features)

evaluator_excluded.evaluate()
def select_important_features(feature_importances, threshold=0.9):

    total_importance = 0

    important_features = []



    for feature, importance in feature_importances.iterrows():

        total_importance += importance[0]

        important_features.append(feature)



        if total_importance >= threshold:

            break

        

    return important_features
# Get important features from estimator

feature_importance_estimator = [estimator for estimator in evaluator_excluded.estimators if estimator['estimator'].__class__.__name__ == 'GradientBoostingClassifier'][0]

important_features = select_important_features(feature_importance_estimator['feature_importances'], threshold=0.95)



evaluator_important_features = ModelEvaluator(pipeline, estimators, important_features)

evaluator_important_features.evaluate()
estimators = [GradientBoostingClassifier(random_state=42)]

tuning_evaluator = ModelEvaluator(pipeline, estimators, important_features)

tuning_evaluator.evaluate()
param_dist = {'learning_rate': scipy.stats.truncnorm(-0.995, 2, loc=0.1, scale=0.05),

              'n_estimators': scipy.stats.randint(80, 1000),

              'max_depth': scipy.stats.randint(3, 8),

              'min_samples_split': scipy.stats.randint(2, 100),

              'min_samples_leaf': scipy.stats.randint(20, 60),

              'max_features': ['sqrt', None, 'log2'],

              'subsample': scipy.stats.uniform(loc=0.6, scale=0.4)}



results = tuning_evaluator.tune_hyperparameters(tuning_evaluator.estimators[0], param_dist, n_iter=20, cv=5)
print(results.best_params_)

print(results.best_score_)
# tuning_evaluator.estimators[0]['tuned_cv_results']
tuning_evaluator.prepare_submissions()