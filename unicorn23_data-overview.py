%matplotlib inline
import matplotlib
import numpy as np
import pandas as pd
matplotlib.style.use('ggplot')
matplotlib.rcParams['figure.figsize'] = (10, 7)
from pandas.tools.plotting import scatter_matrix

df = pd.read_csv("../input/train.csv",index_col=0)
df = df.fillna('NA')
df_dog = df.loc[df['AnimalType'] == 'Dog']
df_cat = df.loc[df['AnimalType'] == 'Cat']
#df['AnimalType'].value_counts().plot(kind='bar',alpha=0.5)

df[['AnimalType','OutcomeType']].groupby(['OutcomeType','AnimalType']).size().unstack().plot.bar()

df[['AnimalType','OutcomeType']].groupby(['AnimalType','OutcomeType']).size().unstack().plot.bar()
df = pd.read_csv("../input/train.csv",index_col=0)
df = df.fillna('NA')
AgeuponOutcomemapping = {'year' : 360, 'month' : 30,'week': 7,'years' : 360,'months' : 360,'weeks': 7,'days':1,'day' : 1}

def change_AgeuponOutcome_to_days(value):
    if "weeks" in value:
        value = int(value.replace("weeks","")) * AgeuponOutcomemapping['weeks']
    elif "years" in value:
        value = int(value.replace("years","")) * AgeuponOutcomemapping['years']
    elif "months" in value:
        value = int(value.replace("months","")) * AgeuponOutcomemapping['months']
    elif "week" in value:
        value = int(value.replace("week","")) * AgeuponOutcomemapping['week']
    elif "year" in value:
        value = int(value.replace("year","")) * AgeuponOutcomemapping['year']
    elif "month" in value:
        value = int(value.replace("month","")) * AgeuponOutcomemapping['month']
    elif "days" in value:
        value = int(value.replace("days","")) * AgeuponOutcomemapping['days']
    elif "day" in value:
        value = int(value.replace("day","")) * AgeuponOutcomemapping['day']
    else:
        value = 0
    
    return int(value)
    
df['AgeuponOutcome'] = df['AgeuponOutcome'].dropna()

df['AgeuponOutcome'] = df['AgeuponOutcome'].apply(change_AgeuponOutcome_to_days)

df['AgeuponOutcome'].plot.hist(xlim=360)
df = pd.read_csv("../input/train.csv",index_col=0)
df = df.fillna('NA')
df = df.drop(['Name','DateTime','OutcomeSubtype','AgeuponOutcome','Color'],axis = 1)
df['OutcomeType'] = df['OutcomeType'].astype('category')
df['AnimalType'] = df['AnimalType'].astype('category')
df['SexuponOutcome'] = df['SexuponOutcome'].astype('category')
df['Breed'] = df['Breed'].astype('category')
cat_columns = df.select_dtypes(['category']).columns
df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
scatter_matrix(df,diagonal='kde')
df = pd.read_csv("../input/train.csv",index_col=0)
df = df.fillna('NA')
AgeuponOutcomemapping = {'year' : 1, 'month' : 12 ,'week': 54,'years' : 1,'months' : 12,'weeks': 54,'days':360,'day' : 360}
def change_AgeuponOutcome_to_years(value):
    if "weeks" in value:
        value = int(value.replace("weeks","")) / AgeuponOutcomemapping['weeks']
    elif "years" in value: 
        value = int(value.replace("years","")) / AgeuponOutcomemapping['years'] 
    elif "months" in value: 
        value = int(value.replace("months","")) / AgeuponOutcomemapping['months'] 
    elif "week" in value:
        value = int(value.replace("week","")) / AgeuponOutcomemapping['week'] 
    elif "year" in value:
        value = int(value.replace("year","")) / AgeuponOutcomemapping['year'] 
    elif "month" in value:
        value = int(value.replace("month","")) / AgeuponOutcomemapping['month'] 
    elif "days" in value:
        value = int(value.replace("days","")) / AgeuponOutcomemapping['days'] 
    elif "day" in value:
        value = int(value.replace("day","")) / AgeuponOutcomemapping['day'] 
    else:
        value = 0
    
    return int(value)
  
df['AgeuponOutcome'] = df['AgeuponOutcome'].dropna()
df['AgeuponOutcome'] = df['AgeuponOutcome'].apply(change_AgeuponOutcome_to_years)
df[['AnimalType','AgeuponOutcome','OutcomeType']].groupby(['AnimalType','AgeuponOutcome','OutcomeType']).size().unstack().plot(alpha=0.5,title="out come by year")

df = pd.read_csv("../input/train.csv",index_col=0)
df = df.fillna('NA')
df[['SexuponOutcome','OutcomeType']].groupby(['OutcomeType','SexuponOutcome']).size().unstack().plot.bar()
df = pd.read_csv("../input/train.csv",index_col=0)
df = df.fillna('NA')
df[['AnimalType','SexuponOutcome','OutcomeType']].groupby(['AnimalType','SexuponOutcome','OutcomeType']).size().unstack().unstack().plot.bar()
matplotlib.rcParams['figure.figsize'] = (15, 8)
df = pd.read_csv("../input/train.csv",index_col=0)
df = df.fillna('NAN')
df['AgeuponOutcome'] = df['AgeuponOutcome'].dropna()
df['AgeuponOutcome'] = df['AgeuponOutcome'].apply(change_AgeuponOutcome_to_years)
df[['OutcomeType','AgeuponOutcome','AnimalType']].boxplot(by=['AnimalType','OutcomeType'])