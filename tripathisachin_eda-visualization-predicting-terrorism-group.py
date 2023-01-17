

# To support both python 2 and python 3

from __future__ import division, print_function, unicode_literals



# Common imports

import numpy as np

import os



os.environ['KMP_DUPLICATE_LIB_OK']='True'



# to make this notebook's output stable across runs

np.random.seed(42)







# Where to save the figures

PROJECT_ROOT_DIR = "."

images = "images"



def save_fig(fig_id, tight_layout=True):

    path = os.path.join(PROJECT_ROOT_DIR, images, fig_id + ".png")

    print("Saving figure", fig_id)

    if tight_layout:

        plt.tight_layout()

    plt.savefig(path, format='png', dpi=300)
import pandas as pd



DATA_PATH = os.path.join("datasets", "terrorism")

#fn that reads csv from the defined location, so if csv updates it can take care of it

def load_data(data_path=DATA_PATH):

    csv_path = os.path.join(data_path, "dataset.csv")

    return pd.read_csv(csv_path,encoding="UTF-8")
terrorism=load_data()
#Renaming the columns to avoid any ambiguity

terrorism.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive','nperps':'terrorists','nkillus':'US_KILL'},inplace=True)
#feature engineering

terrorism=terrorism[['Year','Month','Day','Country','Region','city','latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive','terrorists']]

terrorism['casualities']=terrorism['Killed']+terrorism['Wounded']

g=terrorism.groupby('Group')



terrorism=g.filter(lambda x: len(x) > 10)
%matplotlib inline

import matplotlib.pyplot as plt

terrorism.hist(bins=50, figsize=(20,15))

save_fig("attribute_histogram_plots")

plt.show()
%matplotlib inline

import matplotlib.pyplot as plt

terrorism.plot(kind="scatter", x="latitude", y="longitude", alpha=0.4,

    s=terrorism['Killed'] ,label="casualities", figsize=(10,7),

    c='terrorists', cmap=plt.get_cmap("jet"), colorbar=True,

    sharex=False)

save_fig("Casualties")

plt.legend()
import seaborn as sns

f, ax = plt.subplots(figsize=(9, 6)) 

sns.barplot( y = terrorism['Group'].value_counts().head(10).index,

            x = terrorism['Group'].value_counts().head(10).values,

                palette="GnBu_d")

save_fig("Active_terrorists_grp")

ax.set_title('Most Active Terrorist Organizations' );
#function to display wordclouds

def wcloud_dsplay(x):

    plt.figure(figsize=[20,10])

    plt.imshow(x, interpolation='bilinear')

    plt.axis("off")

    plt.show()
motives=terrorism[['Motive']]



target=terrorism[['Target_type']]



types=terrorism[['AttackType']]



summary=terrorism[['Summary']]

from wordcloud import WordCloud

from PIL import Image

mask = np.array(Image.open('/Users/onion8/ml/images/terror.png'))

wc = WordCloud(background_color="white", max_words=100, mask=mask,

               contour_width=3, contour_color='Black')



# Generate a wordcloud

wc=wc.generate(str(motives))

wc.to_file("/Users/onion8//ml/images/terror_motives.jpg")

wcloud_dsplay(wc)
wc=wc.generate(str(target))

wc.to_file("/Users/onion8//ml/images/terror_target.jpg")

wcloud_dsplay(wc)
wc=wc.generate(str(types))

wc.to_file("/Users/onion8//ml/images/terror_type.jpg")

wcloud_dsplay(wc)
wc=wc.generate(str(summary))

wc.to_file("/Users/onion8//ml/images/terror_smry.jpg")

wcloud_dsplay(wc)
terrorism.describe()
#Imputing nan values of categorical data points by their mode(most frequent)



terrorism_cat=terrorism.select_dtypes(include=[np.object])

terrorism_cat = terrorism_cat.fillna(terrorism_cat.mode().iloc[0])



#Merging it again to the base df

cols_to_use = terrorism.columns.difference(terrorism_cat.columns)

terrorism=pd.merge(terrorism[cols_to_use], terrorism_cat,left_index=True, right_index=True, how='outer')

#Separating features and labels

terrorism_feat=terrorism.drop('Group',axis=1)

terrorism_labels=terrorism['Group']

#Separating numerical and categorical features so that they can be treated differently in data preparation for training

terrorism_num= terrorism_feat.select_dtypes(include=[np.number])

terrorism_cat=terrorism_feat.select_dtypes(include=[np.object])
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn_pandas import CategoricalImputer

from sklearn.preprocessing import OneHotEncoder



#numerical pipeline

num_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="median")),

        

        ('std_scaler', StandardScaler()),

    ])

#categorical pipeline

cat_pipeline= Pipeline([

        

        

        ('1-hot', OneHotEncoder()),

    ])
from sklearn.compose import ColumnTransformer

num_attribs = list(terrorism_num)

cat_attribs = list(terrorism_cat)



#Merging both pipeline

full_pipeline = ColumnTransformer([

        ("num", num_pipeline, num_attribs),

        ("cat", cat_pipeline, cat_attribs)

    ])



#preparing features  for training

terrorism_prepared = full_pipeline.fit_transform(terrorism_feat)
from sklearn.model_selection import train_test_split



#Splitting training and testing features and labels(using same random_state allows split at the same point)

train_set, test_set = train_test_split(terrorism_prepared, test_size=0.2, random_state=42)

train_labels,test_labels=train_test_split(terrorism_labels, test_size=0.2, random_state=42)
terror=train_set.copy()



terror_test=test_set.copy()
from imblearn.over_sampling import SMOTE

# smote for imbalanced data

smote = SMOTE(ratio='minority',random_state=42,kind='svm')

X_sm, y_sm = smote.fit_sample(terror, train_labels)
labels=[]

labels=filter(lambda a: a != 'Unknown', y_sm)
#assigning class_weight to the terrorist groups

class_weight={}

class_wt={}

class_un={}

for i in range(len(np.unique(y_sm))):

    class_wt[labels[i]]=20000

class_un['Unknown'] =1
def merge_two_dicts(x, y):

    z = x.copy()   # start with x's keys and values

    z.update(y)    # modifies z with y's keys and values & returns None

    return z
class_weight = merge_two_dicts(class_wt, class_un)
from sklearn.ensemble import RandomForestClassifier



forest_clf = RandomForestClassifier(n_estimators=10, random_state=42,class_weight=class_weight)
#training

forest_clf.fit(X_sm, y_sm)
print("Predictions:", forest_clf.predict((terror_test)[26:35]))
print("Labels:", list((test_labels)[26:35]))
prediction=forest_clf.predict((terror_test))

actual_labels=list((test_labels))
#Accuracy metrics

from sklearn.metrics import f1_score

#using 'micro' it gives global

f1_score(actual_labels, prediction, average="micro")
#using 'weighted' it gives label wise considering the weights 

f1_score(actual_labels, prediction, average="weighted")
import xgboost

from sklearn.metrics import mean_squared_error





if xgboost is not None:  

    xgb_clf = xgboost.XGBClassifier(random_state=42)

    xgb_clf.fit(X_sm, y_sm)

    y_pred = xgb_clf.predict(test_set)

    y_pred=np.where(y_pred =='Yes', 1, 0)

    val_error = mean_squared_error(actual_labels, y_pred)

    print("Validation MSE:", val_error)
f1_score(test_labels, y_pred, average="weighted")