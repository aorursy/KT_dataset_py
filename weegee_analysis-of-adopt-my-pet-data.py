import pandas as pd

import numpy as np

import matplotlib.pylab as plt

import seaborn as sns 

import missingno as msn

import tqdm

import time



from wordcloud import WordCloud



from sklearn.preprocessing import OneHotEncoder

from sklearn.datasets import dump_svmlight_file

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD, NMF

from sklearn.metrics import cohen_kappa_score, make_scorer

from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split





import xgboost as xgb

import lightgbm as lgb

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import ExtraTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.svm import LinearSVC #(setting multi_class=”crammer_singer”)

from sklearn.linear_model import LogisticRegression, RidgeClassifier #(LogReg: setting multi_class = "multinomial")

from sklearn.ensemble import RandomForestClassifier



%pylab inline
train = pd.read_csv("../input/train/train.csv", index_col="PetID")

test = pd.read_csv("../input/test/test.csv", index_col="PetID")
display(train.columns)

display(test.columns)
train["IsTraining"] = 1

test["IsTraining"]  = 0



merged = pd.concat([train,test])



cat = ["Type", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3", "MaturitySize",

      "FurLength", "Vaccinated", "Dewormed", "Sterilized"]



for c in cat:

    merged[c] = merged[c].astype("category")
merged.dtypes
merged.describe(include='all')
merged.sample(5)
msn.matrix(merged, sort='ascending')
def percConvert(ser):

    return ser/float(ser[-1])
train.dtypes
type_ct = pd.crosstab(train["Type"],train["AdoptionSpeed"],margins=True).apply(percConvert, axis=1)

display(type_ct)
fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(15,5))

sns.countplot("Type", data=train, ax=ax[0])

ax[0].set_xticklabels(["Dog", "Cat"])

ax[0].set_title("Dogs Vs Cats")

ax[0].spines["top"].set_visible(False)

ax[0].spines["right"].set_visible(False)



sns.countplot("Type", hue="AdoptionSpeed", data=train, ax=ax[1])

ax[1].set_title("AdoptionSpeed by Type")

ax[1].set_xticklabels(["Dog", "Cat"])

ax[1].spines["top"].set_visible(False)

ax[1].spines["right"].set_visible(False)
main_count = train["AdoptionSpeed"].value_counts(normalize=True).sort_index()



def prepare_plot_dict(df, col, main_count):

    main_count = dict(main_count)

    plot_dict = {}

    for i in df[col].unique():

        val_count = dict(df.loc[df[col] == i, 'AdoptionSpeed'].value_counts().sort_index())



        for k, v in main_count.items():

            if k in val_count:

                plot_dict[val_count[k]] = ((val_count[k] / sum(i for i in val_count.values())) / 

                                           main_count[k]) * 100 - 100

                

            else:

                plot_dict[0] = 0

                

    return plot_dict

                

def make_count_plot(df, x, hue='AdoptionSpeed', title='', main_count=main_count, super_ax=0):

    if super_ax != 0:

        g = sns.countplot(x=x, data=df, hue=hue, ax=super_ax);

        super_ax.set_title(title);

    else:

        g = sns.countplot(x=x, data=df, hue=hue)

        plt.title("AdoptionSpeed {}".format(title));

    ax = g.axes

    

    plot_dict = prepare_plot_dict(df, x, main_count)

    

    for p in ax.patches:

        h = p.get_height() if str(p.get_height()) != 'nan' else 0 

        text = f"{plot_dict[h]:.0f}%" if plot_dict[h] < 0 else f"+{plot_dict[h]:.0f}%"

        ax.annotate(text, (p.get_x()+p.get_width()/2., h), ha='center', va='center', 

                    fontsize=11, color="green" if plot_dict[h] > 0 else "red", 

                    rotation=0, xytext=(0,10), textcoords="offset points")
plt.figure(figsize=(18,8))

make_count_plot(df=train, x='Type', title='by pet Type')
dict(train.loc[train['Type']== 1, 'AdoptionSpeed'].value_counts().sort_index())
plt.figure()

ax = plt.subplot()

plt.bar(range(5),type_ct.iloc[0,:-1],alpha=.5,color="green",label="Dog")

plt.bar(range(5),type_ct.iloc[1,:-1],alpha=.5,color="purple",label="Cat")

for idx,i in enumerate(type_ct.iloc[2,:-1]):

    if idx==0:

        plt.hlines(y=i,xmin=idx-.5, xmax=idx+.5, color="red", linestyle="dashed", label="MeanPerc")

    else:

        plt.hlines(y=i,xmin=idx-.5, xmax=idx+.5, color="red", linestyle="dashed")

        

plt.legend()

plt.title("% AdoptionSpeed by Type")

plt.ylabel("[%]")

plt.xlabel("AdoptionSpeed")

ax.spines["top"].set_visible(False)

ax.spines["right"].set_visible(False)
print("Age feature has {} unique values.".format(len(train["Age"].value_counts())))

print('-'*35)

print("Most common Ages:")

print(train["Age"].value_counts()[:5])

print('-'*35)

print("Least common Ages:")

print(train["Age"].value_counts()[-5:])



plt.figure()

ax=plt.subplot()

ax.spines["top"].set_visible(False)

ax.spines["right"].set_visible(False)

print('-'*35)



plt.title("Histogram of Age feature")

plt.xlabel("Age")

plt.ylabel("# Animal")

plt.hist(train["Age"].value_counts(), bins=30);
train["AgeBinned"] = pd.qcut(train["Age"],7)
age_ct = pd.crosstab(train["AgeBinned"],train["AdoptionSpeed"],margins=True).apply(percConvert, axis=1)

age_ct_dog = pd.crosstab(train["AgeBinned"][train["Type"]==1],train["AdoptionSpeed"][train["Type"]==1],margins=True).apply(percConvert, axis=1)

age_ct_cat = pd.crosstab(train["AgeBinned"][train["Type"]==2],train["AdoptionSpeed"][train["Type"]==2],margins=True).apply(percConvert, axis=1)

print("Influence of Age on AdoptionSpeed")

display(age_ct)

print('-'*50)

print("Influence of Age on AdoptionSpeed for dogs")

display(age_ct_dog)

print('-'*50)

print("Influence of Age on AdoptionSpeed for cats")

display(age_ct_cat)
plt.figure()

sns.catplot("AgeBinned",col="Type", hue="AdoptionSpeed" ,kind="count" ,data=train)

plt.figure(figsize=(36, 5))

ax = plt.subplot()



ax.plot(range(5), age_ct.iloc[0,:-1], c='#ffffb2',label="AgeGroup1")

ax.plot(range(5), age_ct.iloc[1,:-1], c='#fed976', label="AgeGroup2")

ax.plot(range(5), age_ct.iloc[2,:-1], c='#feb24c',label="AgeGroup3")

ax.plot(range(5), age_ct.iloc[3,:-1], c='#fd8d3c',label="AgeGroup4")

ax.plot(range(5), age_ct.iloc[4,:-1], c='#fc4e2a',label="AgeGroup5")

ax.plot(range(5), age_ct.iloc[5,:-1], c='#e31a1c',label="AgeGroup6")

ax.plot(range(5), age_ct.iloc[6,:-1], c='#b10026',label="AgeGroup7")

ax.legend(fontsize=18)

ax.spines["top"].set_visible(False)

ax.spines["right"].set_visible(False)

ax.set_xticklabels(["","0","","1","","2","","3","","4"]);





fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(18,5), sharey=True)



ax[0].plot(range(5), age_ct_dog.iloc[0,:-1], c='#f2f0f7',label="AgeGroup1")

ax[0].plot(range(5), age_ct_dog.iloc[1,:-1], c='#dadaeb', label="AgeGroup2")

ax[0].plot(range(5), age_ct_dog.iloc[2,:-1], c='#bcbddc',label="AgeGroup3")

ax[0].plot(range(5), age_ct_dog.iloc[3,:-1], c='#9e9ac8',label="AgeGroup4")

ax[0].plot(range(5), age_ct_dog.iloc[4,:-1], c='#807dba',label="AgeGroup5")

ax[0].plot(range(5), age_ct_dog.iloc[5,:-1], c='#6a51a3',label="AgeGroup6")

ax[0].plot(range(5), age_ct_dog.iloc[6,:-1], c='#4a1486',label="AgeGroup7")

ax[0].legend()

ax[0].spines["top"].set_visible(False)

ax[0].spines["right"].set_visible(False)

ax[0].set_xticklabels(["","0","","1","","2","","3","","4"]);



ax[1].plot(range(5), age_ct_cat.iloc[0,:-1], c='#edf8e9',label="AgeGroup1")

ax[1].plot(range(5), age_ct_cat.iloc[1,:-1], c='#c7e9c0', label="AgeGroup2")

ax[1].plot(range(5), age_ct_cat.iloc[2,:-1], c='#a1d99b',label="AgeGroup3")

ax[1].plot(range(5), age_ct_cat.iloc[3,:-1], c='#74c476',label="AgeGroup4")

ax[1].plot(range(5), age_ct_cat.iloc[4,:-1], c='#41ab5d',label="AgeGroup5")

ax[1].plot(range(5), age_ct_cat.iloc[5,:-1], c='#238b45',label="AgeGroup6")

ax[1].plot(range(5), age_ct_cat.iloc[6,:-1], c='#005a32',label="AgeGroup7")

ax[1].legend()

ax[1].spines["top"].set_visible(False)

ax[1].spines["right"].set_visible(False)

ax[1].set_xticklabels(["","0","","1","","2","","3","","4"]);
print("There are {} unique breeds in the dataset.".format(len(train["Breed1"].value_counts())))

print("-"*35)

print(train["Breed1"].value_counts()[:10])

print("-"*35)

print(train[["Breed1", "Breed2"]].sample(5))

print("-"*35)

print("{} % of the dogs are purebred".format(train["Breed2"][train["Breed2"] == 0].shape[0]/train.shape[0]*100))
train["PureBred"] = 0

train["PureBred"][(train["Breed2"] == train["Breed1"]) | 

                 (train["Breed2"] == 0)] = 1
breed_ct = pd.crosstab(train["PureBred"],train["AdoptionSpeed"],margins=True).apply(percConvert, axis=1)

print(breed_ct)
#sns.countplot(train["PureBred"])

sns.countplot("AdoptionSpeed", hue="PureBred", data=train)
train["Breed1"].value_counts()



breeds = pd.read_csv("../input/breed_labels.csv")



breeds.columns = ["Breed1", "Type", "BreedName1"]

breeds.drop("Type", 1, inplace=True)



x=dict(zip(breeds["Breed1"], breeds["BreedName1"]))
plt.figure(figsize=(10,10))

ax = plt.subplot(121)

catBreeds = (" ").join([str(i).replace(" ", "") for i in train.loc[train["Type"]==2, "Breed1"].map(x).values])

wordcloud = WordCloud(max_font_size=None, background_color="black", width=1200, height=1000).generate(catBreeds)

plt.imshow(wordcloud)

plt.title("Top cat breeds")

plt.axis('off')



ax = plt.subplot(122)

dogBreeds = (" ").join([str(i).replace(" ", "") for i in train.loc[train["Type"]==1, "Breed1"].map(x).values])

wordcloud = WordCloud(max_font_size=None, background_color="black", width=1200, height=1000).generate(dogBreeds)

plt.imshow(wordcloud)

plt.title("Top cat breeds")

plt.axis('off')

train.loc[train["Type"]==1, "Breed1"].map(x).values
mixed = []



mixed.append(breeds["Breed1"][(breeds["BreedName1"].str.contains("Hair")) & ((breeds["BreedName1"].str.contains("Long")) |

      (breeds["BreedName1"].str.contains("Medium")) | (breeds["BreedName1"].str.contains("Short")))].values)



mixed.append(breeds["Breed1"][breeds["BreedName1"].str.contains("Breed")].values)



mixed = [i for i in flatten(mixed)]
train["PureBred"] = 1

train["PureBred"][train["Breed2"] != 0] = 0

train["PureBred"][train["Breed1"].isin(mixed)] = 0
print("Purebred Dogs")

print(train.loc[train["Type"]==1, "PureBred"].value_counts())

print(train.loc[train["Type"]==1].shape[0]/838)



train.loc[train["Type"]==2, "PureBred"].value_counts()
breed_ct = pd.crosstab(train["PureBred"],train["AdoptionSpeed"],margins=True).apply(percConvert, axis=1)

print(breed_ct)
fig, ax = plt.subplots(nrows=1, ncols=3,figsize=(17,10))



make_count_plot(train,"PureBred", "AdoptionSpeed", 

                title="AdoptionSpeed by Breed",super_ax=ax[0])



make_count_plot(train.loc[train["Type"]==1],"PureBred", "AdoptionSpeed", 

                title="AdoptionSpeed for Dogs by Breed",super_ax=ax[1])



make_count_plot(train.loc[train["Type"]==2],"PureBred", "AdoptionSpeed", 

                title="AdoptionSpeed for Cats by Breed",super_ax=ax[2])



plt.figure(figsize=(10,10))

ax = plt.subplot(121)

catNames = (" ").join(train.loc[train["Type"]==2, "Name"].fillna('').values)

wordcloud = WordCloud(max_font_size=None, background_color="black", width=1200, height=1000).generate(catNames)

plt.imshow(wordcloud)

plt.title("Top cat names")

plt.axis('off')



ax2 = plt.subplot(122)

dogNames = (" ").join(train.loc[train["Type"]==1, "Name"].fillna('').values)

wordcloud = WordCloud(max_font_size=None, background_color="black", width=1200, height=1000).generate(dogNames)

plt.imshow(wordcloud)

plt.title("Top dog names")

plt.axis('off')
print(train[["AdoptionSpeed"]][(train["Name"].apply(lambda x: str(x).lower()).str.contains("pup")) | +

                        (train["Name"].apply(lambda x: str(x).lower()).str.contains("kit")) ].mean())

print(train[["AdoptionSpeed"]][(~train["Name"].apply(lambda x: str(x).lower()).str.contains(r'\d')) &

                                (~train["Name"].apply(lambda x: str(x).lower()).str.contains("pup")) & +

                                (~train["Name"].apply(lambda x: str(x).lower()).str.contains("kit"))].mean())

print(train[["AdoptionSpeed"]][(train["Name"].apply(lambda x: str(x).lower()).str.contains(r'\d')) &

                              (~train["Name"].apply(lambda x: str(x).lower()).str.contains("pup")) & +

                              (~train["Name"].apply(lambda x: str(x).lower()).str.contains("kit"))].mean())
"""

New features:

Baby = name contains pup or kit

Normal = name does not contain pup or kit

Strange = name contains numbers

"""

merged["BabyName"] = 0

merged["NormalName"] = 0

merged["StrangeName"] = 0



merged["BabyName"][(merged["Name"].apply(lambda x: str(x).lower()).str.contains("pup")) | +

                        (merged["Name"].apply(lambda x: str(x).lower()).str.contains("kit"))] = 1



merged["NormalName"][(~merged["Name"].apply(lambda x: str(x).lower()).str.contains(r'\d')) &

                                (~merged["Name"].apply(lambda x: str(x).lower()).str.contains("pup")) & +

                                (~merged["Name"].apply(lambda x: str(x).lower()).str.contains("kit"))] = 1



merged["StrangeName"][(merged["Name"].apply(lambda x: str(x).lower()).str.contains(r'\d')) &

                              (~merged["Name"].apply(lambda x: str(x).lower()).str.contains("pup")) & +

                              (~merged["Name"].apply(lambda x: str(x).lower()).str.contains("kit"))] = 1
merged[["Name", "BabyName", "NormalName", "StrangeName"]].describe(include='all')
train.columns
train["Color1"].value_counts()
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

make_count_plot(train, "Color1", super_ax=ax[0])

make_count_plot(train, "Color2", super_ax=ax[1])

make_count_plot(train, "Color3", super_ax=ax[2])
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))

make_count_plot(train[train["Type"]==1], "FurLength",title="Dogs", super_ax=ax[0])

make_count_plot(train[train["Type"]==2], "FurLength", title="Cats",super_ax=ax[1])
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15,15))

make_count_plot(train[train["Type"]==1], "Vaccinated",title="Dogs", super_ax=ax[0][0])

make_count_plot(train[train["Type"]==2], "Vaccinated", title="Cats",super_ax=ax[0][1])



make_count_plot(train[train["Type"]==1], "Dewormed",title="Dogs", super_ax=ax[1][0])

make_count_plot(train[train["Type"]==2], "Dewormed", title="Cats",super_ax=ax[1][1])



make_count_plot(train[train["Type"]==1], "Sterilized",title="Dogs", super_ax=ax[2][0])

make_count_plot(train[train["Type"]==2], "Sterilized", title="Cats",super_ax=ax[2][1])


print("The mean age of vaccinated animals is {}".format(round(train["Age"][train["Vaccinated"]==3].mean()),2))

print("The mean age of not vaccinated animals is {}".format(round(train["Age"][train["Vaccinated"]==2].mean()),2))

print("The mean age of animals with unknown vaccination is {}".format(round(train["Age"][train["Vaccinated"]==1].mean()),2))

print('-'*45)

print("The mean age of dewormed animals is {}".format(round(train["Age"][train["Dewormed"]==3].mean()),2))

print("The mean age of not dewormed animals is {}".format(round(train["Age"][train["Dewormed"]==2].mean()),2))

print("The mean age of animals with unknown dewormed is {}".format(round(train["Age"][train["Dewormed"]==1].mean()),2))

print('-'*45)

print("The mean age of sterilized animals is {}".format(round(train["Age"][train["Sterilized"]==3].mean()),2))

print("The mean age of not sterilized animals is {}".format(round(train["Age"][train["Sterilized"]==2].mean()),2))

print("The mean age of animals with unknown sterilization is {}".format(round(train["Age"][train["Sterilized"]==1].mean()),2))
breed_ct = pd.crosstab(train["Quantity"],train["AdoptionSpeed"],margins=True).apply(percConvert, axis=1)

print(breed_ct)
train["MoreThanOne"] = 1

train["MoreThanOne"][train["Quantity"]==1]=0
train["Fee"].value_counts()[:10]
print("Average AdoptionSpeed for animals w/o fee {}".format(round(train["AdoptionSpeed"][train["Fee"]==0].mean(),2)))

print("Average AdoptionSpeed for animals w/ fee {}".format(round(train["AdoptionSpeed"][train["Fee"]!=0].mean(),2)))
train["VideoAmt"].value_counts()



video_ct = pd.crosstab(train["VideoAmt"], train["AdoptionSpeed"], margins=True).apply(percConvert, axis=1)

print(video_ct)
train["Video"] = 0

train["Video"][train["VideoAmt"]>0]=1
make_count_plot(train, "Video")
train["PhotoAmt"].value_counts()



video_ct = pd.crosstab(train["PhotoAmt"], train["AdoptionSpeed"], margins=True).apply(percConvert, axis=1)

print(video_ct)
temp = train.copy(deep=True)

temp["PhotoCut"] = pd.cut(temp["PhotoAmt"], 10)

plt.figure(figsize=(15,5))

make_count_plot(temp, "PhotoCut")
merged_backup = merged.copy(deep=True)
merged = merged_backup.copy(deep=True)
merged.shape
n_components = 5

text_feature = []



print('generating features from "Description".')



svd_ = TruncatedSVD(n_components = n_components, random_state=1337)



nmf_ = NMF(n_components=n_components, random_state=1337)



tfidf_col = TfidfVectorizer(ngram_range=(1, 2),stop_words='english').fit_transform(merged["Description"].fillna('NaN').values)



svd_col = svd_.fit_transform(tfidf_col)

svd_col = pd.DataFrame(svd_col)

svd_col = svd_col.add_prefix('SVD_Description')



nmf_col = nmf_.fit_transform(tfidf_col)

nmf_col = pd.DataFrame(nmf_col)

nmf_col = nmf_col.add_prefix('NMF_Description')



text_feature.append(svd_col)

text_feature.append(nmf_col)

text_feature = pd.concat(text_feature, axis=1)

text_feature.set_index(merged.index, inplace=True)



merged =  merged.merge(text_feature, left_index=True, right_index=True, how='outer')
msn.matrix(merged, sort='ascending')
toEncode = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',

       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',

       'Sterilized', 'Health', 'State']



ohe = OneHotEncoder()



temp = ohe.fit_transform(merged[toEncode])

temp = pd.DataFrame(temp.toarray(), index=merged.index)

merged = merged.merge(temp, left_index=True, right_index=True, how='outer')
train = merged[merged["IsTraining"] == 1]

test = merged[merged["IsTraining"] == 0]
X = train.drop(['SVD_Description0', 'SVD_Description1',

'SVD_Description2', 'SVD_Description3', 'SVD_Description4',

'RescuerID','Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',

'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',

'Sterilized', 'Health', 'State',"Name", "Description","AdoptionSpeed", "IsTraining"],axis=1).values

y = train["AdoptionSpeed"].values



print(X.shape)

print(y.shape)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)



print("Training features size: {}".format(X_train.shape))#

print("Training label size: {}".format(y_train.shape))

print("Testing features size: {}".format(X_test.shape))

print("Testing label size: {}".format(y_test.shape))
X.sample()
Classifier
sns.barplot(Classifier["Classifier"], Classifier["Cohen_Kappa"])

plt.xticks(rotation=90)
plt.bar(x=range(5),height=Classifier[["0_pred", "1_pred","2_pred", "3_pred","4_pred"]].mean(),alpha=.5,color="red")

plt.bar(x=range(5), height=X_temp["AdoptionSpeed"].value_counts(normalize=True).sort_values(),alpha=.5,color="blue")
clf.best_params_
clf.best_score_
params = {'colsample_bytree': [0.75],

'learning_rate': [0.1],

 'max_depth': [10],

 'n_estimators': [100],

 'num_leaves': [33],

 'objective': ['multiclass'],

 'random_seed': [1337],

 'reg_alpha': [0.25],

 'reg_lambda': [0,0],

 'silent': [True],

 'subsample': [0.8]}
LGB = lgb.LGBMClassifier()

clf = GridSearchCV(LGB, params, cv=5)

clf.fit(X_train,y_train)
clf.best_params_
params = clf.best_params_



print(X.shape)

print(test.shape)
LGB.fit(X,y)

LGB.predict(test)
test.drop(['SVD_Description0', 'SVD_Description1',

'SVD_Description2', 'SVD_Description3', 'SVD_Description4',

'RescuerID','Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',

'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',

'Sterilized', 'Health', 'State',"Name", "Description","AdoptionSpeed", "IsTraining"],axis=1, inplace=True)
preds = clf.predict(test)
submission = pd.DataFrame([int(i) for i in preds], index=test.index)

submission.columns = ["AdoptionSpeed"]
submission
submission.to_csv("submission.csv")
import gc

gc.collect()