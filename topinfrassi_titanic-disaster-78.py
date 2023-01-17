import numpy as np

import pandas

import matplotlib.pyplot as plt



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.preprocessing import StandardScaler, Imputer, OneHotEncoder, LabelBinarizer, LabelEncoder
# Possible things to upgrade :

# Split the validation/test sets to make sure they represent the whole population



# In the Name format, we could extract a Title since it's always in the same place. There's probably a good correlation

# Ticket doesn't seem useful, neither does Cabin because of NaN, we'll check that.





np.random.seed(31)

data = pandas.read_csv('../input/train.csv')

data.head(5)
# The main point of this is to check how many null entries there are per column and check the type of each column

data.info()



# There is a lot of NaN in Age.

# There is so much NaN in Cabin that it should be dropped

# Embarked has some NaN too.
data.corr()



# Correlation is strong between Survived and Fare

# Correlation is strong between Survived and PClass.

# Correlation is strong between Fare and PClass, maybe one of them should be dropped to remove false correlation

# Should produce histograms of Fare/Survived and PClass/Survived
v_data = data.copy()

fare_labels = ["Low", "Medium", "Medium-High", "High"]

v_data["Fare"] = pandas.cut(v_data["Fare"], bins=[0,10,50,100,600], labels=fare_labels, include_lowest=True).factorize()[0]



width = 0.3

fare_survived = plt.bar(v_data["Fare"].unique(), v_data[v_data["Survived"] == 1]["Fare"].value_counts(), width,  color="r")

fare_died = plt.bar(v_data["Fare"].unique() + width, v_data[v_data["Survived"] == 0]["Fare"].value_counts(), width, color="g")

plt.legend((fare_survived[0], fare_died[0]), ('Survived', 'Died'))

plt.xticks(v_data["Fare"].unique(), fare_labels)

plt.show()



# There seems to be a good correlation. People who paid less than 50 dollars seem to alot die more (feature?)
pclass_labels = ["Low", "Medium", "High"]

pclass_values = np.sort(v_data["Pclass"].unique())



width = 0.3

pclass_survived = plt.bar(pclass_values, v_data[v_data["Survived"] == 1]["Pclass"].value_counts(), width,  color="r")

pclass_died = plt.bar(pclass_values + width, v_data[v_data["Survived"] == 0]["Pclass"].value_counts(), width, color="g")

plt.legend((pclass_survived[0], pclass_died[0]), ('Survived', 'Died'))

plt.xticks(pclass_values, pclass_labels)

plt.show()



# People in low class died much more than the others (feature?)
class TitleAttributeAdder(BaseEstimator, TransformerMixin):



    def __init__(self, remove_sex=False):

        self.remove_sex = remove_sex



    def fit(self, X, y = None):

        return self



    def transform(self, X, y = None):

        x_copy = X.copy(deep=True)

        titles = x_copy.apply(TitleAttributeAdder.__extract_title, axis=1)

        title_names = (titles.value_counts() < 10)

        titles = titles.apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

        x_copy["Title"] = titles

        return x_copy



    @staticmethod

    def __extract_title(a):

        comma_index = 0

        name = a["Name"]

        try:

            comma_index = name.index(',') + 1

            result = name[comma_index:]

            space_index = result.index(".")

            return result[:space_index].strip()

        except AttributeError as ae:

            print(a)

            raise Exception

        



p_data = data.copy(deep = True)



plt.figure(figsize=(20,15))



ax1 = plt.subplot2grid((2, 2), (0, 0))

ax2 = plt.subplot2grid((2, 2), (0, 1))

ax3 = plt.subplot2grid((2, 2), (1, 0))



width = 0.35

p_data["FamilySize"] = p_data["Parch"] + p_data["SibSp"] + 1

p_data["FamilySize"] = pandas.cut(p_data["FamilySize"], bins=[-1,1,2, 4, 100])

p_data["FamilySize"] = p_data["FamilySize"].factorize()[0]

fam_survived = ax1.bar(p_data["FamilySize"].unique(), p_data["FamilySize"].astype('category')[p_data["Survived"] == 1].value_counts(), width,  color="r")

fam_died = ax1.bar(p_data["FamilySize"].unique() + width, p_data["FamilySize"].astype('category')[p_data["Survived"] == 0].value_counts(), width, color="g")

ax1.legend((fam_survived[0], fam_died[0]), ('Survived', 'Died'))

ax1.set_title("Survival/Death by family size")

ax1.set_xticks(range(p_data["FamilySize"].max() + 1))

ax1.set_xticklabels(["Alone","Couple", "2 < x < 5", ">= 5"])



val, labels = TitleAttributeAdder().transform(data)["Title"].factorize()

p_data["Title"] = val

title_survived = ax2.bar(p_data["Title"].unique(), p_data["Title"].astype('category')[p_data["Survived"] == 1].value_counts(), width,  color="r")

title_died = ax2.bar(p_data["Title"].unique() + width, p_data["Title"].astype('category')[p_data["Survived"] == 0].value_counts(), width, color="g")

ax2.legend((title_survived[0], title_died[0]), ('Survived', 'Died'))

ax2.set_xticks(range(p_data["Title"].max() + 1))

ax2.set_xticklabels(labels)

ax2.set_title("Survival/Death by title")





p_data["Alone"] = p_data["FamilySize"].apply(lambda x : 1 if x == 1 else 0)

alone_survived = ax3.bar(p_data["Alone"].unique(), p_data[p_data["Survived"] == 1]["Alone"].value_counts(), width,  color="r")

alone_died = ax3.bar(p_data["Alone"].unique() + width, p_data[p_data["Survived"] == 0]["Alone"].value_counts(), width, color="g")

ax3.legend((alone_survived[0], alone_died[0]), ('Survived', 'Died'))

ax3.set_xticks([0,1])

ax3.set_xticklabels(["Alone", "Not alone"])

ax3.set_title("Survival/Death if alone")



plt.show()
v_data[["Age", "Fare", "Survived"]].hist()

plt.show()
# Bug live : Les titres ne sont pas tous assez présents dans le X_test pour apparaitre comme colonne avec le One-Hot

# Il faut s'arranger pour que le fit trouve les colonnes et les ajoutent pareille I guess



class PreparationTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, age_format="scaled", fare_format="scaled", title_format="one-hot",

                 embarked_format="one-hot", drop_sex=False, drop_id=False, drop_survived = False):

        

        self.age_format = age_format

        self.fare_format = fare_format

        self.title_format = title_format

        self.embarked_format = embarked_format

        self.drop_sex = drop_sex

        self.drop_id = drop_id

        self.drop_survived = drop_survived

        self.embarked_mode = ""

        self.age_imputer = Imputer(strategy="median")

        self.fare_imputer = Imputer(strategy="median")

        self.title_lbz = LabelBinarizer()

        self.embarked_lbz = LabelBinarizer()

        self.title_fact = LabelEncoder()

        self.emb_fact = LabelEncoder()

        

    def fit(self, X, y=None):

        df = X.copy(deep=True)

        

        # Maybe find a way to avoid this

        df["Title"] = TitleAttributeAdder().transform(df)["Title"]



        self.embarked_mode = df["Embarked"].mode()[0]

        df["Embarked"] = df["Embarked"].fillna(self.embarked_mode)

        self.age_imputer = self.age_imputer.fit(df[["Age"]])

        self.fare_imputer = self.fare_imputer.fit(df[["Fare"]])

        self.title_fact = self.title_fact.fit(df["Title"])

        self.title_lbz = self.title_lbz.fit(df["Title"])

        self.embarked_lbz = self.embarked_lbz.fit(df["Embarked"])

        self.emb_fact = self.emb_fact.fit(df["Embarked"])

        

        return self

    

    def transform(self, X, y=None):

        df = X.copy(deep = True)

        df.reset_index(drop=True, inplace=True)



        # Clean Embarked

        df["Embarked"] = df["Embarked"].fillna(self.embarked_mode)

        if self.embarked_format == "factorized":

            emb = pandas.DataFrame(self.emb_fact.transform(df["Embarked"]), columns=["Embarked"])

        if self.embarked_format == "one-hot":

            labels = ["Emb_" + x for x in self.embarked_lbz.classes_]

            emb = pandas.DataFrame(self.embarked_lbz.transform(df["Embarked"]), columns=labels)



        df.drop(["Embarked"], axis=1, inplace=True)

        df = pandas.concat([df, emb], axis=1, verify_integrity=True)

        

        # Create Title

        df["Title"] = TitleAttributeAdder().transform(df)["Title"]

        if self.title_format == "factorized":

            titles = pandas.DataFrame(self.title_fact.transform(df["Title"]), columns=["Title"])

        if self.title_format == "one-hot":

            labels = ["Title_" + x for x in self.title_lbz.classes_]

            titles = pandas.DataFrame(self.title_lbz.transform(df["Title"]), columns=labels)    



        df.drop(["Title"], axis=1, inplace=True)        

        df = pandas.concat([df, titles], axis=1, verify_integrity=True)



            

        # Create FamilySize Feature

        df["FamilySize"] = df["Parch"] + df["SibSp"] + 1

        #df["FamilySize"] = pandas.cut(df["FamilySize"], bins=[-1,1,2, 4, 100])

        #df["FamilySize"] = df["FamilySize"].factorize()[0]

        

        # Factorize Sex

        df["Sex"] = df["Sex"].factorize()[0]        

        

        # Add Alone Feature

        df["Alone"] =  df["FamilySize"].apply(lambda x : 1 if x == 1 else 0)

        

        # Clean Fare

        df["Fare"] = self.fare_imputer.transform(df[["Fare"]])

        

        if self.fare_format == "scaled":

            df["Fare"] = StandardScaler().fit_transform(df["Fare"].values.reshape(-1,1))

        elif self.fare_format == "binned":

            df["Fare"] = pandas.cut(df["Fare"], bins=[0,10,50,100,600]).factorize()[0]        

        

        # Clean Age

        df["Age"] = self.age_imputer.transform(df[["Age"]])

        

        if self.age_format == "scaled":

            df["Age"] = StandardScaler().fit_transform(df["Age"].values.reshape(-1,1))

        elif self.age_format == "binned":

            df["Age"] = pandas.cut(df["Age"], np.linspace(0,80,9)).factorize()[0]



        # Drop useless columns

        df.drop(["Cabin","Ticket", "Name"], axis = 1, inplace = True)



        if self.drop_sex == True:

            df.drop(["Sex"], axis=1, inplace = True)

        if self.drop_id == True:

            df.drop(["PassengerId"], axis=1, inplace = True)

        if "Survived" in df.columns and self.drop_survived == True:

            df.drop(["Survived"], axis=1, inplace=True)

            

        return df
pipeline = PreparationTransformer(embarked_format="factorized", title_format="factorized")



df = data.copy(deep=True)

pipeline.fit_transform(df).head(1)
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_auc_score

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import make_scorer



from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
X = data.copy(deep=True)

#X = X.apply(np.random.permutation)

y = X["Survived"]



X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.4, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)

prep = PreparationTransformer(embarked_format="factorized", title_format="one-hot", age_format="scaled", fare_format="scaled", drop_id=True, drop_survived=True)

prep = prep.fit(X_train)
classifiers = [

    KNeighborsClassifier(3),

    SVC(C= 1.1, degree= 2, kernel= 'poly'),

    SVC(gamma=2, C=1),

    GaussianProcessClassifier(1.0 * RBF(1.0)),

    DecisionTreeClassifier(max_depth=5),

    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),

    MLPClassifier(solver="lbfgs"),

    AdaBoostClassifier(),

    GaussianNB(),

    QuadraticDiscriminantAnalysis()]





x_train_transformed = prep.fit_transform(X_train)

x_val_transformed = prep.transform(X_val)



for clf in classifiers:

    clf = clf.fit(x_train_transformed, y_train)

    

    y_pred = clf.predict(x_val_transformed)

    val_score = roc_auc_score(y_val, y_pred)

    

    print("{0} : Cross-validation score {1}".format(clf.__class__.__name__, val_score))
grid_params = {

    "svc__C" : [1,1.1,1.2,1.3,1.4],

    "svc__kernel" : ["poly", "linear"],

    "svc__degree" : [2,3,4,5],

}



clf = Pipeline([

    ("prep", PreparationTransformer(title_format="one-hot", embarked_format="factorized", age_format="scaled", fare_format="scaled", drop_id=True, drop_survived=True)),

    ("svc", SVC())])



rscv = RandomizedSearchCV(clf, param_distributions=grid_params, n_jobs=1, n_iter=15, scoring=make_scorer(roc_auc_score))

rscv = rscv.fit(X_train, y_train)



rscv.best_params_
prep = PreparationTransformer(embarked_format="one-hot", title_format="factorized", age_format="scaled", fare_format="factorized", drop_sex=False, drop_id=True, drop_survived=True)



prep = prep.fit(X_train)



# GaussianNB ended up being the better

clf = GaussianNB() #SVC(C= 1.1, degree= 2, kernel= 'poly')

sub_xt = prep.transform(X_train)

sub_xv = prep.transform(X_val)



results = []

for i in range(20, len(X_train),20):

    clf = clf.fit(sub_xt[:i], y_train[:i])

    

    y_pred_train = clf.predict(sub_xt)

    y_pred_val = clf.predict(sub_xv)

    

    score_train = roc_auc_score(y_train, y_pred_train)

    score_val = roc_auc_score(y_val, y_pred_val)

    results.append((i, score_train, score_val))

    

indexes = [x[0] for x in results]

scores_train = [1 - x[1] for x in results]

scores_val = [1 - x[2] for x in results]



train_plot = plt.plot(indexes, scores_train, color="g")

val_plot = plt.plot(indexes, scores_val, color="r")

plt.legend((train_plot[0], val_plot[0]), ("Train error", "Val error"))

plt.title("Validation Curve")

plt.show()



print("Lowest error : {0}".format(np.array(scores_val).min()))

xtest_prep = prep.transform(X_test)

clf = GaussianNB() #SVC(C= 1.1, degree= 2, kernel= 'poly')

clf = clf.fit(sub_xt, y_train)



test_pred = clf.predict(xtest_prep)

roc_auc_score(y_test, test_pred)
test = pandas.read_csv('../input/test.csv')

X_prediction = test.copy(deep = True)

X_prediction_ids = test["PassengerId"]



# Caution not to call fit_transform, the model was already fitted
X_tran = prep.transform(X)

X_prediction_tran = prep.transform(X_prediction)

print(X_prediction_tran.shape)

print(X_tran.shape)

clf = clf.fit(X_tran, y)

y_pred_pre = clf.predict(X_prediction_tran)
result = np.column_stack((X_prediction_ids, y_pred_pre))

result[0:5]
with open('data/submission_1.csv', 'w') as f : 

    f.write('PassengerId,Survived\n')

    for row in result:

        f.write('{0},{1}\n'.format(row[0], row[1]))

        

    f.close()

    

print("Done! :)")