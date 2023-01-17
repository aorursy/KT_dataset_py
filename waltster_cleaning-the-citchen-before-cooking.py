%logstart -o -t log.txt
#Debugging: Use only 100 samples for training and testing
DEBUG_SAMPLES=0
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
sb.set()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
with open("../input/train.json", "r") as file_train:
    dat = json.load(file_train)
    dat = pd.DataFrame(dat)
    dat.set_index("id", inplace=True, verify_integrity=True)
dat.head()
dat.shape
#debugging
if DEBUG_SAMPLES:
    dat=dat.sample(DEBUG_SAMPLES)
dat.isna().any()
recipes_by_country = dat.groupby("cuisine").count().sort_values("ingredients", ascending=False)
recipes_by_country["ingredients"].plot(kind="bar", figsize=(10,5), fontsize=14)
dat["num_ingredients"] = dat.apply(lambda x: len(x["ingredients"]), axis=1)
dat.head()
dat.groupby("cuisine").describe().sort_values([("num_ingredients","mean")], ascending=False)
from collections import Counter

ing_counts_by_cuisine = {}

def increment_counter(y):
    #y is a Series object with the column names as index
    cuisine = y["cuisine"]
    ing_counts_by_cuisine[cuisine].update(y["ingredients"])

def count_ing(x):
    #x is a DataFrame containing all recipes for one cuisine
    cuisine = x["cuisine"].iloc[0]
    ing_counts_by_cuisine[cuisine] = Counter()
    x.apply(increment_counter, axis=1)

dat_grouped_by_cuisine = dat.groupby("cuisine").apply(count_ing)
most_common = 10

# Lets draw an HTML table showing the most common ingredients per cuisine
# Make a Dataframe out of it
ings = pd.DataFrame(
        [[items[0] for items in ing_counts_by_cuisine[cuisine].most_common(most_common)] for cuisine in ing_counts_by_cuisine],
        columns=["Ingredient "+str(idx+1) for idx in range(most_common)],
        index=[cuisine for cuisine, items in ing_counts_by_cuisine.items()])
ings

# This time I am using the CountVectorizer of scikit to do the actual counting of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
# lets first create a huge list concatenating all ingredient lists of all training data we have
# Iterating over a two nested iterables. This is not performant. Is there a better way?
huge_ing_list = pd.Series([ing for idx, row in dat.iterrows() for ing in row["ingredients"]])
huge_ing_list.shape
cv.fit_transform(huge_ing_list)
pd.Series(cv.vocabulary_).sample(50)
def to_lowercase_wo_whitespace(x):
    #x is a series
    x["ingredients"] = [ing.lower() for ing in x["ingredients"]]
    x["ingredients"] = [ing.strip() for ing in x["ingredients"]]
    
    return x

dat = dat.apply(to_lowercase_wo_whitespace, axis=1);
def count_ingredients():
    ing_counts = Counter()

    def count_ing_separately(x):
        #x is Series with one recipe
        ing_counts.update(x["ingredients"])

        return x

    dat.apply(count_ing_separately, axis=1)
    ing_counts = pd.DataFrame(ing_counts.most_common(), columns=['ingredient', 'occurance'])

    #Lets sort them
    ing_counts.sort_values('occurance',ascending=False)
    return ing_counts
ing_counts = count_ingredients()
ing_counts
ing_counts.shape
import re
def remove_numbers_symbols(x):
    ings = x["ingredients"]
    for idx, ing in enumerate(ings):
        ings[idx] = re.sub(r"\W+", " ", ing)
        ings[idx] = re.sub(r"oz[. ]", " ", ing)
    x["ingredients"] = ings
    return x

dat = dat.apply(remove_numbers_symbols, axis=1)
dat.sample(20)
from nltk.stem import PorterStemmer

ps = PorterStemmer()
def stem_ingredients(x):
    ings = x["ingredients"]
    for idx, ing in enumerate(ings):
        ings[idx] = ps.stem(ing)
    x["ingredients"] = ings
    return x

dat = dat.apply(stem_ingredients, axis=1)
dat.sample(20)
import fuzzywuzzy
from fuzzywuzzy import process
#Count the preprocessed ingredients again
ing_counts = count_ingredients()

#Make the ingredient to be the new index and get a unique Series after the last preprocessing steps
ing_counts.set_index('ingredient', inplace=True)
ing_counts.head()
ing_counts.shape
from mlxtend.preprocessing import minmax_scaling

# We want to save our fuzzy match resolution to avoid doing the matching for one ingredient several times
ing_counts["fuzzy_match"] = None

def fuzzy_match(x, min_fuzzy_ratio = 90):
    recipe_ingredients = x["ingredients"]
    for idx, ing in enumerate(recipe_ingredients):
        cur_fuzzy_match = ing_counts.loc[ing, ["fuzzy_match"]].iloc[0]
        if (cur_fuzzy_match == None):
            #Lets use my ingredient counters dataframe to get the whole list of ingredients
            matches = fuzzywuzzy.process.extract(ing, ing_counts.index, limit=5, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
            #Minimimum fuzzy matching ratio
            matches = [match for match in matches if match[1] >= min_fuzzy_ratio]
            #Interleave with the count values of the words
            matches_df=pd.DataFrame(matches, columns=["ingredient", "match_score"]) 
            matches_df = matches_df.merge(ing_counts, how='inner', on="ingredient")
            #Normalize
            matches_df['match_score_norm']= matches_df["match_score"] / 100
            if (len(matches_df) > 2):
                matches_df['occurance_norm']= minmax_scaling(matches_df["occurance"], columns=[0])
            else:
                matches_df['occurance_norm'] = 1.0
            #Create an overall scoring
            matches_df['overall_score'] = matches_df['match_score_norm'] * matches_df['occurance_norm']
            # Get the ingredient with the maximum scoring
            idx_max = matches_df['overall_score'].idxmax()
            #Replace the ingredient by the ingredient with the maximum overall scoring
            if (recipe_ingredients[idx] != matches_df.loc[idx_max, "ingredient"]):
                print ("Ingredient {} replaced by {}".format(recipe_ingredients[idx], matches_df.loc[idx_max, "ingredient"]))
                recipe_ingredients[idx] = matches_df.loc[idx_max, "ingredient"]

            #Save the choice to avoid further fuzzy matching when the ingredient shows up again
            if (ing in ing_counts.index.values):
                ing_counts.loc[ing, ["fuzzy_match"]] = matches_df.loc[idx_max, "ingredient"]
        else:
            recipe_ingredients[idx] = cur_fuzzy_match
    x["ingredients"] = recipe_ingredients
    
    return x
dat = dat.apply(fuzzy_match, axis=1);
#degugging
#dat.sample(1).apply(fuzzy_match, axis=1);
ing_counts
#This is our final ingredients list
ingredients_list = ing_counts["fuzzy_match"].unique()

ingredients_list.shape
import nltk
tagged_ings = nltk.pos_tag(huge_ing_list)
type(tagged_ings)
ing_counts[ing_counts["occurance"] == 1].shape
ing_counts[ing_counts["occurance"] == 1].shape[0] / ing_counts.shape[0] * 100
ing_counts.head()
matched_ingredients = ing_counts["fuzzy_match"].unique()
ing_counts_reduced = pd.DataFrame([ ing_counts.loc[ing, :] for ing in matched_ingredients if ing_counts.loc[ing, "occurance"] > 10 ])
ing_counts_reduced.drop("fuzzy_match", axis=1);
ing_counts_reduced.index.name = "ingredient"
ing_counts_reduced.shape
def remove_unique_ings(x):
    recipe_ingredients = []
    for ing in x["ingredients"]:
        if ing in ing_counts_reduced.index.values.tolist():
            recipe_ingredients.append(ing)
    x["ingredients"] = recipe_ingredients
    
    return x
# Commenting out as discussed above...
#dat = dat.apply(remove_unique_ings, axis=1);
from sklearn.base import BaseEstimator, TransformerMixin
class CuisineTransformer(BaseEstimator, TransformerMixin):  
    """Data cleaning for the cuisine classification"""

    def __init__(self, min_fuzzy_ratio=90, min_occurance=10, predefined_ingredients = [], enable_stemming=True, debug_print=False):
    
        """
        Called when initializing the classifier
        """
        self.min_fuzzy_ratio = min_fuzzy_ratio
        self.min_occurance = min_occurance
        self.enable_stemming = enable_stemming
        self.debug_print = debug_print
        
        self.ps = PorterStemmer()
        self.ing_counts = Counter()
        self.ing_list = pd.DataFrame()
        self.ing_list_reduced = pd.DataFrame()
        self.predefined_ingredients = pd.DataFrame(predefined_ingredients)
        
        self.processed_ingredients_ = pd.DataFrame()
        self.transformed_lower_wo_whitespace_ = pd.DataFrame()
        self.transformed_removed_numbers_ = pd.DataFrame()
        self.transformed_stemmed_ingredients_ = pd.DataFrame()
        self.transformed_fuzzy_matched_ingredients_ = pd.DataFrame()
        self.transformed_removed_unique_ingredients_ = pd.DataFrame()
        self.transformed_ = pd.DataFrame()

    def fit(self, X, y=None):
        """
        The main work is done in the transform method
        """
        return self

    def transform(self, X, y=None):
        self.transformed_ = X
        self.transformed_lower_wo_whitespace_ = self.transformed_.apply(self._to_lowercase_wo_whitespace, axis=1)
        self.transformed_removed_numbers_ = self.transformed_lower_wo_whitespace_.apply(self._remove_numbers_symbols, axis=1)
        
        if (self.enable_stemming == True):
            self.transformed_stemmed_ingredients_ = self.transformed_removed_numbers_.apply(self._stem_ingredients, axis=1)
        
        if len(self.predefined_ingredients) == 0:
            self._count_ingredients(self.transformed_stemmed_ingredients_)
            self.ing_list.set_index('ingredient', inplace=True)
        else:
            self.ing_list = self.predefined_ingredients
        
        self.ing_list["fuzzy_match"] = None
        self.transformed_fuzzy_matched_ingredients_ = self.transformed_stemmed_ingredients_.apply(self._fuzzy_match, axis=1);
        matched_ingredients = self.ing_list["fuzzy_match"].unique()
        
        if (self.min_occurance > 1):
            self.ing_list_reduced = pd.DataFrame([ self.ing_list.loc[ing, :] for ing in matched_ingredients if self.ing_list.loc[ing, "occurance"] > 10 ])
            self.ing_list_reduced.drop("fuzzy_match", axis=1);
            self.ing_list_reduced.index.name = self.ing_list.index.name
            self.transformed_removed_unique_ingredients_ = self.transformed_fuzzy_matched_ingredients_.apply(self._remove_unique_ings, axis=1)
            
            self.processed_ingredients_ = self.ing_list_reduced.index
            self.transformed_ = self.transformed_removed_unique_ingredients_
        else:
            self.processed_ingredients_ = matched_ingredients
            self.transformed_ = self.transformed_fuzzy_matched_ingredients_
            
        return self
    
    def _to_lowercase_wo_whitespace(self, x):
        #x is a series
        x["ingredients"] = [ing.lower() for ing in x["ingredients"]]
        x["ingredients"] = [ing.strip() for ing in x["ingredients"]]
        return x
    
    def _remove_numbers_symbols(self, x):
        ings = x["ingredients"]
        for idx, ing in enumerate(ings):
            ings[idx] = re.sub(r"\W+", " ", ing)
            ings[idx] = re.sub(r"oz[. ]", " ", ing)
        x["ingredients"] = ings
        return x
    
    def _stem_ingredients(self, x):
        ings = x["ingredients"]
        for idx, ing in enumerate(ings):
            ings[idx] = self.ps.stem(ing)
        x["ingredients"] = ings
        return x

    def _count_ingredients(self, x):
        x.apply(self._count_ing_separately, axis=1)
        self.ing_list = pd.DataFrame(self.ing_counts.most_common(), columns=['ingredient', 'occurance'])

        #Lets sort them
        self.ing_list.sort_values('occurance',ascending=False)
    
    def _count_ing_separately(self, x):
        self.ing_counts.update(x["ingredients"])
        return x
    
    def _fuzzy_match(self, x):
        recipe_ingredients = x["ingredients"]
        for idx, ing in enumerate(recipe_ingredients):
            # Check if the ingredient is already in our ingredient list
            # If the ingredient list was predefined it might not be in there
            if ing in self.ing_list.index:
                cur_fuzzy_match = self.ing_list.loc[ing, ["fuzzy_match"]].iloc[0]
            else:
                cur_fuzzy_match = None
                
            if (cur_fuzzy_match == None):
                #Lets use my ingredient counters dataframe to get the whole list of ingredients
                matches_raw = fuzzywuzzy.process.extract(ing, self.ing_list.index, limit=5, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
                #Minimimum fuzzy matching ratio
                matches = [match for match in matches_raw if match[1] >= self.min_fuzzy_ratio]
                #Interleave with the count values of the words
                matches_df=pd.DataFrame(matches, columns=["ingredient", "match_score"]) 
                matches_df = matches_df.merge(self.ing_list, how='inner', on="ingredient")
                if (len(matches_df) > 0):
                    #Normalize
                    matches_df['match_score_norm']= matches_df["match_score"] / 100
                    if (len(matches_df) > 2):
                        matches_df['occurance_norm']= minmax_scaling(matches_df["occurance"], columns=[0])
                    else:
                        matches_df['occurance_norm'] = 1.0
                    #Create an overall scoring
                    matches_df['overall_score'] = matches_df['match_score_norm'] * matches_df['occurance_norm']
                    # Get the ingredient with the maximum scoring
                    idx_max = matches_df['overall_score'].idxmax()
                    #Replace the ingredient by the ingredient with the maximum overall scoring
                    if (recipe_ingredients[idx] != matches_df.loc[idx_max, "ingredient"]):
                        if (self.debug_print == True):
                            print ("Ingredient {} replaced by {}".format(recipe_ingredients[idx], matches_df.loc[idx_max, "ingredient"]))
                        recipe_ingredients[idx] = matches_df.loc[idx_max, "ingredient"]

                    #Save the choice to avoid further fuzzy matching when the ingredient shows up again
                    if (ing in self.ing_list.index.values):
                        self.ing_list.loc[ing, ["fuzzy_match"]] = matches_df.loc[idx_max, "ingredient"]
                else:
                    if (self.debug_print == True):
                        print ("          No fuzzy match for ingredient {}. Next match is {} with score {}. Use original".format(ing, matches_raw[0][0], matches_raw[0][1]))
            else:
                recipe_ingredients[idx] = cur_fuzzy_match
        x["ingredients"] = recipe_ingredients

        return x

    def _remove_unique_ings(self, x):
        recipe_ingredients = []
        for ing in x["ingredients"]:
            if ing in self.ing_list_reduced.index.values.tolist():
                recipe_ingredients.append(ing)
        x["ingredients"] = recipe_ingredients

        return x
# Lets use Sklearn to transform the ingredient lists to binarized feature matrix first
from sklearn.preprocessing import MultiLabelBinarizer
y = dat["cuisine"]
#Pass the ingredients_list as a preordering to MLB. We need this to restore the labels later-on
mlb = MultiLabelBinarizer(ingredients_list)
X = mlb.fit_transform(dat["ingredients"])
X.shape
feature_labels = mlb.classes_
feature_labels.shape
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectPercentile
kbest_model = SelectPercentile(score_func=chi2, percentile=50)
%time X_chi2 = kbest_model.fit_transform(X,y)
# lets look at the score
X_chi2.shape
chi2_features_by_importance = pd.DataFrame({ "Feature": feature_labels, "Score": kbest_model.scores_})
chi2_features_by_importance = chi2_features_by_importance.iloc[kbest_model.get_support(), :]
chi2_features_by_importance.sort_values(by="Score", ascending=False, inplace=True)
chi2_features_by_importance.head(20)
chi2_features_by_importance.shape
# Lets use the recursive feature eleminiation (RFE) from sklearn
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
rfe_model = RFE( \
    estimator=LogisticRegression(solver="newton-cg",        #newton-cg work well on multinomial datasets
                                 multi_class="multinomial", #lets use softmax instead of OVR
                                 n_jobs=-1,                 #use all CPUs
                                 max_iter=5),               #stop quickly        
    step=100)                                        #increased step size to make things faster
%time X_rfe = rfe_model.fit_transform(X, y)
X_rfe.shape
rfe_features_by_importance = pd.DataFrame({ "Feature": feature_labels, "Ranking": rfe_model.ranking_})
rfe_features_by_importance = rfe_features_by_importance.iloc[rfe_model.support_,:]
rfe_features_by_importance.head(20)
#The ranking is always 1 if the feature got selected, hence always 1 in this DataFrame
rfe_features_by_importance.shape
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
rfc = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=8)
%time rfc.fit(X, y)
rfc_model = SelectFromModel(rfc,
                            prefit=True,           #Fit was already called
                            threshold="0.1*mean")  #More features to have a similar amount of features
                                                   #like chosen by the other feature selection algo's
%time X_rfc = rfc_model.transform(X)
X_rfc.shape
rfc_features_by_importance = pd.DataFrame({ "Feature": feature_labels, "Importance": rfc.feature_importances_})
rfc_features_by_importance = rfc_features_by_importance.iloc[rfc_model.get_support(), :]
rfc_features_by_importance.head(20)
rfc_features_by_importance.shape
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
# This setting seems to work quite well. Parameters come from tuning, done below
svc = LinearSVC(C=10)
svc_ada = AdaBoostClassifier(base_estimator=svc, algorithm="SAMME")
#Prediction comes below
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
bnb_ada = AdaBoostClassifier(base_estimator=bnb, algorithm="SAMME")
#Prediction comes below
from sklearn.model_selection import cross_val_score
# Naive Bayes on original but cleaned up dataset X
bnb = BernoulliNB()
%time cross_bnb = cross_val_score(bnb, X, y, cv=10)
cross_bnb
np.mean(cross_bnb)
# Naive Bayes on dataset that was scope of univariate feature selection
%time cross_chi2 = cross_val_score(bnb, X_chi2, y, cv=10)
cross_chi2
np.mean(cross_chi2)
# Naive Bayes on dataset that was scope of recursive feature elimination
%time cross_rfe = cross_val_score(bnb, X_rfe, y, cv=10)
cross_rfe
np.mean(cross_rfe)
# Naive Bayes on dataset that was scope of model based feature selection (Random forest)
%time cross_rfc = cross_val_score(bnb, X_rfc, y, cv=10)
cross_rfc
np.mean(cross_rfc)
# Adaboost performs bad and takes incredibly much time... commenting out
# otherwise the kernel exceeds time limit...

# An ensemble of Naive Bayes created by Ada Boost
#%time cross_ada = cross_val_score(bnb_ada, X, y, cv=10)
#cross_ada
#np.mean(cross_ada)
from sklearn.multiclass import OneVsRestClassifier
svc = LinearSVC(C=10)
# SVC on original but cleaned up dataset X
%time cross_svc = cross_val_score(svc, X, y, cv=10)
cross_svc
np.mean(cross_svc)
# SVC on dataset that was scope of univariate feature selection
%time cross_chi2_svc = cross_val_score(svc, X_chi2, y, cv=10)
cross_chi2_svc
np.mean(cross_chi2_svc)
# SVC on dataset that was scope of recursive feature elimination
%time cross_rfe_svc = cross_val_score(svc, X_rfe, y, cv=10)
cross_rfe_svc
np.mean(cross_rfe_svc)
# SVC on dataset that was scope of model based feature selection (Random forest)
%time cross_rfc_svc = cross_val_score(svc, X_rfc, y, cv=10)
cross_rfc_svc
np.mean(cross_rfc_svc)
# Adaboost performs bad and takes incredibly much time... commenting out
# otherwise the kernel exceeds time limit...

# An ensemble of SVC's created by Ada Boost
#%time cross_ada = cross_val_score(svc_ada, X, y, cv=10)
#cross_ada
#np.mean(cross_ada)
#Train the selected model with all training data
model_selection = svc
X_selection = X_rfc
feature_selection = rfc_features_by_importance["Feature"]
param_grid = [
  {
   'C': [1, 0.1, 0.01, 10]
  }
]
from sklearn.model_selection import GridSearchCV
gsc = GridSearchCV(model_selection, param_grid, n_jobs=-1, cv=10, iid=True)
%time gsc.fit(X_selection, y)
means = gsc.cv_results_['mean_test_score']
stds = gsc.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, gsc.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
gsc.best_params_
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
%time y_pred = cross_val_predict(model_selection, X_selection, y, cv=10)
confusion = confusion_matrix(y, y_pred)
y_classes = np.unique(y)
y_classes.sort()
fig, ax = plt.subplots(figsize=(10,10))  
sb.heatmap(confusion, xticklabels=y_classes, yticklabels=y_classes, cbar=False, annot=True, ax=ax)
plt.xlabel('predicted value')
plt.ylabel('true value');
#Train the selected model with all training data
C=gsc.best_params_['C']
#For RBF kernels ... they take too long we use linear instead...
#gamma=gsc.best_params_['gamma']
#kernel=gsc.best_params_['kernel']
model_selection = LinearSVC(C=C)
#Train model with entire training data
model_selection.fit(X_selection,y)
with open("../input/test.json", "r") as file_test:
    test_set = json.load(file_test)
    test_set = pd.DataFrame(test_set)
    test_set.set_index("id", inplace=True, verify_integrity=True)
test_set.head()
#Debugging
if DEBUG_SAMPLES:
    test_set = test_set.sample(DEBUG_SAMPLES)
test_set.shape
#Process the test_set
#This time we want to fuzzy match against a lot more. If things are still somehow similar its good
#Furthermore we dont' want to throw away any ingredient
ct = CuisineTransformer(min_fuzzy_ratio=90, min_occurance=1, predefined_ingredients=ing_counts_reduced)
ct.fit_transform(test_set)
ct.transformed_
#If we would have used the CuisineTransformer for the train set we could just take out
#the processed_ingredients from there and put it in the transformer for the test set 
ct.transformed_.shape
ct.transformed_.head()
mlb_test = MultiLabelBinarizer(feature_selection)
X_test = mlb_test.fit_transform(ct.transformed_["ingredients"]);
print ("Known classes from train set: {}.".format(ing_counts_reduced.index.shape[0]))
print ("Total classes from test set: {}.".format(ct.processed_ingredients_.shape[0]))
unknown_ings = [ing_test for ing_test in ct.processed_ingredients_ if ing_test not in ing_counts_reduced.index.values]
unknown_ings = pd.Series(unknown_ings)
print ("Total unknown classes: {}.".format(unknown_ings.shape[0]))
print ("{:.2f}% unknown".format(unknown_ings.shape[0]/ct.processed_ingredients_.shape[0]*100))
unknown_ings.head()
y_test = model_selection.predict(X_test)
test_ids = ct.transformed_.index
#test_ids = test_set.index
sub = pd.DataFrame({'id': test_ids, 'cuisine': y_test}, columns=['id', 'cuisine'])
sub.to_csv('output.csv', index=False)
sub.head()
# Do some consistency checking
# join by index and compare if the ingredients lists are similar
ct.transformed_.join(test_set, how="inner", lsuffix="_transformed", rsuffix="_sub").sample(10)
# This one looks good. Now check if the index complies with the index in the submission test set
sub_with_index = sub.set_index("id")
sub_with_index.join(ct.transformed_, how="inner", lsuffix="_transformed", rsuffix="_sub").sample(100)