import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
#data imports

data_train = pd.read_csv("../input/train.csv")

data_test = pd.read_csv("../input/test.csv")

data_train.head()
data_train.describe()
features_soil = ['Soil_Type1', 'Soil_Type2', 'Soil_Type3',

       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',

       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',

       'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',

       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',

       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',

       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',

       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',

       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',

       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']

data_train["Soil_Count"] = data_train[features_soil].apply(sum, axis=1)

data_train.head()
data_train.Soil_Count.describe()
data_test[features_soil].describe()
data_train["Soil_Type"] = data_train[features_soil].apply(np.argmax, axis=1)

data_train.head()
data_train["Soil_Type"] = data_train["Soil_Type"].apply(lambda x: x.split("Soil_Type")[-1])

data_train.head()
features_wilderness = ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3','Wilderness_Area4']

data_train["Wilderness_Area"] = data_train[features_wilderness].apply(sum, axis=1)

data_train.Wilderness_Area.describe()
data_train["Wilderness_Area"] = data_train[features_wilderness].apply(np.argmax, axis=1)

data_train["Wilderness_Area"] = data_train["Wilderness_Area"].apply(lambda x: x.split("Wilderness_Area")[-1])

data_train.Wilderness_Area.head()
sns.countplot(data_train.Cover_Type)

plt.show()
data_train.columns
features = ['Elevation', 'Aspect', 'Slope','Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',

       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', "Cover_Type"]

sns.heatmap(data=data_train[features].corr(), annot=True, linecolor="w", fmt=".1")

plt.show()
fig = plt.figure(figsize=(12,8))

for ind, each in enumerate(["Elevation", "Aspect" , "Slope", "Hillshade_3pm", "Hillshade_Noon", "Hillshade_9am"]):

    plt.subplot(2, 3, ind + 1)

    sns.distplot(data_train[each])

plt.show()
sns.distplot(data_train["Hillshade_Noon"].apply(lambda x: x**4))

plt.show()
sns.distplot(data_train["Hillshade_9am"].apply(lambda x: x**4))

plt.show()
sns.distplot(data_train["Slope"].apply(np.sqrt))

plt.show()
data_train["Aspect_Slope"] = data_train.Aspect * data_train.Slope

sns.distplot(data_train["Aspect_Slope"].apply(np.cbrt))

plt.show()
data_train["Elevation_Slope"] = np.sqrt(data_train.Elevation * data_train.Slope)

sns.distplot(data_train["Elevation_Slope"])

plt.show()
data_train["Elevation_Aspect"] = np.sqrt(data_train.Elevation * data_train.Aspect)

sns.distplot(data_train["Elevation_Aspect"])

plt.show()
data_train["Hillshade_1"] = (data_train.Hillshade_3pm * data_train.Hillshade_Noon * data_train.Hillshade_9am)

sns.distplot(data_train["Hillshade_1"])

plt.show()
fig = plt.figure(figsize=(12,8))

for ind, each in enumerate(['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',

       'Horizontal_Distance_To_Roadways',  'Horizontal_Distance_To_Fire_Points']):

    plt.subplot(2, 2, ind + 1)

    sns.distplot(data_train[each])

plt.show()
sns.distplot(np.sqrt(data_train["Horizontal_Distance_To_Roadways"]))

plt.show()
sns.distplot(np.sqrt(data_train["Horizontal_Distance_To_Fire_Points"]))

plt.show()
def clear_dataset(dataset):

    features_soil = ['Soil_Type1', 'Soil_Type2', 'Soil_Type3',

       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',

       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',

       'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',

       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',

       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',

       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',

       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',

       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',

       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']

    features_wilderness = ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3','Wilderness_Area4']

    dataset["Soil_Type"] = dataset[features_soil].apply(np.argmax, axis=1)

    dataset["Soil_Type"] = dataset["Soil_Type"].apply(lambda x: x.split("Soil_Type")[-1]).astype(int)

    dataset = dataset.drop(["Soil_Type15", "Soil_Type7"], axis=1)

    dataset["Wilderness_Area"] = dataset[features_wilderness].apply(np.argmax, axis=1)

    dataset["Wilderness_Area"] = dataset["Wilderness_Area"].apply(lambda x: x.split("Wilderness_Area")[-1]).astype(int)

    #dataset = dataset.drop(features_wilderness, axis=1)

    dataset["Hillshade_1"] = (dataset.Hillshade_Noon * dataset.Hillshade_9am)

    dataset["Hillshade_1_sqrt"] = np.sqrt(dataset["Hillshade_1"])

    dataset["Hillshade_2"] = (dataset.Hillshade_3pm * dataset.Hillshade_9am)

    dataset["Hillshade_2_sqrt"] = np.sqrt(dataset["Hillshade_2"])

    dataset["Hillshade_3"] = (dataset.Hillshade_3pm * dataset.Hillshade_Noon * dataset.Hillshade_9am)

    dataset["Hillshade_3_sqrt"] = np.sqrt(dataset["Hillshade_3"])

    dataset.Hillshade_1 = dataset.Hillshade_1.astype(float)

    dataset["DistanceToHydrology"] = np.sqrt(dataset.Horizontal_Distance_To_Hydrology ** 2 + dataset.Vertical_Distance_To_Hydrology ** 2)

    dataset["Horizontal_Distance_To_Roadways_sqrt"]= np.sqrt(dataset["Horizontal_Distance_To_Roadways"])

    dataset["Horizontal_Distance_To_Fire_Points_sqrt"]= np.sqrt(dataset["Horizontal_Distance_To_Fire_Points"])

    dataset["Slope_sqrt"] = np.sqrt(dataset["Slope"])

    dataset["Hillshade_9am_cube"] = dataset["Hillshade_9am"].apply(lambda x: x**3)

    dataset["Hillshade_Noon_cube"] = dataset["Hillshade_Noon"].apply(lambda x: x**3)

    dataset["Aspect_Slope_cbrt"] = np.cbrt(dataset.Aspect * dataset.Slope)

    dataset["Elevation_Slope_sqrt"] = np.sqrt(dataset.Elevation * dataset.Slope)

    dataset["Elevation_Aspect_sqrt"] = np.sqrt(dataset.Elevation * dataset.Aspect)

    dataset["Elevation_sqrt"] = np.sqrt(dataset.Elevation)

    return dataset
data_train = pd.read_csv("../input/train.csv")

final_train = clear_dataset(data_train)
x_data = final_train.drop(["Cover_Type", "Id"], axis=1)

y_data = final_train["Cover_Type"]
from sklearn.decomposition import PCA, TruncatedSVD

pca = PCA(n_components = 2 )  # whitten = normalize

x_pca = pca.fit_transform(x_data)

print("variance ratio: ", pca.explained_variance_ratio_)

print("sum: ",sum(pca.explained_variance_ratio_))
df = pd.DataFrame(np.stack([x_pca[:,0], x_pca[:,1], y_data], axis=1), columns=["p1", "p2", "Cover_Type"])

color = ["blue", "green", "purple", "yellow", "red", "orange", "cyan"]

plt.figure(1,figsize=(9,6))

for each in y_data.unique():

    plt.scatter(df.p1[df.Cover_Type == each],df.p2[df.Cover_Type == each],color = color[each - 1],label = each, alpha=0.5)

plt.legend()

plt.show()
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.25, random_state=42)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report



def get_metrics(y_test, y_predicted):  

    # true positives / (true positives+false positives)

    precision = precision_score(y_test, y_predicted, pos_label=None,

                                    average='weighted')             

    # true positives / (true positives + false negatives)

    recall = recall_score(y_test, y_predicted, pos_label=None,

                              average='weighted')

    

    # harmonic mean of precision and recall

    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')

    

    # true positives + true negatives/ total

    accuracy = accuracy_score(y_test, y_predicted)

    return accuracy, precision, recall, f1
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(n_estimators=100, max_depth=19, max_features=11,n_jobs=-1, random_state=42)

clf.fit(x_train, y_train)



y_predicted = clf.predict(x_val)
accuracy, precision, recall, f1 = get_metrics(y_val, y_predicted)

print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
from sklearn.metrics import confusion_matrix

import itertools

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
cm = confusion_matrix(y_val, y_predicted)

plt.figure(1, figsize=(9,6))

plot_confusion_matrix(cm, classes = set(y_data.unique()))

plt.show()
def plot_feature_importances(clf, X_train, y_train=None, 

                             top_n=10, figsize=(8,8), print_table=False, title="Feature Importances"):

    '''

    plot feature importances of a tree-based sklearn estimator

    

    Note: X_train and y_train are pandas DataFrames

    

    Note: Scikit-plot is a lovely package but I sometimes have issues

              1. flexibility/extendibility

              2. complicated models/datasets

          But for many situations Scikit-plot is the way to go

          see https://scikit-plot.readthedocs.io/en/latest/Quickstart.html

    

    Parameters

    ----------

        clf         (sklearn estimator) if not fitted, this routine will fit it

        

        X_train     (pandas DataFrame)

        

        y_train     (pandas DataFrame)  optional

                                        required only if clf has not already been fitted 

        

        top_n       (int)               Plot the top_n most-important features

                                        Default: 10

                                        

        figsize     ((int,int))         The physical size of the plot

                                        Default: (8,8)

        

        print_table (boolean)           If True, print out the table of feature importances

                                        Default: False

        

    Returns

    -------

        the pandas dataframe with the features and their importance

        

    Author

    ------

        George Fisher

    '''

    

    __name__ = "plot_feature_importances"

    

    import pandas as pd

    import numpy  as np

    import matplotlib.pyplot as plt

    

    from xgboost.core     import XGBoostError

    from lightgbm.sklearn import LightGBMError

    

    try: 

        if not hasattr(clf, 'feature_importances_'):

            clf.fit(X_train.values, y_train.values.ravel())



            if not hasattr(clf, 'feature_importances_'):

                raise AttributeError("{} does not have feature_importances_ attribute".

                                    format(clf.__class__.__name__))

                

    except (XGBoostError, LightGBMError, ValueError):

        clf.fit(X_train.values, y_train.values.ravel())

            

    feat_imp = pd.DataFrame({'importance':clf.feature_importances_})    

    feat_imp['feature'] = X_train.columns

    feat_imp.sort_values(by='importance', ascending=False, inplace=True)

    feat_imp = feat_imp.iloc[:top_n]

    

    feat_imp.sort_values(by='importance', inplace=True)

    feat_imp = feat_imp.set_index('feature', drop=True)

    feat_imp.plot.barh(title=title, figsize=figsize)

    plt.xlabel('Feature Importance Score')

    plt.show()

    

    if print_table:

        from IPython.display import display

        print("Top {} features in descending order of importance".format(top_n))

        display(feat_imp.sort_values(by='importance', ascending=False))

        

    return feat_imp
a = plot_feature_importances(clf, x_train, y_train, top_n=x_train.shape[1], title=clf.__class__.__name__)
from xgboost import XGBClassifier

clf = XGBClassifier(n_estimators=200, learning_rate=0.3, max_depth=3,n_jobs=-1, seed=42, objective="multi:softmax")

clf.fit(x_train, y_train)

y_predicted = clf.predict(x_val)

accuracy, precision, recall, f1 = get_metrics(y_val, y_predicted)

print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
cm = confusion_matrix(y_val, y_predicted)

plt.figure(1, figsize=(9,6))

plot_confusion_matrix(cm, classes = set(y_data.unique()))

plt.show()
a = plot_feature_importances(clf, x_train, y_train, top_n=x_train.shape[1], title=clf.__class__.__name__)
from lightgbm import LGBMClassifier

clf = LGBMClassifier(n_estimators=200, learning_rate=0.3, max_depth=3,n_jobs=-1, seed=42, objective="multi:softmax")

clf.fit(x_train, y_train)

y_predicted = clf.predict(x_val)

accuracy, precision, recall, f1 = get_metrics(y_val, y_predicted)

print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
cm = confusion_matrix(y_val, y_predicted)

plt.figure(1, figsize=(9,6))

plot_confusion_matrix(cm, classes = set(y_data.unique()))

plt.show()
a = plot_feature_importances(clf, x_train, y_train, top_n=x_train.shape[1], title=clf.__class__.__name__)
data_test = pd.read_csv("../input/test.csv")

final_test = clear_dataset(data_test)
clf = RandomForestClassifier(n_estimators=100,max_depth=11, max_features=21,min_samples_leaf=0.001,criterion="entropy",n_jobs=-1, random_state=42)

clf.fit(x_train, y_train)

y_predicted = clf.predict(x_val)

accuracy, precision, recall, f1 = get_metrics(y_val, y_predicted)

print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
y_predicted = clf.predict(x_train)

accuracy, precision, recall, f1 = get_metrics(y_train, y_predicted)

print("train accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
clf = RandomForestClassifier(n_estimators=100,max_depth=11, max_features=21,min_samples_leaf=0.001,criterion="entropy",n_jobs=-1, random_state=42)

clf.fit(x_data, y_data)

test_preds = clf.predict(final_test.drop(["Id"], axis=1))

output = pd.DataFrame({'Id': data_test.Id,

                       'Cover_Type': test_preds})

output.to_csv('rf_submission.csv', index=False)