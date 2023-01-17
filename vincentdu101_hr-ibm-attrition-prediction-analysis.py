# HR Employee Attrition Predictor

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import graphviz

import seaborn as sns

import sklearn.metrics as metrics

import plotly.graph_objs as go



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

 

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.impute import SimpleImputer

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import label_binarize

from sklearn.compose import ColumnTransformer, make_column_transformer

from matplotlib.colors import ListedColormap

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
def graphROCCurve(y_test, y_pred):    

    fpr = {}

    tpr = {}

    roc_auc = {}

    

    # Compute micro-average ROC curve and ROC area

    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred, pos_label="Yes")

    roc_auc = metrics.auc(fpr, tpr)

    

    plt.figure()

    lw = 2

    plt.plot(fpr, tpr, color='darkorange',

             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic')

    plt.legend(loc="lower right")

    plt.show()
def graphFeaturesImportant(rf_classifier, features):

    trace = go.Scatter(

        y = features, 

        x = dataset.columns.values, mode = "markers",

        marker = dict(

            sizemode = "diameter", sizeref=1, size=13, 

            color=features, colorscale="Portland",

            showscale=True

        ),

        text = dataset.columns.values

    )

    data = [trace]



    layout = go.Layout(

        autosize = True,

        title = "Random Forest Feature Importance",

        hovermode = "closest",

        xaxis = dict(

            ticklen=5, showgrid=True, zeroline=True, showline=True

        ),

        yaxis = dict(

            title="Feature Importance", showgrid=True, zeroline=True,

            ticklen=5, gridwidth=2

        ),

        showlegend=False

    )

    

    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig)
# developing the Multi Layer Perceptron Neural Network

def creatingNeuralNetworkPredictor(X_train, y_train, X_test, y_test, preprocess):

    print("\nNeural Network Classifier Section")

    print("---------------------------------")

    

    # initialize the Multi Layer Perceptron Neural Network 

    mlp_classifier = MLPClassifier(solver="adam", alpha=1e-5, max_iter=500,

                               hidden_layer_sizes=(13, 13, 13))

    

    # hook up the preprocess step with the classifier params and create the pipeline

    model = make_pipeline(preprocess, mlp_classifier)

    

    # fitting the Multi Layer Perceptron to the training set

    model.fit(X_train, y_train)

    

    print("Training set Score: ", model.score(X_train, y_train))

    print("Testing set Score: ", model.score(X_test, y_test))    

    

    return model
# developing the Random Forest Classifier

def creatingRandomForestPredictor(X_train, y_train, X_test, y_test, preprocess):

    print("\nRandom Forest Classifier Section")

    print("---------------------------------")

    

    # initialize the Multi Layer Perceptron Neural Network 

    random_forest_classifier = RandomForestClassifier(**{'n_jobs': -1,

        'n_estimators': 800

    })

    

    # hook up the preprocess step with the classifier params and create the pipeline

    model = make_pipeline(preprocess, random_forest_classifier)

    

    # fitting Random Forest to the training set

    model.fit(X_train, y_train)

    

    print("Training set Score: ", model.score(X_train, y_train))

    print("Testing set Score: ", model.score(X_test, y_test))    

    

    return model
# importing the data

dataset = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")

dataset = dataset.drop(["YearsWithCurrManager"], axis=1)

dataset.head()
# using ColumnTransformer only approach

transformed_data = dataset.loc[:, dataset.columns != "Attrition"]

X = transformed_data.values

y = dataset.Attrition.values



numerical_features = transformed_data.dtypes == "int64"

categorical_features = ~numerical_features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)



preprocess = make_column_transformer(

    (OneHotEncoder(), categorical_features),

    (make_pipeline(SimpleImputer(), StandardScaler()), numerical_features)

)
# outputting data summary

print("Summary Info About the Dataset")

print("Does category contain null values?")

print(dataset.isnull().any(), "\n")

print("Said Yes to Attrition: ", y[(y == "Yes")].size)

print("Said No to Attrition:  ", y[(y == "No")].size)

print("Total responses:       ", y.size)
nn_model = creatingNeuralNetworkPredictor(X_train, y_train, X_test, y_test, preprocess)
# Predicting the Test set results

nn_y_pred = nn_model.predict(X_test)



# output results

nn_y_pred_prob = nn_model.predict_proba(X_test)[:, 1]



print("Accuracy Score of Prediction : ", metrics.accuracy_score(y_test, nn_y_pred) * 100)

print("\nClassification Report")

print(metrics.classification_report(y_test, nn_y_pred))

print("Zero One Loss: ", metrics.zero_one_loss(y_test, nn_y_pred))

print("Log Loss:      ", metrics.log_loss(y_test, nn_y_pred_prob))
graphROCCurve(y_test, nn_y_pred_prob)

print("ROC AUC Score: ", metrics.roc_auc_score(y_test, nn_y_pred_prob, average="macro"))
nn_cm = metrics.confusion_matrix(y_test, nn_y_pred)

sns.heatmap(nn_cm)

print("\nConfusion Matrix")

print(pd.crosstab(y_test.ravel(), nn_y_pred.ravel(), rownames=['True'], colnames=['Predicted'], margins=True))

plt.show()
rf_model = creatingRandomForestPredictor(X_train, y_train, X_test, y_test, preprocess)


# Predicting the Test set results

rf_y_pred = rf_model.predict(X_test)



# output results

rf_y_pred_prob = rf_model.predict_proba(X_test)[:, 1]

print("Accuracy Score of Prediction : ", metrics.accuracy_score(y_test, rf_y_pred) * 100)

print("\nClassification Report")

print(metrics.classification_report(y_test, rf_y_pred))

print("Zero One Loss: ", metrics.zero_one_loss(y_test, rf_y_pred))

print("Log Loss:      ", metrics.log_loss(y_test, rf_y_pred_prob))
graphROCCurve(y_test, rf_y_pred_prob)

print("ROC AUC Score: ", metrics.roc_auc_score(y_test, rf_y_pred_prob, average="macro"))
rf_cm = metrics.confusion_matrix(y_test, rf_y_pred)

print("\nConfusion Matrix")

print(pd.crosstab(y_test.ravel(), rf_y_pred.ravel(), rownames=['True'], colnames=['Predicted'], margins=True))

sns.heatmap(rf_cm, center=True)

plt.show()