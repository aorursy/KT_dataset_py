import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from sklearn.preprocessing import MinMaxScaler

!pip install lime

from lime.lime_text import LimeTextExplainer

import string

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv("/Users/valentinvivant/Desktop/Hakathon_03_2020/Data/train_set.csv")

df_test = pd.read_csv("/Users/valentinvivant/Desktop/Hakathon_03_2020/Data/test_set.csv")
df.head(2)
df = df.dropna(subset=["review"])
df_train, df_val = train_test_split(

    df, 

    test_size=0.2, 

    random_state=0, 

    stratify=df['binary_target']

)
def convert_text_to_lowercase(df, colname):

    df[colname] = df[colname].str.lower()

    return df

    



def text_cleaning(df, colname):

    """

    Takes in a string of text, then performs the following:

    1. convert text to lowercase

    2. ??

    """

    df = (

        df

        .pipe(convert_text_to_lowercase, colname)

    )

    return df
df_train_clean = text_cleaning(df_train, 'review')

df_val_clean = text_cleaning(df_val, 'review')
Count_Vectorizer = CountVectorizer(max_features=20000)



logit = LogisticRegression(random_state=0)



pipeline = Pipeline([

    ('vectorizer', Count_Vectorizer),

    ('model', logit)])
x_train = df_train_clean['review']

y_train = df_train_clean['binary_target']



x_val = df_val_clean['review']

y_val = df_val_clean['binary_target']
pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_val)
print(confusion_matrix(y_val, y_pred))

print(classification_report(y_val, y_pred))
df_test_cleaned = text_cleaning(df_test, 'review')

x_test = df_test_cleaned['review']



predictions = pipeline.predict(x_test)
soumission = pd.DataFrame({"review_id": df_test['review_id'], "prediction": predictions})



soumission['prediction'] = soumission['prediction'].astype('bool')

soumission['review_id'] = soumission['review_id'].astype('str')



soumission.head().dtypes



soumission.to_csv('/Users/valentinvivant/Desktop/soumission.csv', index=False)
class_names = [0, 1]

explainer = LimeTextExplainer(class_names=class_names)



def lime_model_interpreter(clf, idx, n_features):

    text_idx = x_val.iloc[idx]

    target_idx = y_val.iloc[idx]



    exp = explainer.explain_instance(text_idx, clf.predict_proba, num_features=n_features)

    print('Document id: %d' % idx)

    print('Probability(True) =', clf.predict_proba([text_idx])[0,1])

    print('True class: %s' % class_names[target_idx])



    exp.show_in_notebook(text=True)
lime_model_interpreter(pipeline, 2, n_features=6) 
lime_model_interpreter(pipeline, 7, n_features=6) 