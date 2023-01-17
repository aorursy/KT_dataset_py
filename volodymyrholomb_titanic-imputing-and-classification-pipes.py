import numpy as np

import pandas as pd



from sklearn.ensemble import RandomForestRegressor

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import FunctionTransformer

from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV



from xgboost import XGBClassifier
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")
train.head()
train.isna().sum()
train["Pclass"] = train["Pclass"].astype(str)
train.pivot_table(values="Survived",

                  index="Sex",

                  columns="Pclass",

                  margins=True)
train_exp = (train

             .copy()

             .assign(Title=train["Name"]

                     .astype(str)

                     .str.extract(r"\,\s([a-zA-Z\s]*)\.\s")))
# (train_exp

# .groupby(["Title", "Sex"])["PassengerId"]

# .count()

# .unstack()

# .fillna(0))
(train_exp

 .pivot_table(values="PassengerId",

              index="Title",

              columns="Sex",

              aggfunc="count",

              fill_value=0))
features_subset = ["Pclass",

                   "Name",

                   "Sex",

                   "Age",

                   "SibSp",

                   "Parch",

                   "Embarked"]



train_subset = train.loc[~train["Age"].isna(), features_subset]



y_subset = train_subset["Age"]

X_subset = train_subset.drop(["Age"], axis=1)



model_subset = RandomForestRegressor(random_state=1234)





def parse_name(df):

    return (df.assign(Title=df["Name"]

                      .astype(str)

                      .str.extract(r"\,\s([a-zA-Z\s]*)\.\s")))





def family_counter(df):

    return (df.assign(w_family=df.apply(

        lambda r: int(r["SibSp"] + r["Parch"] > 0),

        axis=1)))





construct_features = Pipeline(

    steps=[

        ("add_title", FunctionTransformer(parse_name)),

        ("add_family", FunctionTransformer(family_counter))])



preprocess_cats = Pipeline(

    steps=[

        ("imputing", SimpleImputer(strategy="most_frequent")),

        ("encoding", OneHotEncoder(handle_unknown="ignore"))])



pre_processor = ColumnTransformer(

    transformers=[

        ("drop_cols", "drop", ["Name", "SibSp", "Parch"]),

        ("process_cats",

         preprocess_cats,

         ["Pclass", "Sex", "Embarked", "Title"])])



filling_pipe = Pipeline(

    steps=[

        ("f_engineering", construct_features),

        ("preprocessing", pre_processor),

        ("modeling", model_subset)])



rmse_subset = -1 * cross_val_score(filling_pipe,

                                   X_subset,

                                   y_subset,

                                   scoring="neg_root_mean_squared_error")



print("RMSE: {} ({})".format(rmse_subset.mean(),

                             rmse_subset.std()))
missed_subset = (train

                 .loc[train["Age"].isna(), features_subset]

                 .drop(["Age"], axis=1))



filling_pipe.fit(X_subset, y_subset)



train.loc[train["Age"].isna(), "Age"] = (filling_pipe

                                         .predict(missed_subset))
train["Age_desc"], age_bins = pd.qcut(train["Age"],

                                      q=6,

                                      precision=0,

                                      retbins=True)
features_class = ["Survived",

                  "Pclass",

                  "Name",

                  "Sex",

                  "Age_desc",

                  "SibSp",

                  "Parch",

                  "Embarked"]



train_class = train[features_class]



y_class = train_class["Survived"]

X_class = train_class.drop(["Survived"], axis=1)



y_count = train["Survived"].value_counts()

spw = y_count[0] / y_count[1]



model_class = XGBClassifier(random_state=1234,

                            objective="binary:logistic",

                            scale_pos_weight=spw,

                            njobs=-1)



pre_processor_class = ColumnTransformer(

    transformers=[

        ("drop_cols", "drop", ["Name", "SibSp", "Parch"]),

        ("process_cats",

         preprocess_cats,

         ["Pclass", "Sex", "Age_desc", "Embarked", "Title"])])



class_pipe = Pipeline(

    steps=[

        ("f_engineering", construct_features),

        ("preprocessing", pre_processor_class),

        ("xgbr", model_class)])
param_grid = {"xgbr__n_estimators": [500, 750, 1000],

              "xgbr__learning_rate": [0.001, 0.01, 0.1]}



search_cv = GridSearchCV(class_pipe,

                         param_grid=param_grid,

                         scoring="roc_auc",

                         n_jobs=-1)

search_cv.fit(X_class,

              y_class)



print("Best roc_auc on CV: {}:".format(search_cv.best_score_))

print(search_cv.best_params_)
class_pipe.set_params(xgbr__n_estimators=750,

                      xgbr__learning_rate=0.001)



class_pipe.fit(X_class,

               y_class)
test["Age_desc"] = pd.cut(test["Age"],

                          bins=age_bins,

                          precision=0)



predicts = class_pipe.predict(test[["Pclass",

                                    "Name",

                                    "Sex",

                                    "Age_desc",

                                    "SibSp",

                                    "Parch",

                                    "Embarked"]])



my_submission = pd.DataFrame(

    np.column_stack((test["PassengerId"], predicts)),

    columns=["PassengerId", "Survived"])



my_submission.to_csv("output.csv", index=False)