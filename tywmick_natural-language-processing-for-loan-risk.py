import joblib



prev_notebook_folder = "../input/building-a-neural-network-to-predict-loan-risk/"

loans = joblib.load(prev_notebook_folder + "loans_for_nlp.joblib")

num_loans = loans.shape[0]

print(f"This dataset includes {num_loans:,} loans.")
loans.head()
nlp_cols = ["title", "desc"]



loans[nlp_cols].describe()
import re

import numpy as np



for col in nlp_cols:

    replace_empties = lambda x: x if re.search("\S", x) else np.NaN

    loans[col] = loans[col].map(replace_empties, na_action="ignore")



description = loans[nlp_cols].describe()

description
for col in nlp_cols:

    percentage = int(description.at["count", col] / num_loans * 100)

    print(f"`{col}` is used in {percentage}% of loans in the dataset.")



percentage = int(description.at["freq", "title"] / num_loans * 100)

print(f'The title "Debt consolidation" is used in {percentage}% of loans.')
# `issue_d` is just the month and year the loan was issued, by the way.

loans["issue_d"] = loans["issue_d"].astype("datetime64[ns]")



print("Total date range:")

print(loans["issue_d"].agg(["min", "max"]))

print("\n`title` date range:")

print(loans[["title", "issue_d"]].dropna(axis="index")["issue_d"].agg(["min", "max"]))

print("\n`desc` date range:")

print(loans[["desc", "issue_d"]].dropna(axis="index")["issue_d"].agg(["min", "max"]))
import pandas as pd



with pd.option_context("display.min_rows", 50):

    print(loans["title"].value_counts())
loans[loans["title"] == "up up down down left right left right ba"][

    ["loan_amnt", "title", "issue_d"]

]
loans["desc"].value_counts()
pattern = "^\s*Borrower added on \d\d/\d\d/\d\d > "

prefix_count = (

    loans["desc"]

    .map(lambda x: True if re.search(pattern, x, re.I) else None, na_action="ignore")

    .count()

)

print(

    f"{prefix_count:,} loan descriptions begin with that pattern.",

    f"({description.loc['count', 'desc'] - prefix_count:,} do not.)",

)
other_desc_map = loans["desc"].map(

    lambda x: False if pd.isna(x) or re.search(pattern, x, re.I) else True

)

other_descs = loans["desc"][other_desc_map]

other_descs.value_counts()
from datetime import datetime, date



for row in loans[["desc", "issue_d"]].itertuples():

    if not pd.isna(row.desc):

        month_after_issue = date(

            day=row.issue_d.day,

            month=row.issue_d.month % 12 + 1,

            year=row.issue_d.year + row.issue_d.month // 12,

        )



        date_strings = re.findall("\d\d/\d\d/\d\d", row.desc)

        dates = []

        for string in date_strings:

            try:

                dates.append(datetime.strptime(string, "%m/%d/%y").date())

            except:

                continue



        for d in dates:

            if d >= month_after_issue:

                print(f"{row.issue_d} – {row.desc}")

                break
def clean_desc(desc):

    if pd.isna(desc):

        return desc

    else:

        return re.sub(

            "\s*Borrower added on \d\d/\d\d/\d\d > |<br>", lambda x: " ", desc

        ).strip()





loans["desc"] = loans["desc"].map(clean_desc)
loans["title"].fillna(

    loans["purpose"].map(lambda x: x.replace("_", " ").capitalize()), inplace=True

)
from pandas.api.types import CategoricalDtype





loans = loans.drop(columns=["issue_d"])



float_cols = ["annual_inc", "dti", "inv_mths_since_last_delinq",

    "inv_mths_since_last_record", "revol_util", "inv_mths_since_last_major_derog",

    "annual_inc_joint", "dti_joint", "bc_util", "inv_mo_sin_rcnt_rev_tl_op",

    "inv_mo_sin_rcnt_tl", "inv_mths_since_recent_bc", "inv_mths_since_recent_bc_dlq",

    "inv_mths_since_recent_inq", "inv_mths_since_recent_revol_delinq", "pct_tl_nvr_dlq",

    "percent_bc_gt_75", "fraction_recovered"]

int_cols = ["loan_amnt", "delinq_2yrs", "cr_hist_age_mths", "fico_range_low",

    "fico_range_high", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal",

    "total_acc", "collections_12_mths_ex_med", "acc_now_delinq", "tot_coll_amt",

    "tot_cur_bal", "total_rev_hi_lim", "acc_open_past_24mths", "avg_cur_bal",

    "bc_open_to_buy", "chargeoff_within_12_mths", "delinq_amnt", "mo_sin_old_il_acct",

    "mo_sin_old_rev_tl_op", "mort_acc", "num_accts_ever_120_pd", "num_actv_bc_tl",

    "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl", "num_op_rev_tl",

    "num_rev_accts", "num_rev_tl_bal_gt_0", "num_sats", "num_tl_120dpd_2m",

    "num_tl_30dpd", "num_tl_90g_dpd_24m", "num_tl_op_past_12m", "pub_rec_bankruptcies",

    "tax_liens", "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",

    "total_il_high_credit_limit"]

ordinal_cols = ["emp_length"]

category_cols = ["term", "home_ownership", "purpose", "application_type"]

text_cols = nlp_cols



size_metrics = pd.DataFrame(

    {

        "previous_dtype": loans.dtypes,

        "previous_size": loans.memory_usage(index=False, deep=True),

    }

)

previous_size = loans.memory_usage(deep=True).sum()





for col_name in float_cols:

    loans[col_name] = pd.to_numeric(loans[col_name], downcast="float")



for col_name in int_cols:

    loans[col_name] = pd.to_numeric(loans[col_name], downcast="unsigned")



emp_length_categories = ["< 1 year", "1 year", "2 years", "3 years", "4 years",

    "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"]

emp_length_type = CategoricalDtype(categories=emp_length_categories, ordered=True)

loans["emp_length"] = loans["emp_length"].astype(emp_length_type)



for col_name in category_cols:

    loans[col_name] = loans[col_name].astype("category")





current_size = loans.memory_usage(deep=True).sum()

reduction = (previous_size - current_size) / previous_size

print(f"Reduced DataFrame size in memory by {int(reduction * 100)}%.")



size_metrics["current_dtype"] = loans.dtypes

size_metrics["current_size"] = loans.memory_usage(index=False, deep=True)

pd.options.display.max_rows = 100

size_metrics
import spacy

from sklearn.preprocessing import FunctionTransformer





def get_doc_vectors(X):

    n_cols = X.shape[1]

    nlp = spacy.load("en_core_web_lg", disable=["tagger", "parser", "ner"])

    

    result = []

    for row in X:

        result_row = []

        for i in range(n_cols):

            result_row.append(nlp(row[i]).vector)

            

        result.append(np.concatenate(result_row))

        

    return np.array(result)





vectorizer = FunctionTransformer(get_doc_vectors)
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer, make_column_selector

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

from pathlib import Path





def generate_cat_encoder(col_name):

    categories = list(loans[col_name].cat.categories)

    if loans[col_name].cat.ordered:

        return (

            col_name,

            OrdinalEncoder(categories=[categories], dtype=np.uint8),

            [col_name],

        )

    else:

        return (

            col_name,

            OneHotEncoder(categories=[categories], drop="if_binary", dtype=np.bool_),

            [col_name],

        )





Path("../tmp/transformer_cache").mkdir(parents=True, exist_ok=True)

transformer = ColumnTransformer(

    [

        (

            "nlp_cols",

            Pipeline(

                [

                    (

                        "nlp_imputer",

                        SimpleImputer(strategy="constant", fill_value=""),

                    ),

                    ("nlp_vectorizer", vectorizer),

                    ("nlp_scaler", StandardScaler(with_mean=False)),

                ],

                verbose=True,

            ),

            make_column_selector("^(title|desc)$"),

        ),

    ]

    + [generate_cat_encoder(col_name) for col_name in ordinal_cols + category_cols],

    remainder=StandardScaler(),

    verbose=True,

)
import tensorflow as tf

from tensorflow.keras import Sequential, Input

from tensorflow.keras.layers import Dense, Dropout

from sklearn.model_selection import train_test_split

from tqdm import tqdm



np.random.seed(0)

tf.random.set_seed(1)





class ProgressBar(tf.keras.callbacks.Callback):

    def __init__(self, epochs=100):

        self.epochs = epochs

    

    def on_train_begin(self, logs=None):

        self.progress_bar = tqdm(desc="Training model", total=self.epochs, unit="epoch")



    def on_epoch_end(self, epoch, logs=None):

        self.progress_bar.update()



    def on_train_end(self, logs=None):

        self.progress_bar.close()





class FinalMetrics(tf.keras.callbacks.Callback):

    def on_train_end(self, logs=None):

        metrics_msg = "Final metrics:"

        for metric, value in logs.items():

            metrics_msg += f" {metric}: {value:.5f} -"

        metrics_msg = metrics_msg[:-2]

        print(metrics_msg)





def run_pipeline(X, y, transformer, validate=True):

    X_train, X_val, y_train, y_val = (

        train_test_split(X, y, test_size=0.2, random_state=2)

        if validate

        else (X, None, y, None)

    )



    X_train_t = transformer.fit_transform(X_train)

    X_val_t = transformer.transform(X_val) if validate else None



    model = Sequential()

    model.add(Input((X_train_t.shape[1],)))

    model.add(Dense(64, activation="relu"))

    model.add(Dropout(0.3))

    model.add(Dense(32, activation="relu"))

    model.add(Dropout(0.3))

    model.add(Dense(16, activation="relu"))

    model.add(Dropout(0.3))

    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_logarithmic_error")



    history = model.fit(

        X_train_t,

        y_train,

        validation_data=(X_val_t, y_val) if validate else None,

        batch_size=128,

        epochs=100,

        verbose=0,

        callbacks=[ProgressBar(), FinalMetrics()],

    )

    

    return history.history, model, transformer
import dill



history_1, _, _ = run_pipeline(

    loans.drop(columns="fraction_recovered").copy(),

    loans["fraction_recovered"],

    transformer,

)



Path("save_points").mkdir(exist_ok=True)

dill.dump_session("save_points/model_1.pkl")
# Restore save point if needed

import dill



try:

    history_1

except NameError:

    save_points_folder = "../input/improving-loan-risk-prediction-with-nlp/save_points/"

    dill.load_session(save_points_folder + "model_1.pkl")
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline





def plot_loss_metrics(history, model_num=None):

    for metric, values in history.items():

        sns.lineplot(x=range(len(values)), y=values, label=metric)

    plt.xlabel("epoch")

    plt.title(

        f"Model {f'{model_num} ' if model_num else ''} loss metrics during training"

    )

    plt.show()





plot_loss_metrics(history_1, "1")
history_2, _, _ = run_pipeline(

    loans.drop(columns=["fraction_recovered", "desc"]).copy(),

    loans["fraction_recovered"],

    transformer,

)



Path("save_points").mkdir(exist_ok=True)

dill.dump_session("save_points/model_2.pkl")
# Restore save point if needed

import dill



try:

    history_2

except NameError:

    save_points_folder = "../input/improving-loan-risk-prediction-with-nlp/save_points/"

    dill.load_session(save_points_folder + "model_2.pkl")
plot_loss_metrics(history_2, "2")