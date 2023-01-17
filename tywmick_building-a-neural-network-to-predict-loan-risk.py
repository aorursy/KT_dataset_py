import pandas as pd



loans_raw = pd.read_csv(

    "../input/lending-club/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv",

    low_memory=False,

)



loans_raw.shape
dictionary_df = pd.read_excel("https://resources.lendingclub.com/LCDataDictionary.xlsx")



# Drop blank rows, strip white space, convert to Python dictionary, fix one key name

dictionary_df.dropna(axis="index", inplace=True)

dictionary_df = dictionary_df.applymap(lambda x: x.strip())

dictionary_df.set_index("LoanStatNew", inplace=True)

dictionary = dictionary_df["Description"].to_dict()

dictionary["verification_status_joint"] = dictionary.pop("verified_status_joint")



# Print in order of dataset columns (which makes more sense than dictionary's order)

for col in loans_raw.columns:

    print(f"•{col}: {dictionary[col]}")



# Hiding the output because it's quite a few lines, but feel free to take a peek by

# clicking the "Output" button
cols_for_output = ["term", "installment", "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee"]
loans_raw["emp_title"].nunique()
cols_to_drop = ["id", "member_id", "funded_amnt", "funded_amnt_inv", "int_rate", "grade", "sub_grade", "emp_title", "pymnt_plan", "url", "desc", "title", "zip_code", "addr_state", "initial_list_status", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low", "policy_code", "hardship_flag", "hardship_type", "hardship_reason", "hardship_status", "deferral_term", "hardship_amount", "hardship_start_date", "hardship_end_date", "payment_plan_start_date", "hardship_length", "hardship_dpd", "hardship_loan_status", "orig_projected_additional_accrued_interest", "hardship_payoff_balance_amount", "hardship_last_payment_amount", "disbursement_method", "debt_settlement_flag", "debt_settlement_flag_date", "settlement_status", "settlement_date", "settlement_amount", "settlement_percentage", "settlement_term"]



loans = loans_raw.drop(columns=cols_to_drop)
loans.groupby("loan_status")["loan_status"].count()
credit_policy = "Does not meet the credit policy. Status:"

len_credit_policy = len(credit_policy)

remove_credit_policy = (

    lambda status: status[len_credit_policy:]

    if credit_policy in str(status)

    else status

)

loans["loan_status"] = loans["loan_status"].map(remove_credit_policy)



rows_to_drop = loans[

    (loans["loan_status"] != "Charged Off") & (loans["loan_status"] != "Fully Paid")

].index

loans.drop(index=rows_to_drop, inplace=True)



loans.groupby("loan_status")["loan_status"].count()
loans[cols_for_output].info()
loans.groupby("term")["term"].count()
onehot_cols = ["term"]



loans["term"] = loans["term"].map(lambda term_str: term_str.strip())



extract_num = lambda term_str: float(term_str[:2])

loans["term_num"] = loans["term"].map(extract_num)

cols_for_output.remove("term")

cols_for_output.append("term_num")
received = (

    loans["total_rec_prncp"]

    + loans["total_rec_int"]

    + loans["total_rec_late_fee"]

    + loans["recoveries"]

    - loans["collection_recovery_fee"]

)

expected = loans["installment"] * loans["term_num"]

loans["fraction_recovered"] = received / expected



loans.groupby("loan_status")["fraction_recovered"].describe()
import numpy as np



loans["fraction_recovered"] = np.where(

    (loans["loan_status"] == "Fully Paid") | (loans["fraction_recovered"] > 1.0),

    1.0,

    loans["fraction_recovered"],

)

loans.groupby("loan_status")["fraction_recovered"].describe()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



sns.kdeplot(

    data=loans["fraction_recovered"][loans["loan_status"] == "Charged Off"],

    label="Charged Off",

    shade=True,

)

plt.axis(xmin=0, xmax=1)

plt.title('Distribution of "fraction recovered"')

plt.show()
loans.drop(columns=cols_for_output, inplace=True)

loans.info(verbose=True, null_counts=True)
negative_mark_cols = ["mths_since_last_delinq", "mths_since_last_record", "mths_since_last_major_derog", "mths_since_recent_bc_dlq", "mths_since_recent_inq", "mths_since_recent_revol_delinq", "mths_since_recent_revol_delinq", "sec_app_mths_since_last_major_derog"]

joint_cols = ["annual_inc_joint", "dti_joint", "verification_status_joint", "revol_bal_joint", "sec_app_fico_range_low", "sec_app_fico_range_high", "sec_app_earliest_cr_line", "sec_app_inq_last_6mths", "sec_app_mort_acc", "sec_app_open_acc", "sec_app_revol_util", "sec_app_open_act_il", "sec_app_num_rev_accts", "sec_app_chargeoff_within_12_mths", "sec_app_collections_12_mths_ex_med", "sec_app_mths_since_last_major_derog"]

confusing_cols = ["open_acc_6m", "open_act_il", "open_il_12m", "open_il_24m", "mths_since_rcnt_il", "total_bal_il", "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc", "all_util", "inq_fi", "total_cu_tl", "inq_last_12m"]
loans["issue_d"] = loans["issue_d"].astype("datetime64[ns]")



# Check date range of confusing columns

loans[confusing_cols + ["issue_d"]].dropna(axis="index")["issue_d"].agg(

    ["count", "min", "max"]

)
# Compare to all entries from Dec 2015 onward

loans["issue_d"][loans["issue_d"] >= np.datetime64("2015-12-01")].agg(

    ["count", "min", "max"]

)
new_metric_cols = confusing_cols
mths_since_last_cols = [

    col_name

    for col_name in loans.columns

    if "mths_since" in col_name or "mo_sin_rcnt" in col_name

]

mths_since_old_cols = [

    col_name for col_name in loans.columns if "mo_sin_old" in col_name

]



for col_name in mths_since_last_cols:

    loans[col_name] = [

        0.0 if pd.isna(months) else 1 / 1 if months == 0 else 1 / months

        for months in loans[col_name]

    ]

loans.loc[:, mths_since_old_cols].fillna(0, inplace=True)



# Rename inverse columns

rename_mapper = {}

for col_name in mths_since_last_cols:

    rename_mapper[col_name] = col_name.replace("mths_since", "inv_mths_since").replace(

        "mo_sin_rcnt", "inv_mo_sin_rcnt"

    )

loans.rename(columns=rename_mapper, inplace=True)





def replace_list_value(l, old_value, new_value):

    i = l.index(old_value)

    l.pop(i)

    l.insert(i, new_value)





replace_list_value(new_metric_cols, "mths_since_rcnt_il", "inv_mths_since_rcnt_il")

replace_list_value(

    joint_cols,

    "sec_app_mths_since_last_major_derog",

    "sec_app_inv_mths_since_last_major_derog",

)
loans.groupby("application_type")["application_type"].count()
joint_loans = loans[:][loans["application_type"] == "Joint App"]

joint_loans[joint_cols].info()
joint_new_metric_cols = ["revol_bal_joint", "sec_app_fico_range_low", "sec_app_fico_range_high", "sec_app_earliest_cr_line", "sec_app_inq_last_6mths", "sec_app_mort_acc", "sec_app_open_acc", "sec_app_revol_util", "sec_app_open_act_il", "sec_app_num_rev_accts", "sec_app_chargeoff_within_12_mths", "sec_app_collections_12_mths_ex_med", "sec_app_inv_mths_since_last_major_derog"]

joint_loans[joint_new_metric_cols + ["issue_d"]].dropna(axis="index")["issue_d"].agg(

    ["count", "min", "max"]

)
# Check without `sec_app_revol_util` column

joint_new_metric_cols_2 = ["revol_bal_joint", "sec_app_fico_range_low", "sec_app_fico_range_high", "sec_app_earliest_cr_line", "sec_app_inq_last_6mths", "sec_app_mort_acc", "sec_app_open_acc", "sec_app_open_act_il", "sec_app_num_rev_accts", "sec_app_chargeoff_within_12_mths", "sec_app_collections_12_mths_ex_med", "sec_app_inv_mths_since_last_major_derog"]

joint_loans[joint_new_metric_cols_2 + ["issue_d"]].dropna(axis="index")["issue_d"].agg(

    ["count", "min", "max"]

)
joint_loans["issue_d"].agg(["count", "min", "max"])
onehot_cols.append("application_type")



# Fill joint columns in individual applications

for joint_col, indiv_col in zip(

    ["annual_inc_joint", "dti_joint", "verification_status_joint"],

    ["annual_inc", "dti", "verification_status"],

):

    loans[joint_col] = [

        joint_val if app_type == "Joint App" else indiv_val

        for app_type, joint_val, indiv_val in zip(

            loans["application_type"], loans[joint_col], loans[indiv_col]

        )

    ]



loans.info(verbose=True, null_counts=True)
cols_to_search = [

    col for col in loans.columns if col not in new_metric_cols + joint_new_metric_cols

]

loans.dropna(axis="index", subset=cols_to_search).shape
loans.dropna(axis="index", subset=cols_to_search, inplace=True)
loans[["earliest_cr_line", "sec_app_earliest_cr_line"]]
def get_credit_history_age(col_name):

    earliest_cr_line_date = loans[col_name].astype("datetime64[ns]")

    cr_hist_age_delta = loans["issue_d"] - earliest_cr_line_date

    MINUTES_PER_MONTH = int(365.25 / 12 * 24 * 60)

    cr_hist_age_months = cr_hist_age_delta / np.timedelta64(MINUTES_PER_MONTH, "m")

    return cr_hist_age_months.map(

        lambda value: np.nan if pd.isna(value) else round(value)

    )





cr_hist_age_months = get_credit_history_age("earliest_cr_line")

cr_hist_age_months
loans["earliest_cr_line"] = cr_hist_age_months

loans["sec_app_earliest_cr_line"] = get_credit_history_age(

    "sec_app_earliest_cr_line"

).astype("Int64")

loans.rename(

    columns={

        "earliest_cr_line": "cr_hist_age_mths",

        "sec_app_earliest_cr_line": "sec_app_cr_hist_age_mths",

    },

    inplace=True,

)

replace_list_value(

    joint_new_metric_cols, "sec_app_earliest_cr_line", "sec_app_cr_hist_age_mths"

)
categorical_cols = ["term", "emp_length", "home_ownership", "verification_status", "purpose", "verification_status_joint"]

for i, col_name in enumerate(categorical_cols):

    print(

        loans.groupby(col_name)[col_name].count(),

        "\n" if i < len(categorical_cols) - 1 else "",

    )
loans.drop(

    columns=[

        "verification_status",

        "verification_status_joint",

        "issue_d",

        "loan_status",

    ],

    inplace=True,

)
onehot_cols += ["home_ownership", "purpose"]

ordinal_cols = {

    "emp_length": [

        "< 1 year",

        "1 year",

        "2 years",

        "3 years",

        "4 years",

        "5 years",

        "6 years",

        "7 years",

        "8 years",

        "9 years",

        "10+ years",

    ]

}
loans_1 = loans.drop(columns=new_metric_cols + joint_new_metric_cols)

loans_2 = loans.drop(columns=joint_new_metric_cols)

loans_2.info(verbose=True, null_counts=True)
loans_2["il_util"][loans_2["il_util"].notna()].describe()
query_df = loans[["il_util", "total_bal_il", "total_il_high_credit_limit"]].dropna(

    axis="index", subset=["il_util"]

)

query_df["il_util_compute"] = (

    query_df["total_bal_il"] / query_df["total_il_high_credit_limit"]

).map(lambda x: float(round(x * 100)))

query_df[["il_util", "il_util_compute"]]
(query_df["il_util"] == query_df["il_util_compute"]).describe()
query_df["compute_diff"] = abs(query_df["il_util"] - query_df["il_util_compute"])

query_df["compute_diff"][query_df["compute_diff"] != 0].describe()
loans["il_util_imputed"] = [

    True if pd.isna(util) & pd.notna(bal) & pd.notna(limit) else False

    for util, bal, limit in zip(

        loans["il_util"], loans["total_bal_il"], loans["total_il_high_credit_limit"]

    )

]

new_metric_onehot_cols = ["il_util_imputed"]

loans["il_util"] = [

    0.0

    if pd.isna(util) & pd.notna(bal) & (limit == 0)

    else float(round(bal / limit * 100))

    if pd.isna(util) & pd.notna(bal) & pd.notna(limit)

    else util

    for util, bal, limit in zip(

        loans["il_util"], loans["total_bal_il"], loans["total_il_high_credit_limit"]

    )

]



loans_2 = loans.drop(columns=joint_new_metric_cols)

loans_2.info(verbose=True, null_counts=True)
loans_2.dropna(axis="index", inplace=True)



loans_3 = loans.dropna(axis="index")

loans_3.info(verbose=True, null_counts=True)
from sklearn.model_selection import train_test_split

from sklearn_pandas import DataFrameMapper

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from tensorflow.keras import Sequential, Input

from tensorflow.keras.layers import Dense, Dropout





def run_pipeline(

    data,

    onehot_cols,

    ordinal_cols,

    batch_size,

    validate=True,

):

    X = data.drop(columns=["fraction_recovered"])

    y = data["fraction_recovered"]

    X_train, X_valid, y_train, y_valid = (

        train_test_split(X, y, test_size=0.2, random_state=0)

        if validate

        else (X, None, y, None)

    )



    transformer = DataFrameMapper(

        [

            (onehot_cols, OneHotEncoder(drop="if_binary")),

            (

                list(ordinal_cols.keys()),

                OrdinalEncoder(categories=list(ordinal_cols.values())),

            ),

        ],

        default=StandardScaler(),

    )



    X_train = transformer.fit_transform(X_train)

    X_valid = transformer.transform(X_valid) if validate else None



    input_nodes = X_train.shape[1]

    output_nodes = 1



    model = Sequential()

    model.add(Input((input_nodes,)))

    model.add(Dense(64, activation="relu"))

    model.add(Dropout(0.3, seed=0))

    model.add(Dense(32, activation="relu"))

    model.add(Dropout(0.3, seed=1))

    model.add(Dense(16, activation="relu"))

    model.add(Dropout(0.3, seed=2))

    model.add(Dense(output_nodes))

    model.compile(optimizer="adam", loss="mean_squared_logarithmic_error")



    history = model.fit(

        X_train,

        y_train,

        batch_size=batch_size,

        epochs=100,

        validation_data=(X_valid, y_valid) if validate else None,

        verbose=2,

    )



    return history.history, model, transformer





print("Model 1:")

history_1, _, _ = run_pipeline(

    loans_1,

    onehot_cols,

    ordinal_cols,

    batch_size=128,

)

print("\nModel 2:")

history_2, _, _ = run_pipeline(

    loans_2,

    onehot_cols + new_metric_onehot_cols,

    ordinal_cols,

    batch_size=64,

)

print("\nModel 3:")

history_3, _, _ = run_pipeline(

    loans_3,

    onehot_cols + new_metric_onehot_cols,

    ordinal_cols,

    batch_size=32,

)
sns.lineplot(x=range(1, 101), y=history_1["loss"], label="loss")

sns.lineplot(x=range(1, 101), y=history_1["val_loss"], label="val_loss")

plt.xlabel("epoch")

plt.title("Model 1 loss metrics during training")

plt.show()
import joblib



_, final_model, final_transformer = run_pipeline(

    loans_1,

    onehot_cols,

    ordinal_cols,

    batch_size=128,

    validate=False,

)



final_model.save("loan_risk_model")

joblib.dump(final_transformer, "data_transformer.joblib")
# Exports for "Can I Grade Loans Better than LendingClub?"

expected.rename("expected_return", inplace=True)

loans_for_eval = loans_1.join([loans_raw[["issue_d", "grade", "sub_grade"]], expected])

joblib.dump(loans_for_eval, "loans_for_eval.joblib")



# Exports for "Improving Loan Risk Prediction With Natural Language Processing"

loans_for_nlp = loans_1.join(loans_raw[["issue_d", "title", "desc"]])

joblib.dump(loans_for_nlp, "loans_for_nlp.joblib")