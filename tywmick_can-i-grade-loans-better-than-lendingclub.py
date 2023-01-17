import joblib

prev_notebook_folder = "../input/building-a-neural-network-to-predict-loan-risk/"
loans = joblib.load(prev_notebook_folder + "loans_for_eval.joblib")
loans.shape
loans.head()
from sklearn.model_selection import train_test_split

loans["date"] = loans["issue_d"].astype("datetime64[ns]")
loans.sort_values("date", axis="index", inplace=True, kind="mergesort")

train, test = train_test_split(loans, test_size=0.2, shuffle=False)
train, test = train.copy(), test.copy()
print(f"The test set contains {len(test):,} loans.")
from statistics import mean

lc_grade_a = test[test["grade"] == "A"]
print(f"LendingClub gave {len(lc_grade_a):,} loans in the test set an A grade.")

print("\nAverage `fraction_recovered` on LendingClub's grade A loans:")
print(format(mean(lc_grade_a["fraction_recovered"]), ".5f"))
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


onehot_cols = ["term", "application_type", "home_ownership", "purpose"]
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
# Train the model
_, model, transformer = run_pipeline(
    train.drop(columns=["issue_d", "date", "grade", "sub_grade", "expected_return"]),
    onehot_cols,
    ordinal_cols,
    batch_size=128,
    validate=False,
)

# Make predictions
X_test = transformer.transform(
    test.drop(
        columns=[
            "fraction_recovered",
            "issue_d",
            "date",
            "grade",
            "sub_grade",
            "expected_return",
        ]
    )
)
test["model_predictions"] = model.predict(X_test)

# Gather top predictions
test_sorted = test.sort_values("model_predictions", axis="index", ascending=False)
ty_grade_a = test_sorted.iloc[0:len(lc_grade_a)]

# Display results
print("\nAverage `fraction_recovered` on Ty's grade A loans:")
print(format(mean(ty_grade_a["fraction_recovered"]), ".5f"))