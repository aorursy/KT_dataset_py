import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



# data analysis

import numpy as np

import pandas as pd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

from IPython.display import display, HTML



# modeling

from scipy import stats

from sklearn.model_selection import train_test_split

import statsmodels.api as sm

import sklearn.metrics as metrics

from statsmodels.stats.outliers_influence import variance_inflation_factor
sns.set(style="ticks", color_codes=True)

pd.options.mode.chained_assignment = None

pd.options.display.float_format = '{:.3f}'.format

pd.options.display.max_colwidth = -1
insurance = pd.read_csv("/kaggle/input/insurance/insurance.csv")

insurance.head()
insurance.info()
insurance[['sex', 'smoker', 'region']] = insurance[['sex', 'smoker', 'region']].astype('category')

insurance.dtypes
for col in insurance.select_dtypes('category').columns:

    print(col, ":", insurance[col].cat.categories)
pair_plot = sns.pairplot(insurance, diag_kind = "kde", corner = True, markers = '+',

                         kind = "reg")

pair_plot.fig.suptitle("Pair Plot of Numerical Variables", size = 25, y = 1.05)

pair_plot
fig, axes = plt.subplots(1, 3, figsize=(15,5))

for ax, col in zip(axes, insurance.select_dtypes('category').columns):

    sns.violinplot(x = col, y = "charges", data = insurance, ax = ax)

plt.tight_layout()

fig.suptitle("Violin Plot of Categorical Variables", size = 28, y = 1.05)

plt.show()
fig, axes = plt.subplots(3, 3, figsize=(15,15))



for row, cat in enumerate(insurance.select_dtypes('category').columns):

    for col, num in enumerate(insurance.select_dtypes(np.number).columns[:-1]):

        sns.scatterplot(x = num, y = "charges", hue = cat, data = insurance,

                        alpha = 0.6, ax = axes[row][col])

    

plt.tight_layout()

fig.suptitle("Scatter Plot of Each Numerical and Categorical Variables", size = 28, y = 1.025)

plt.show()
y_boxplot = sns.boxplot(insurance['charges'])

y_boxplot.set_title("Boxplot of Charges")

y_boxplot
insurance_wo_outlier = insurance.copy()

while True:

    y_boxplot = plt.boxplot(insurance_wo_outlier['charges'])

    lower_whisker, upper_whisker = [item.get_ydata()[1] for item in y_boxplot['whiskers']]

    outlier_flag = (insurance_wo_outlier['charges'] < lower_whisker) | (insurance_wo_outlier['charges'] > upper_whisker)

    num_outlier = sum(outlier_flag)

    if num_outlier == 0:

        before = insurance.shape[0]

        after = insurance_wo_outlier.shape[0]

        print("Total Outlier Removed: {}/{} ({}%)".format(before-after, before,

                                                          round(100*(before-after)/before, 3)))

        print("Final Range: ({}, {})".format(lower_whisker, upper_whisker))

        

        break

    print("Remove Outlier: {}/{} ({}%)".format(num_outlier, insurance_wo_outlier.shape[0],

                                               round(100*num_outlier/insurance_wo_outlier.shape[0], 3)))

    plt.show()

    insurance_wo_outlier = insurance_wo_outlier[-outlier_flag]
corr_heatmap = sns.heatmap(insurance.corr(method = "pearson"),

                           annot = True, fmt='.3f', linewidths = 5, cmap = "Reds")

corr_heatmap.set_title("Pearson Correlation", size = 25)

corr_heatmap
def check_linearity(data, target_var, SL = 0.05):

    cor_test_list = []

    for col in data.drop(target_var, axis = 1).columns:

        if col in data.select_dtypes('category').columns:

            cor_test = stats.spearmanr(data[col], data[target_var])

            cor_type = "Spearman"

        else:

            cor_test = stats.pearsonr(data[col], data[target_var])

            cor_type = "Pearson"

        cor_dict = {"Predictor": col,

                    "Type": cor_type,

                    "Correlation": cor_test[0],

                    "P-Value": cor_test[1],

                    "Conclusion": "significant" if cor_test[1] < SL else "not significant"}

        cor_test_list.append(cor_dict)

    return pd.DataFrame(cor_test_list)



check_linearity(insurance, "charges")
X_raw = insurance.drop(["charges"], axis = 1)

y = insurance.charges.values
X = pd.get_dummies(X_raw, columns = insurance.select_dtypes('category').columns, drop_first = True)

X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 333)

print("X Train:", X_train.shape)

print("X Test:", X_test.shape)

print("y Train:", y_train.shape)

print("y Test:", y_test.shape)
model_all = sm.OLS(y_train, sm.add_constant(X_train))

result_all = model_all.fit()

print(result_all.summary())
def backwardEliminationByAIC(X, y, show_iter = True):

    X_step = X.copy()

    num_iter = 1

    drop_list = []

    while True:

        res_list = []

        for col in ['none'] + list(X_step.columns):

            X_curr = X_step.drop(col, axis = 1) if col != "none" else X_step

            model = sm.OLS(y, sm.add_constant(X_curr)).fit()

            res_list.append({"drop": col, "aic": model.aic})



        curr_res = pd.DataFrame(res_list).sort_values("aic")

        col_to_be_removed = list(curr_res["drop"])[0]

    

        if show_iter:

            print("Iteration {}: Drop {}".format(num_iter, col_to_be_removed))

            display(HTML(curr_res.to_html(index=False)))



        if col_to_be_removed == "none":

            break 

        else:

            drop_list.append(col_to_be_removed)

            X_step = X_step.drop(col_to_be_removed, axis = 1)

        num_iter += 1

    X_back = X.drop(drop_list, axis = 1)

    model_back = sm.OLS(y, sm.add_constant(X_back))

    return model_back



model_back = backwardEliminationByAIC(X, y)

predictor_back = model_back.exog_names[1:]

predictor_back
def forwardSelectionByAIC(X, y, show_iter = True):

    X_step = pd.DataFrame(sm.add_constant(X)['const'])

    num_iter = 1

    add_list = []



    while True:

        res_list = [{"add": "none", "aic": sm.OLS(y, sm.add_constant(X_step)).fit().aic}]

        for col in list(set(X.columns) - set(add_list)):

            X_curr = X[add_list + [col]]

            model = sm.OLS(y, sm.add_constant(X_curr)).fit()

            res_list.append({"add": col, "aic": model.aic})



        curr_res = pd.DataFrame(res_list).sort_values("aic")

        col_to_be_added = list(curr_res["add"])[0]



        if show_iter:

            print("Iteration {}: Add {}".format(num_iter, col_to_be_added))

            display(HTML(curr_res.to_html(index=False)))



        if col_to_be_added == "none":

            break 

        else:

            add_list.append(col_to_be_added)

            X_step = X[add_list]

        num_iter += 1

    X_forward = X[add_list]

    model_forward = sm.OLS(y, sm.add_constant(X_forward))

    return model_forward



model_forward = forwardSelectionByAIC(X, y)

predictor_forward = model_forward.exog_names[1:]

predictor_forward
set(predictor_back) == set(predictor_forward)
def experimentModel(X_train, X_test, y_train, ignore_var=[]):

    X_train_new = X_train.drop(ignore_var, axis = 1)

    X_test_new = X_test.drop(ignore_var, axis = 1)

    model_new = sm.OLS(y_train, sm.add_constant(X_train_new))

    result_new = model_new.fit()

    return X_test_new, result_new
# 2. Model without predictor sex

X_test_wo_sex, result_wo_sex = experimentModel(X_train, X_test, y_train,

                                               ignore_var = ['sex_male'])



# 3. Model without predictor sex and region

X_test_wo_sex_region, result_wo_sex_region = experimentModel(X_train, X_test, y_train,

                                                             ignore_var = ['sex_male', 'region_northwest', 'region_southeast', 'region_southwest'])

X_wo_outlier = X.iloc[insurance_wo_outlier.index]

y_wo_outlier = insurance_wo_outlier.charges.values

X_train_wo_outlier, X_test_wo_outlier, y_train_wo_outlier, y_test_wo_outlier = train_test_split(X_wo_outlier, y_wo_outlier, test_size = 0.2, random_state = 333)

print("X Train:", X_train_wo_outlier.shape)

print("X Test:", X_test_wo_outlier.shape)

print("y Train:", y_train_wo_outlier.shape)

print("y Test:", y_test_wo_outlier.shape)
X_test_wo_outlier_all, result_wo_outlier_all = experimentModel(X_train_wo_outlier, X_test_wo_outlier, y_train_wo_outlier)

X_test_wo_outlier_wo_sex, result_wo_outlier_wo_sex = experimentModel(X_train_wo_outlier, X_test_wo_outlier, y_train_wo_outlier,

                                                                     ignore_var = ['sex_male'])

X_test_wo_outlier_wo_sex_region, result_wo_outlier_wo_sex_region = experimentModel(X_train_wo_outlier, X_test_wo_outlier, y_train_wo_outlier,

                                                                                   ignore_var = ['sex_male', 'region_northwest', 'region_southeast', 'region_southwest'])
def evalRegression(model, X_true, y_true, outlier = True, decimal = 5):

    y_pred = model.predict(sm.add_constant(X_true))

    

    metric = {

        "Predictor": sorted(model.model.exog_names[1:]),

        "Outlier": "Included" if outlier else "Excluded",

        "R-sq": round(model.rsquared, decimal),

        "Adj. R-sq": round(model.rsquared_adj, decimal),

        "RMSE": round(np.sqrt(metrics.mean_squared_error(y_true, y_pred)), decimal),

        "MAPE": round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, decimal)

    }

    

    return metric
eval_df = pd.DataFrame([evalRegression(result_all, X_test, y_test),

              evalRegression(result_wo_sex, X_test_wo_sex, y_test),

              evalRegression(result_wo_sex_region, X_test_wo_sex_region, y_test),

              evalRegression(result_wo_outlier_all, X_test_wo_outlier_all, y_test_wo_outlier, outlier = False),

              evalRegression(result_wo_outlier_wo_sex, X_test_wo_outlier_wo_sex, y_test_wo_outlier, outlier = False),

              evalRegression(result_wo_outlier_wo_sex_region, X_test_wo_outlier_wo_sex_region, y_test_wo_outlier, outlier = False)],

              index = range(1, 7))

eval_df.index.name = "Model"

eval_df
# Model 5

final_X_test = X_test_wo_outlier_wo_sex

final_y_test = y_test_wo_outlier

final_result = result_wo_outlier_wo_sex
def evalRegularizedRegression(model_result, X_test, y_test):

    model = model_result.model

    eval_regularized_list = []

    for alpha in np.linspace(0, 10, 101):

        for fit_type in [0, 1]:

            result_regularized = model.fit_regularized(alpha = round(alpha, 2),

                                                       L1_wt = fit_type,

                                                       start_params = model_result.params)

            final = sm.regression.linear_model.OLSResults(model, 

                                                          result_regularized.params,

                                                          model.normalized_cov_params)

            metric = {}

            metric["alpha"] = alpha

            metric["Fit Type"] = "Ridge" if fit_type == 0 else "Lasso"

            metric.update(evalRegression(final, X_test, y_test, outlier = False))

            metric.pop("Predictor")



            eval_regularized_list.append(metric)

    return pd.DataFrame(eval_regularized_list)



eval_regularized = evalRegularizedRegression(final_result, final_X_test, final_y_test)
for metric in ["Adj. R-sq", "MAPE"]:

    facet = sns.FacetGrid(eval_regularized, col = "Fit Type")

    facet = facet.map(plt.plot, "alpha", metric)

    facet.set_axis_labels("Penalty Weight")

    facet.fig.suptitle("Evaluate Regularized Linear Regression by {}".format(metric), y = 1.05)
eval_regularized[(eval_regularized["Adj. R-sq"] == max(eval_regularized["Adj. R-sq"])) &

                 (eval_regularized["MAPE"] == min(eval_regularized["MAPE"]))]
print(final_result.summary())
print(final_result.params)
y_pred = final_result.predict(sm.add_constant(final_X_test))

residual = final_y_test - y_pred

residual.describe()
def plotResidualNormality(residual):

    residual_histogram = sns.distplot(residual)

    residual_histogram.set_xlabel("Residual")

    residual_histogram.set_ylabel("Relative Frequency")

    residual_histogram.set_title("Distribution of Residuals", size = 25)

    return residual_histogram



plotResidualNormality(residual)
def statResidualNormality(residual):

    W, pvalue = stats.shapiro(residual)

    print("Shapiro-Wilk Normality Test")

    print("W: {}, p-value: {}".format(W, pvalue))

    

statResidualNormality(residual)
def plotResidualHomoscedasticity(y_true, y_pred):

    residual = y_true - y_pred

    residual_scatter = sns.scatterplot(y_pred, residual)

    residual_scatter.axhline(0, ls = '--', c = "red")

    residual_scatter.set_xlabel("Fitted Value")

    residual_scatter.set_ylabel("Residual")

    residual_scatter.set_title("Scatter Plot of Residuals", size = 25)

    return residual_scatter



plotResidualHomoscedasticity(final_y_test, y_pred)
def statResidualHomoscedasticity(X, residual):

    lm, lm_pvalue, fvalue, f_pvalue = sm.stats.diagnostic.het_breuschpagan(residual,

                                                                           np.array(sm.add_constant(X)))



    print("Studentized Breusch-Pagan Test")

    print("Lagrange Multiplier: {}, p-value: {}".format(lm, lm_pvalue))

    print("F: {}, p-value: {}".format(fvalue, f_pvalue))

    

statResidualHomoscedasticity(final_X_test, residual)
vif_list = []

for idx, col in enumerate(final_result.model.exog_names[1:]):

    vif_dict = {"Variable": col,

                "VIF":  variance_inflation_factor(final_result.model.exog, idx+1)}

    vif_list.append(vif_dict)

    

pd.DataFrame(vif_list)