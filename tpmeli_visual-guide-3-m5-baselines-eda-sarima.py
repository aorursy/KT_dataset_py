# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

sns.set_style('whitegrid')

sns.set_context('poster')



import matplotlib.pyplot as plt

from pandas_profiling import ProfileReport

from IPython.display import HTML



from tqdm import tqdm



%matplotlib inline

plt.rcParams["figure.figsize"] = (16,12)

plt.rcParams['axes.titlesize'] = 16



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_sales = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

submission_file = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')
print(train_sales.shape)

train_sales.sample(2)
CREATE_TIDY_DF = False  # We will not create the tidy DF here.  Offline instead due to memory constraints.

LOAD_TIDY_DF = True  # When done, it will load externally for ease instead of processing in the kernel.
if CREATE_TIDY_DF:

    # Create a list comprehension for all the date columns to melt.

    d_cols = ['d_' + str(i + 1) for i in range(1913)]



    # Melt columns into rows so that each row is a separate and discrete entry with one target

    tidy_df = pd.melt(frame = train_sales, 

                     id_vars = ['id', 'item_id', 'cat_id', 'store_id'],

                     var_name = 'd',

                     value_vars = d_cols,

                     value_name = 'sales')



    # This has duplicate ID's now.  We should add the date to the id to make each row unique.

    new_ids = tidy_df['id'] + '_' + tidy_df['d']

    tidy_df['id'] = new_ids



    # Check this turned out ok so far.

    tidy_df.head()
if CREATE_TIDY_DF:

    # Merge the prices.  

    # NOTE - For now we are aggregating on the mean price of each item.

    # TO DO: We will want to set the price with the week or run some statistics on price volatility over time.



    price_means = sell_prices.groupby(['item_id']).mean()

    

    # Now, merge this and the date col

    with_prices_df = pd.merge(left = tidy_df, right = calendar,

                            on = 'd')



    with_prices_df.head(10)

    # Let's see the results.
if CREATE_TIDY_DF:

    with_date_info_df = pd.merge(left = with_prices_df, right = price_means,

                            on = 'item_id')

    

    total_tidy_df = with_date_info_df

    total_tidy_df.columns

    

    # Drop d and drop item_id (price is an informative proxy)

    total_tidy_df.drop(['d', 'wday', 'item_id'], axis = 1, inplace = True)

    

    # fill categorical NaNs with 0's.

    total_tidy_df = total_tidy_df.fillna(0)

    

    print(with_date_info_df.iloc[0])



if CREATE_TIDY_DF:

    

    # Categorical encoded column helper function.

    def categorically_encode_col(df, col):

        encoded_df = pd.get_dummies(df[col], 

                                    prefix = str(col),

                                   drop_first = False)



        return encoded_df

    

    total_tidy_df.columns



# Categorically encode the categorical columns and then drop the originals.

# This makes them ML ready.



if CREATE_TIDY_DF:

    

    # Categorically encode categorical columns

    cols_to_encode = ['cat_id', 'store_id', 'weekday', 'event_type_1', 'event_type_2' ]

    

    for col in cols_to_encode:

        new_cols = pd.DataFrame(categorically_encode_col(total_tidy_df, col))

        total_tidy_df = pd.concat([total_tidy_df, new_cols], axis = 1)

        # total_tidy_df.drop(col, inplace = True)  # Drop the un-encoded column
if CREATE_TIDY_DF:

    total_tidy_df.columns
if CREATE_TIDY_DF:

    # Export if necessary

    total_tidy_df.to_csv('total_tidy_df.csv')
def disp_boxplot(data, title, xlabel, ylabel):

    sns.set_style('whitegrid')

    sns.set_context('poster')

    palette = sns.color_palette("mako_r", 6)

    

    ax = sns.boxplot(data=data, palette = palette)

    ax.set(title = title,

          xlabel = xlabel,

          ylabel = ylabel)

    

    try:

        ax.axhline(y = data.mean().mean(), color = 'b', label = 'Mean of all datapoints', linestyle = '--', linewidth = 1.5)

        ax.ahline(y = data.median().median(), color = 'g', label = 'Median of all datapoints', linestyle = '--', linewidth = 1.5)

    except:

        pass

    

    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)

    

    plt.legend()

    plt.show()
dept_sales = train_sales.groupby(['dept_id']).mean().mean()

dept_sum = train_sales.groupby(['dept_id']).sum().T.reset_index(drop = True)
disp_boxplot(data = dept_sum, title = 'Total Sales by Department',

            xlabel = "Department", ylabel = "Total Sales")
dept_storeloc_cross = pd.crosstab(train_sales['dept_id'], train_sales['store_id'])

print(dept_storeloc_cross)

ax = sns.heatmap(dept_storeloc_cross,

                linewidths = 0.4,

                cmap="BuGn")

ax.set(title = 'Number of items in each category per store - Uniform')
n_items_dept = train_sales['dept_id'].value_counts()

mean_of_total_sales_per_dept = dept_sum.mean(axis = 0)



ax = sns.regplot(n_items_dept, mean_of_total_sales_per_dept)

ax.set(title = 'Do departments with more items sell more? - No',

      xlabel = 'Number of items Per Department',

      ylabel = 'Mean total sales per department.')

plt.show()
cat_sum = train_sales.groupby(['cat_id']).sum().T.reset_index(drop = True)

disp_boxplot(data = cat_sum, title = 'Total Sales by Category',

            xlabel = "Category", ylabel = "Total Sales")
state_sum = train_sales.groupby(['state_id']).sum().T.reset_index(drop = True)

state_mean = train_sales.groupby(['state_id']).mean().T.reset_index(drop = True)



disp_boxplot(data = state_sum, title = 'Total Sales by State ID',

            xlabel = "State ID", ylabel = "Total Sales")



disp_boxplot(data = state_mean, title = 'Mean Sales by State ID',

            xlabel = "State ID", ylabel = "Mean Sales")
train_sales.head()
store_sum = train_sales.groupby(['store_id']).sum().T.reset_index(drop = True)

store_mean = train_sales.groupby(['store_id']).mean().T.reset_index(drop = True) 



disp_boxplot(data = store_sum, title = 'Total Sales by Store ID',

            xlabel = "Store ID", ylabel = "Total Sales")





disp_boxplot(data = store_mean, title = 'Mean Sales Per Day by Store ID',

            xlabel = "Store ID", ylabel = "Total Sales")
ax = sns.regplot(x = np.arange(dept_sales.shape[0]), y = dept_sales,

                 scatter_kws = {'color':'blue', 'alpha': 0.1},

                 order = 3, line_kws = {'color':'green'},)



ax.set(title = "Mean Total Sales Per Item Per Day Over Time",

      xlabel = 'Day ID', ylabel = 'Total sale per item per day')



plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose

weeks_per_year = 365



time_series = store_sum["CA_1"]

sj_sc = seasonal_decompose(time_series, period = weeks_per_year)

sj_sc.plot()



plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose

days_per_week = 7



time_series = store_sum["CA_1"]

sj_sc = seasonal_decompose(time_series, period = days_per_week)

sj_sc.plot()



plt.show()
from statsmodels.tsa.statespace.sarimax import SARIMAX



def sarima_train_test(t_series, p = 2, d = 1, r = 2, NUM_TO_FORECAST = 56, do_plot_results = True):

    NUM_TO_FORECAST = NUM_TO_FORECAST  # Similar to train test splits.

    dates = np.arange(t_series.shape[0])



    model = SARIMAX(t_series, order = (p, d, r), trend = 'c')

    results = model.fit()

    results.plot_diagnostics(figsize=(18, 14))

    plt.show()



    forecast = results.get_prediction(start = -NUM_TO_FORECAST)

    mean_forecast = forecast.predicted_mean

    conf_int = forecast.conf_int()



    print(mean_forecast.shape)



    # Plot the forecast

    plt.figure(figsize=(14,16))

    plt.plot(dates[-NUM_TO_FORECAST:],

            mean_forecast.values,

            color = 'red',

            label = 'forecast')





    plt.plot(dates[-NUM_TO_FORECAST:],

            t_series.iloc[-NUM_TO_FORECAST:],

            color = 'blue',

            label = 'actual')

    plt.legend()

    plt.title('Predicted vs. Actual Values')

    plt.show()

    

    residuals = results.resid

    mae_sarima = np.mean(np.abs(residuals))

    print('Mean absolute error: ', mae_sarima)

    print(results.summary())

sarima_train_test(time_series)
USE_SARIMA_PREDS = True
if USE_SARIMA_PREDS:

    # Clean this code up.

    

    sarima_preds = pd.read_csv('/kaggle/input/m5-untuned-sarima-preds/Sarima_preds_submission.csv')

    sarima_preds[sarima_preds < 0] = 0  # Convert all negative numbers into 0.

    sarima_preds['id']= submission_file['id']

    

    # sarima_preds.to_csv('submission.csv', index = False)

    

    submission_df = sarima_preds

    

    #Cleaning

    submission_df = submission_df.iloc[:,:29]

    submission_df = submission_df.drop(['Unnamed: 0'], axis = 1)

    submission_df.index = submission_file['id']

    submission_df.reset_index(inplace = True)

    submission_df.columns = submission_file.columns

    submission_df.head()

    

    sarima_df = submission_df.copy()
train_sales.shape
def subset_validation_set(df):

    is_validation_subset = df['id'].str.contains('validation')

    valid_subset = df[is_validation_subset]

    return valid_subset



def create_evaluation_rows(df):

    val_idx = df['id']

    



def prepare_submission_file(df, i, val_or_eval):

    ########################################################

    # This function does several things:

    #  It aggregates data from all the files and makes them

    #  inputtable to traditional ML algorithms, such as trees.

    

    #  It will output the a dataframe for each 'day'

    

    # Returns a submission_like dataframe that is ready to put

    # in an estimator.

    # Please cite and upvote this kernel if you use this code

    #########################################################

    

    # Extract the validation samples

    # valid_subset = subset_validation_set(df)

    valid_subset = df  # Fix this.

    

    # assert valid_subset.shape[0] == train_sales.shape[0], "The rows are not equal"

    

    # To do: validate the indices as well.

    

    # Collect date information 2016-04-25 to 2016-05-22 <-- The validation set days.

    # First denote which columns you care about.

    # + i is so that I can loop through to get all 28 dates.

    

    if val_or_eval == 'val':

        # Validation range: 2016-04-25 to 2016-05-22

        

        # d_87 - d_114

        i1 = 87 + i

        i1_str = "d_" + str(i1)



        # d_453 - d_480  <-- Leap year.  366 days after.

        i2 = 453 + i

        i2_str = "d_" + str(i2)



        # d_818 - d_845

        i3 = 818 + i

        i3_str = "d_" + str(i3)



        # d_1183 - d_1210

        i4 = 1183 + i

        i4_str = "d_" + str(i4)



        # d_1548 - d_1575

        i5 = 1548 + i

        i5_str = "d_" + str(i5)



        # d_1914 - d_1941  <-- Not in validation set

        # i6 = 1941 + i

        # i6_str = "d_" + str(i6)

        

    elif val_or_eval == 'eval':

        # Evaluation range: 2016-05-23 to 2016-06-19

        # d_87 - d_114

        eval_num_ahead = 28

        i1 = 87 + eval_num_ahead + i

        i1_str = "d_" + str(i1)



        # d_453 - d_480  <-- Leap year.  366 days after.

        i2 = 453 + eval_num_ahead + i

        i2_str = "d_" + str(i2)



        # d_818 - d_845

        i3 = 818 + eval_num_ahead + i

        i3_str = "d_" + str(i3)



        # d_1183 - d_1210

        i4 = 1183 + eval_num_ahead + i

        i4_str = "d_" + str(i4)



        # d_1548 - d_1575

        i5 = 1548 + eval_num_ahead + i

        i5_str = "d_" + str(i5)



        # d_1914 - d_1941  <-- Not in validation set

        # i6 = 1941 + eval_num_ahead + i

        # i6_str = "d_" + str(i6)

        

    

    all_important_days = [i1_str, i2_str, i3_str, i4_str, i5_str]

    col_names = ['this_day_1', 'this_day_2', 'this_day_3', 'this_day_4', 'this_day_5' ]

    

    # Extract the data from just these rows.

    important_days_subset = valid_subset[all_important_days]

    important_days_subset.columns = col_names

    

    # TO DO:

    # Rename them F1-F28?

    

    # Add the cost of the item.

    

    # create is_special_event

    

    # categorically encode special_event.

    

    # encode_is_snap

    

    return important_days_subset



DO_BASELINE_MEAN_PRED = False
if DO_BASELINE_MEAN_PRED:

    

    # Do a loop for every F_DAYNUM

    # Retrieve the information for that.

    # Create a submission based on that information.



    ####################################################

    # TO DO: CLEAN THIS UP.

    # Do the validation set.

    final_submission_df = train_sales.copy()

    final_cols = ['id']  # This will just be the columns we want in the final submission file.



    for i in tqdm(range(28)):



        important_days_df = prepare_submission_file(train_sales, i, val_or_eval = 'val')



        # Round or don't roud here?

        mean_of_days = important_days_df.mean(axis = 1)   # Shouldn't this be axis = 0?  Works only with ax = 1.



        this_col = "F" + (str(i + 1))

        final_cols.append(this_col)



        final_submission_df[this_col] = mean_of_days



    # Do the same thing for the evaluation set for now.

    # Update this later to include calendar and other information.



    ####################################################

    # Do the evaluation set.

    final_submission_df_eval = train_sales.copy()

    final_cols = ['id']

    for i in tqdm(range(28)):



        important_days_df_eval = prepare_submission_file(train_sales, i, val_or_eval = 'eval')



        mean_of_days = important_days_df_eval.mean(axis = 1)  # Shouldn't this be axis = 0?  Works only with ax = 1.



        this_col = "F" + (str(i + 1))

        final_cols.append(this_col)



        final_submission_df_eval[this_col] = mean_of_days



    final_submission_df_eval['id'] = final_submission_df_eval['id'].str.replace('validation', 'evaluation')



    # CONCATENATE THE VAL AND EVAL SETS

    submission_df = pd.concat([final_submission_df, final_submission_df_eval])



    def clean_submission_file(df):

        df = df[final_cols]

        return df

    

    mean_of_days_df = submission_df.copy()

    mean_of_days_df = clean_submission_file(mean_of_days_df)

    
if DO_BASELINE_MEAN_PRED:

    print(submission_df.shape)

    submission_df = clean_submission_file(submission_df)

    print(submission_df.shape)

if DO_BASELINE_MEAN_PRED:

    submission_df.head(5)
# Understanding the forecast metric.

# https://robjhyndman.com/papers/foresight.pdf
submission_df.to_csv('submission.csv', index = False)

print(submission_df.shape)

print("Submission file created")
# For Viewing

eval_start = int(60980 / 2)

eval_head_end = eval_start + 2



submission_df.iloc[eval_start:eval_head_end, :]  # Take a look at the first few of the eval set.
lineplot_data = submission_df.iloc[eval_start:eval_head_end, 1:].T

lineplot_data.index = range(len(lineplot_data.index))  # Clean up the F1 naming.



ax = sns.scatterplot(data = lineplot_data, legend = False)

ax2 = sns.lineplot(data = lineplot_data, legend = False)



ax.set(title = 'Predicted Sales of 2 Different Products over 28 days',

      xlabel = 'Days',

      ylabel = 'Mean Products Sold')



plt.xticks(rotation = 90)

plt.show()