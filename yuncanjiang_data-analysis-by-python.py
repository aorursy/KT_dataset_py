import pandas as pd

from matplotlib import pyplot as plt

from scipy import stats

from scipy.stats import pearsonr

from sklearn.linear_model import LinearRegression
# read data 

courses_df = pd.read_csv('../input/udemy-courses/udemy_courses.csv')

# general info of this data

courses_df.info()
# get the y-axis, y = num_subscribers

temp_df1 = courses_df['num_subscribers']

y1 = temp_df1.values



# get the x-axis, x = num_reviews

# x_reviews

review_df = courses_df['num_reviews']

x1 = review_df.values



# draw scatter plot 

plt.xlabel('num_reviews')

plt.ylabel('num_subscribers')

plt.scatter(x1,y1)

plt.show()
# get the num_subcribers and num_reviews column 

temp_df2 = courses_df.iloc[:,[5,6]]



# filter these two columns with num_reviews <= 10000

filter_cols = temp_df2[temp_df2['num_reviews']<=10000]



# get the new x,y axis after filter 

y2 = filter_cols['num_subscribers'].values

x2 = filter_cols['num_reviews'].values



# draw the trendline this time

slope, intercept, r, p, std_err = stats.linregress(x2,y2)

def myfunc(x2):

    return slope * x2 + intercept



mymodel = list(map(myfunc,x2))



plt.plot(x2,mymodel)



# draw scatter plot again

plt.xlabel('num_reviews')

plt.ylabel('num_subscribers')

plt.scatter(x2,y2)

plt.show()
corr, _ = pearsonr(x2,y2)

'Pearson"s correlation: %.3f' % corr

# we have a high positive correlation!
# prepare x and y variables 

y2 = filter_cols['num_subscribers'].values

x2 = filter_cols['num_reviews'].values.reshape(-1, 1)



# built the regression model

model = LinearRegression().fit(x2,y2)



# rounded to two decimal digits

model_coef = float('%.2f' % model.coef_)

model_intercept = float('%.2f' % model.intercept_)



# model expression 

f'y = {model_coef}*x + {model_intercept}'
# r-square value

r_sq = model.score(x2,y2)

r_sq = float('%.3f' % r_sq)

f'r-square value:{r_sq}'



# not very high r-square value