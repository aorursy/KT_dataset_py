import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



%matplotlib inline

from matplotlib import rcParams

rcParams['figure.figsize'] = 12, 10
np.random.seed(10)



in_sample_x = np.pi / 180 * np.random.uniform(0, 400, size=200)

out_sample_x = np.pi / 180 * np.random.uniform(0, 400, size=80)



in_sample_x = np.sort(in_sample_x)

out_sample_x = np.sort(out_sample_x)
in_sample_y = 1 - np.cos(in_sample_x) + in_sample_x + np.random.normal(0, 0.6, len(in_sample_x))

out_sample_y = 1 - np.cos(out_sample_x) + out_sample_x + np.random.normal(0, 0.6, len(out_sample_x))



data_in = pd.DataFrame(np.column_stack([in_sample_x, in_sample_y]), columns=['x', 'y'])

data_out = pd.DataFrame(np.column_stack([out_sample_x, out_sample_y]), columns=['x', 'y'])
plt.figure(1)

plt.subplot(221)

plt.plot(data_in['x'], data_in['y'], '.', color='blue', label="In sample observations")

plt.plot(data_in['x'], 1 - np.cos(in_sample_x) + in_sample_x, color='red', label="True function")

plt.legend(loc = "lower right")

plt.title('Training sample')

plt.xlabel('x (radians)')

plt.ylabel('y', rotation=0)

plt.subplot(222)

plt.plot(data_out['x'], data_out['y'], '.', color='green', label="Out of sample observations")

plt.plot(data_out['x'], 1 - np.cos(out_sample_x) + out_sample_x, color='red', label="True function")

plt.title('Validation sample')

plt.xlabel('x (radians)')

plt.ylabel('y', rotation=0)

plt.legend(loc = "lower right");
for i in range(2,40): # (Power of 1 is already there)

    colname = 'x_%d' % i # New var will be x_power

    data_in[colname] = data_in['x']**i

    data_out[colname] = data_out['x']**i



data_out.head()
# Fit the models with increasing polynomial order (i.e. complexity)

from sklearn.linear_model import LinearRegression

def linear_regression(data1, data2, power, models_to_plot):

    # Initialize predictors:

    predictors=['x']

    if power >= 2:

        predictors.extend(['x_%d' % i for i in range(2, power + 1)])

    

    # Fit the model 

    linreg = LinearRegression(normalize=True)

    linreg.fit(data1[predictors], data1['y'])

    y_pred = linreg.predict(data1[predictors])

    y_pred_out = linreg.predict(data2[predictors])

    

    # Check if a plot is to be made for the entered power

    if power in models_to_plot:

        # If so, it is plotted against the training data 

        # on the subplot specified by models_to_plot

        plt.subplot(models_to_plot[power])

        plt.tight_layout()

        plt.plot(data1['x'], y_pred, color = 'red')

        plt.plot(data1['x'], data1['y'],'.', color ='blue' )

        plt.title('Training sample: Plot for power: %d'%power)

        # The testing data is plotted on the adjacent subplot

        plt.subplot(models_to_plot[power] + 1)

        plt.tight_layout()

        plt.plot(data1['x'], y_pred, color = 'red')

        plt.plot(data2['x'], data2['y'], '.', color = 'green')

        plt.title('Validation sample: Plot for power: %d' % power)

    

    # Return the result in pre-defined format

    rss = sum((y_pred-data1['y'])**2) / len(y_pred)

    cvrss = sum((y_pred_out-data2['y'])**2) / len(y_pred_out)

    ret = [rss, cvrss]

    ret.extend([linreg.intercept_])

    ret.extend(linreg.coef_)

    return ret
col = ['rss', 'cross-rss','intercept'] + ['coef_x_%d' % i for i in range(1,40)]

ind = ['model_pow_%d' % i for i in range(1,40)]

coef_matrix_simple = pd.DataFrame(index=ind, columns=col)
# This dictionary defines the orders of polynomial model 

# to plot (keys) and corresponding subplots (values)

models_to_plot = {1:421, 7:423, 14:425, 19:427}



# Iterate through all powers and assimilate results

for i in range(1,40):

    coef_matrix_simple.iloc[i-1, 0:i+3] = linear_regression(data_in, data_out, power=i, models_to_plot=models_to_plot)

coef_matrix_simple
ax = coef_matrix_simple['rss'][0:30].plot(color='blue', label='MSE on training sample')

coef_matrix_simple['cross-rss'][0:30].plot(ax=ax, color='green',  label='MSE on validation sample')

ax.legend( loc='upper left')

ax.set_xlabel("Model complexity (Power of x)")

ax.set_ylabel("MSE");