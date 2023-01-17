%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
class LinearRegression():
    def __init__ (self):
        # Dataframes passed in for training and validation
        self.df_train = None
        self.df_valid = None
        
        # Model variables
        self.slope = None
        self.intercept = None
        self.n = None
        self.y_hat = None
        self.x_sum = None
        self.y_sum = None
        self.x2_sum = None
        self.y2_sum = None
        self.xy_sum = None
        
        # Scoring and assessment variables
        self.prediction = None
        self.residuals = None
        self.r2 = None
        self.mse = None
    
    
    def fit(self, df):
        '''
        Fits the linear regression model.
        
        Args:
        df - DataFrame with two columns labelled 'x' and 'y'
        
        Calculates and stores:
            - df_train with y_hat and residuals added to df
            - slope
            - intercept
            - r2
        
        Returns None
        '''
        # Set df_train to be the df passedd into fit()
        self.df_train = df
        
        # Set n to be the length of the df
        self.n = len(df)
        
        # Calculate the slope and intercept
        self.slope = self.calc_slope(self.n, self.df_train)
        self.intercept = self.calc_intercept()
        
        # Run calc_residuals and append results to df_train 
        self.df_train = self.calc_residuals(self.df_train)
        
        # Run calc_r2 on the df_train
        self.calc_r2(self.df_train)
        
        
    def calc_slope(self, n, df):
        '''
        Calculates the slope for linear regression
        
        Args:
        n - int of the number of rows in the df
        df - DataFrame with two columns labelled 'x' and 'y'
        
        Returns float
        '''
        # Step 1 For each (x,y) calculate x2 and xy
        x2 = df.x **2
        xy = df.x * df.y

        # Step 2: Sum all x, y, x2 and xy
        x_sum = df.x.sum()
        self.x_sum = x_sum
        
        y_sum = df.y.sum()
        self.y_sum = y_sum
        
        x2_sum = x2.sum()
        self.x2_sum = x2_sum
        
        xy_sum = xy.sum()
        self.xy_sum = xy_sum

        # Step 3: Calculate the slope 
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - (x_sum**2))
        
        return slope
        
        
    def calc_intercept(self):
        '''
        Calculates the intercept for linear regression
        
        Args:
        None
        
        Returns float
        '''
        # Step 4: Calculate the intercept
        intercept = (self.y_sum - self.slope * self.x_sum) / self.n
    
        return intercept
    
    
    def calc_residuals(self, df):
        '''
        Calculates the predictions and errors (residuals) of a DataFrame.
        Appends 'y_hat' and 'residuals' columns to the DataFrame
        
        Args:
        df - DataFrame with two columns labelled 'x' and 'y'
        
        Returns DataFrame
        '''
        # Calculate all predicitons 
        all_preds = []
        
        for value in df['x']:
            self.predict(value)
            all_preds.append(self.prediction)
        
        # Add predicitons to df
        df['y_hat'] = pd.Series(all_preds)
            
        #Calculate the residuals, rounded to 2 decimal places
        df['residuals'] = round(df.y - df.y_hat, 2)
        
        return df
    
    
    def calc_r2(self, df):
        '''
        Calculates r2 (r squared)
        
        Args:
        df - DataFrame with two columns labelled 'x' and 'y'
        
        Returns None
        '''
        # Step 1: Calculate the sum of xy - previously done
        # Step 2: Calculate the sum of x2 - previously done 
        # Step 3: Calculate the sum of y2
        y2 = df.y**2
        y2_sum = y2.sum()
        self.y2_sum = y2_sum
        
        # Step 4: Calculate the correlation coefficient
        top = self.n * self.xy_sum - self.x_sum * self.y_sum
        bottom = math.sqrt((self.n * self.x2_sum - self.x_sum**2) * (self.n * self.y2_sum - self.y_sum**2))
        r = top / bottom
        self.r2 = r**2
        

    def predict(self, x, rounding=2, print_pred=False):
        '''
        Calculates the predicition for a single value (x) using the trained model
        
        Args:
        x - x value for calculating the prediction y_hat
        rounding - set rounding for returned y_hat. Default 2
        print_pred - print the prediction (y_hat) to screen. Default False
        
        Returns None
        '''        
        prediction = self.slope * x + self.intercept
        self.prediction = prediction
        if print_pred is True:
            print('pred:',round(self.prediction, rounding))
    
    
    def valid(self, df):
        '''
        Calculate the mean square error (mse) of a validation set using the trained model
        
        Args:
        df - DataFrame for validation set with two columns labelled 'x' and 'y'
        
        Returns Float
        '''
        # Pass df and set to df_valid
        self.df_valid = df
        
        # Calculate residuals
        df = self.calc_residuals(df)

        # Calc MSE (Mean square error)
        mse = np.mean(df['residuals'])**2

        return mse
        
# Import data x = age, y = bloodpressure
data = pd.read_csv('../input/train.csv')
# Display the data in a scatter plot to see if there is an approximate linear relationship
data.plot.scatter('x', 'y')
# Set training set to be 80% of the data
n_train = int((len(data)*0.8))

# Create training set by randomly sampling data 
train = data.sample(n=n_train, random_state=1)

# Create validation set which does not contain any of the values from train
valid = data.copy()
for index in train.index:
    valid.drop(index, inplace=True)

# Reset the indexes of each df
train = train.reset_index()
valid = valid.reset_index()
# Create a Linear Regression model called m
m = LinearRegression()
# Fit the linear regression model using the training df
m.fit(train)
# Check the r2 value of the trained model to assess correlation
m.r2
# Check the residuals of the model to ensure linear regression is appropriate
m.df_train.plot.scatter('x', 'residuals')
# Plot the y and y_hat predictions of the training set
y = m.df_train.y
y_hat = m.df_train.y_hat
x = m.df_train.x

# Plot the scatter plot
fig, ax = plt.subplots()
plt.ylabel('Blood Pressure')
plt.suptitle('Training Set')
plt.title('Y vs Line of Best Fit')
plt.scatter(x, y, c='blue')
plt.plot(x, y_hat, c='red')
plt.legend()
# Use the validation set to test the model. Returns mean-square error
m.valid(valid)
# Plot the y and y_hat predictions of the validation set
y = m.df_valid.y
y_hat = m.df_valid.y_hat
x = m.df_valid.x

# Plot the scatter plot
fig, ax = plt.subplots()
plt.ylabel('Blood Pressure')
plt.suptitle('Validation Set')
plt.title('Y vs Predictions')
plt.scatter(x, y, c='blue')
plt.scatter(x, y_hat, c='red')
plt.legend()