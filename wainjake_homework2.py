import turicreate as tc
import turicreate.aggregate as agg
sales = tc.SFrame('../input/basicml-lecture1/Homework2/home_data.sframe')
sales.head()
group_price = sales.groupby('zipcode', operations={'mean': agg.MEAN('price')})
group_price.sort('mean', ascending=False)
filter_sales = sales[(sales['sqft_living'] > 2000) & (sales['sqft_living'] <= 4000)]
filter_sales
num = filter_sales.num_rows()
num 
ratio = num/sales.num_rows()
ratio
training_set, test_set = sales.random_split(.8,seed=0)
my_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','zipcode']
advanced_features = [
'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
'condition', # condition of house       
'grade', # measure of quality of construction       
'waterfront', # waterfront property       
'view', # type of view        
'sqft_above', # square feet above ground        
'sqft_basement', # square feet in basement        
'yr_built', # the year built        
'yr_renovated', # the year renovated        
'lat', 'long', # the lat-long of the parcel       
'sqft_living15', # average sq.ft. of 15 nearest neighbors         
'sqft_lot15' ]  # average lot size of 15 nearest neighbors
my_features_model = tc.linear_regression.create(training_set,target='price',features=my_features, validation_set=None)
my_RMSE = my_features_model.evaluate(test_set)['rmse']
my_RMSE
ad_features_model = tc.linear_regression.create(training_set,target='price',features=advanced_features, validation_set=None)
ad_RMSE = ad_features_model.evaluate(test_set)['rmse']
ad_RMSE
my_RMSE, ad_RMSE
