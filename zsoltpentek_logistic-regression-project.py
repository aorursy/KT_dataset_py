import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
ad_data = pd.read_csv('../input/advertising/advertising.csv')
ad_data.head()
ad_data.info()
ad_data.describe()
sns.set_style('whitegrid')
sns.distplot(ad_data['Age'], bins=20, kde=False)
sns.jointplot(data=ad_data, x='Age', y='Area Income')
sns.jointplot(data=ad_data, x='Age', y='Daily Time Spent on Site', kind='kde', color='red')
sns.jointplot(data=ad_data, x='Daily Time Spent on Site', y='Daily Internet Usage', color='green')
sns.pairplot(data=ad_data, hue='Clicked on Ad', palette='bwr')
from sklearn.model_selection import train_test_split
cleaned_ad_data = ad_data.drop(['Ad Topic Line', 'City', 'Country', 'Timestamp'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(cleaned_ad_data.drop('Clicked on Ad', axis=1), cleaned_ad_data['Clicked on Ad'], test_size=0.3)
from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()
lm.fit(X=X_train, y=y_train)
predictions = lm.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))