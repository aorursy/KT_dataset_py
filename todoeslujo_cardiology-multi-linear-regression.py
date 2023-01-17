import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns;
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
df=pd.read_excel("../input/CNumerical.xls", "Sheet1")
print(df.head(10))
import statsmodels.api as sm

bpchol = np.array(df['blood pressure'], df['cholesterol'])
mhr = np.array(df['maximum heart rate'])

try_bpchol = sm.add_constant(bpchol)
res = sm.OLS(mhr, try_bpchol).fit()
print(res.params)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
model = sm.OLS(mhr, try_bpchol)
results_formula = model.fit()
results_formula.params 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

x_surf, y_surf = np.meshgrid(np.linspace(df['blood pressure'].min(), df['blood pressure'].max()),np.linspace(df['cholesterol'].min(), df['cholesterol'].max()))

onlyX = pd.DataFrame({'blood pressure': x_surf.ravel(), 'cholesterol': y_surf.ravel()})

fittedY=results_formula.predict(exog=onlyX)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['blood pressure'],df['cholesterol'],df['maximum heart rate'],c='blue', marker='o', alpha=0.5)
ax.plot_surface(x_surf,y_surf,fittedY.reshape(x_surf.shape), color='None', alpha=0.01)
ax.set_xlabel('blood pressure')
ax.set_ylabel('cholesterol')
ax.set_zlabel('maximum heart rate')
ax.set_zlim([df['maximum heart rate'].min(),df['maximum heart rate'].max()]) 
plt.show()
