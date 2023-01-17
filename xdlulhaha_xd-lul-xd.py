import numpy as np

import pandas as pd



%matplotlib inline
autos = pd.read_csv('../input/autos.csv', encoding='ISO-8859-1')



autos = autos[

    (autos.price > autos.price.quantile(0.05)) & (autos.price < autos.price.quantile(0.95)) &

    (autos.powerPS > autos.powerPS.quantile(0.05)) & (autos.powerPS < autos.powerPS.quantile(0.95))

]
autos['pricePerPower'] = autos.price / autos.powerPS



autos.groupby("brand").aggregate({'pricePerPower': ['mean', 'median']}).sort_values(('pricePerPower', 'mean'), ascending=False)
import datetime



now = datetime.datetime.now()



autos['ageInMonths'] = (

    (now.year * 12 + now.month) - (autos.yearOfRegistration * 12 + autos.monthOfRegistration)

)



autos = autos[(autos.ageInMonths >= 0) & (autos.ageInMonths <= 50 * 12)]

autos.ageInMonths.plot.hist(rwidth=0.9)
(autos.groupby('brand').ageInMonths.mean() / 12).sort_values(ascending=False)
autos[(autos.ageInMonths <= 20 * 12)].plot.scatter('ageInMonths', 'price', alpha=0.002)