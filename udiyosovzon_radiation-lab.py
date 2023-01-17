import re

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

import cufflinks as cf



import statsmodels.api as sm



# display and layout settings

sns.set(font_scale=1.25, rc={"figure.figsize": (12,6.67)})

cf.go_offline()
# file names of text files containing measurements of nano particles

file_names = ["dw.txt", "25nm - diluted.txt", "25nm - full.txt", "65nm - full.txt", "210nm - full.txt"]

folder = "../input/nanoparticle_measurements/"



# load each measurement data to a numpy array

data = {file[:-4]: np.loadtxt(fname=folder+file, skiprows=17, max_rows=651) for file in file_names}

data_keys = list(data.keys())[1:]



# take radiation mesurments and store in pandas Datafraame

radiation_data = {key: d[:,1] for key, d in data.items()}

wl = data["dw"][:,0]

radiation_data = pd.DataFrame(radiation_data, index=pd.Index(data=wl, name="wavelength"))





# show first 5 rows

radiation_data.head()
# length of trajectory through medium in nano meters

L = 10**7    # eqals to 1 cm
def calculate_sigma(a, n_med, n_par):

    

    """

    Returns a function that computes the dispersion cross section for wavelength values

    

    Parameters

    ----------

    a: float

        Mean diameter of particles in nano-meter

        

    n_med: float

        Refractive index of medium

        

    n_par: float

        Refractive index of particles

    """

    pi = np.pi

    m2 = (n_par / n_med)**2    # m squared

    

    # wl - wavelength

    return lambda wl: ( 8*pi/3 ) * ( a**2 ) * ( (2*pi*a*n_med/wl)**4 ) * ( ((m2-1)/(m2+2))**2 )
# extract diameter

diameters = np.array([int(re.match(r"\d+nm", key)[0][:-2]) for key in data_keys])



# calculate cross section values for each diamter for wavelength equals 1

cross_sections = calculate_sigma(diameters, 1.4, 1.6)(1)



# store in pandas Series

sigma = pd.Series(data=cross_sections, index=data_keys)



sigma
def calculate_density(a, x):

    

    """

    Calculate number density.

    

    Parameters

    ----------

    a: float

        Mean diameter of particles in nano-meter

        

    x: float

        Mass density in percentage

    """

    return (5.66 * x) / ((100-x) * np.pi * (a**3))
# store mass concentration value in pandas Series

mass_con = [0.0001, 0.001, 6.7e-05, 5e-06]

mass_con = pd.Series(data=mass_con, index=data_keys)



mass_con
# calculate denisty, store results in pandas DataFrame

density = calculate_density(diameters, mass_con)



density
# create X vector

X = (1/(wl**4))





# create Y vector

I0 = radiation_data["dw"]

I = radiation_data.drop("dw", axis=1)



y_numerator = -np.log(I.divide(I0, axis="rows"))

y_denominator = sigma * L



Y = y_numerator.divide(y_denominator).set_index(X)



Y.head()
Y = Y.where(Y >= 0, np.nan)



Y.head()
# return a tuple (min, max) for array a

minmax = lambda a: (a.min(), a.max())



# functions for pretty print print digits after decimal point

sc_print = lambda x: np.format_float_scientific(x, precision=4, unique=False)    # scientific notation

dc_print = lambda x: np.around(np.abs(x), decimals=3)    # float notation
def linear_model(key, mask=None):

    

    """

    Calculate OLS model, present data visually

    """

    # extract relevant values for x and y

    if mask is None:

        key_values = Y[key].dropna()

    else:

        key_values = Y[key][mask].dropna()

    

    x = key_values.index.to_numpy()

    y = key_values.values



    # extract and store theoretical value for number density

    theo_val = density[key]



    ### calculate linear model OLS ###

    ##################################

    model = sm.OLS(endog=y, exog=sm.add_constant(x)).fit()

    a0, a1 = model.params    # get params

    s0, s1 = model.bse    # get standard error



    #### plot ####

    ##############

    fig, ax = plt.subplots()

    line_label = f"y = {sc_print(a0)} + {sc_print(a1)} * x"

    sns.regplot(x,y, ax=ax, scatter_kws={"color": "lightblue"}, line_kws={"label": line_label})



    # layout

    x_units = "[1/nm\N{SUPERSCRIPT FOUR}]"

    y_units = "[1/nm\N{SUPERSCRIPT SEVEN}]"

    a1_units = "[1/nm\N{SUPERSCRIPT THREE}]"

    ax.set(xlim=minmax(x), ylim=minmax(y), xlabel=f"x {x_units}", ylabel=f"y {y_units}")

    ax.set_title(key, y=1.05, fontsize=26, bbox=dict(facecolor="paleturquoise"))

    ax.legend(facecolor="white", loc="upper left")



    # create summary text 

    summary = f"""

    {"-"*75}

    R-squared{" "*18}{dc_print(model.rsquared)}

    Fitted value (a1){" "*9}{sc_print(a1)} \u00B1 {sc_print(s1)} {a1_units}

    Theoretical value{" "*7}{sc_print(theo_val)} {a1_units}

    Deviation{" "*20}{dc_print(np.abs(a1-theo_val)*100 / theo_val)} %

    {"-"*75}"""

    

    

    # add summary text

    ax.annotate(summary, (0,0), (0, -50), xycoords="axes fraction", textcoords="offset points", va="top")



    plt.show()

    

    return model
radiation_data.iplot(xTitle="Wavelength [nm]", yTitle="Radiation passed")
for key in data_keys:

    model = linear_model(key)
mask = (wl > 400) & (wl < 850)



for key in data_keys:

    model = linear_model(key, mask)

mask = (wl > 600) & (wl < 850)



for key in data_keys[1:]:

    model = linear_model(key, mask)