# Import dem libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

from matplotlib.ticker import MultipleLocator

import seaborn as sns

from datetime import datetime
intel_dataframe = pd.read_csv("../input/Intel_CPUs.csv")



# Let's see the columns it has

intel_dataframe.head()
# I have to drop the ones without Launch Date as this analysis needs it

# to sort using the Launch Date :(

intel_dataframe = intel_dataframe[np.invert(intel_dataframe["Launch_Date"].isnull())]
# Now we need to sort the data by Quarter

# Sadly pandas doesn't have much built-in functionality

def sortmap(x):

    firstPart = x[:2]

    if firstPart[0] == "0":

        temp = list(firstPart)

        temp[0] = "Q"

        firstPart = "".join(temp)

    try:

        restOfIt = int(x[3:])

    except ValueError:

        restOfIt = int(x[4:])

    matcher = {

        "Q1": "-02",

        "Q2": "-05",

        "Q3": "-08",

        "Q4": "-11"

    }

    date = ("20" if restOfIt < 18 else "19") + str(restOfIt).rjust(2,"0") + matcher[firstPart]

    return date



intel_dataframe["Launch_Date"] = intel_dataframe["Launch_Date"].apply(sortmap)
intel_sorted = intel_dataframe.sort_values(by="Launch_Date")
# Handle the strings in processor speed

def ProcessorMapper(x):

    value = int(float(x[:-4]))

    if x[-3] == "G":

        value *= 1000

    return value
# Sns settings

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

sns.set_palette(flatui)

sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
def cleanify(tag, func):

    clean_df = intel_sorted[np.invert(intel_sorted[tag].isnull())].copy()

    clean_df.reset_index()

    clean_df[tag] = clean_df[tag].apply(func)

    return clean_df

intel_processor_base_freq = cleanify("Processor_Base_Frequency", ProcessorMapper)
# Smoothify graph by taking a mean

def smoothify(tag, df, decreasing=False):

    val_till_now = 0

    if decreasing:

        val_till_now = df[tag].max()

    groups = df.groupby('Launch_Date')[tag].max().reset_index()

    for index, values in groups.iterrows():

        value = groups.loc[index,tag]

        if decreasing:

            if value < val_till_now:

                val_till_now = value

        else:

            if value > val_till_now:

                val_till_now = value

        groups.loc[index,tag] = val_till_now

    return groups

smooth_intel_processor_base_freq = smoothify("Processor_Base_Frequency", intel_processor_base_freq)
def plot_graph(df, tag, title, y_multiples=250):

    fig, ax = plt.subplots(figsize=(10, 7))



    ax.plot(

        pd.to_datetime(df["Launch_Date"]),

        df[tag]

    )

    hfmt = mdates.DateFormatter("%Y")



    ax.xaxis.set_major_formatter(hfmt)

    ax.yaxis.set_label_text(title)

    ax.xaxis.set_major_locator(mdates.YearLocator())

    ax.format_xdata = hfmt



    ax.yaxis.set_major_locator(MultipleLocator(y_multiples))



    plt.show()

plot_graph(smooth_intel_processor_base_freq, "Processor_Base_Frequency", "Max Processor Frequency available (in MHz)")
# Remove the nan ones

intel_chip_lithography = cleanify("Lithography", lambda x: int(x[:-3]))
smooth_intel_chip_lithography = smoothify("Lithography", intel_chip_lithography, decreasing=True)
plot_graph(smooth_intel_chip_lithography, "Lithography", "Thickness (in nm)", y_multiples=15)
def doItAllForMe(tag, y_axis_text, y_multiples=250, y_axis_fmt_func=lambda x: x, pure=False, decreasing=False):

    value = cleanify(tag, y_axis_fmt_func)

    if not pure:

        plot_graph(smoothify(tag, value, decreasing), tag, y_axis_text, y_multiples)

    else:

        plot_graph(value, tag, y_axis_text, y_multiples)

doItAllForMe("nb_of_Threads", "No of Threads", 5, lambda x: int(x))

doItAllForMe("nb_of_Cores", "No of Cores", 5, lambda x: int(x))
intel_sorted[intel_sorted["Launch_Date"] == "2012-11"].loc[:, ["Product_Collection", "nb_of_Cores"]]
import re



def PriceProcessor(x):

    x = x.replace(",", "")

    match = re.match("\$([0-9]*\.[0-9]*)", x)

    found = match.groups()

    ans = float(found[0])

    return ans
doItAllForMe("Recommended_Customer_Price", "Recommended Customer Price", 1000, PriceProcessor)
intel_sorted["Status"].value_counts().plot(kind='pie')
intel_sorted["Instruction_Set"].value_counts().plot(kind='pie')
def RatioFinder(x):

    ins_32 = x[x["Instruction_Set"] == "32-bit"].shape[0]

    ins_64 = x[x["Instruction_Set"] == "64-bit"].shape[0]

    if ins_32 == ins_64:

        return 1.

    if ins_32 == 0:

        return 1.

    return ins_64/ins_32/x.shape[0]
year_intel = intel_sorted.copy()

year_intel["Launch_Date"] = year_intel["Launch_Date"].apply(lambda x: x[:4])

df = pd.DataFrame(intel_sorted.groupby(by="Launch_Date").apply(RatioFinder))

df["Launch_Date"] = df.index

#plot_graph(df=lol,tag="Instruction_Set",title="64 bit processors released",y_multiples=2)
fig, ax = plt.subplots(figsize=(10, 7))



ax.plot(

    pd.to_datetime(df.loc[::2,"Launch_Date"]),

    df.iloc[::2, 0]

)

hfmt = mdates.DateFormatter("%Y")



ax.xaxis.set_major_formatter(hfmt)

ax.yaxis.set_label_text("64 vs 32 bit")

ax.xaxis.set_major_locator(mdates.YearLocator())

ax.format_xdata = hfmt



ax.yaxis.set_major_locator(MultipleLocator(0.1))



plt.show()