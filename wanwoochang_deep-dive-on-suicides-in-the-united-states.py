import numpy as np

import pandas as pd



from bokeh.models import ColumnDataSource, FactorRange, HoverTool, formatters

from bokeh.plotting import figure, output_notebook, show



import os
output_notebook()
pd.set_option("display.float_format", lambda x: "%.3f" %x)
df = pd.read_csv("../input/master.csv")
#Column rename

df.columns = ["country", "year", "sex", "age", "suicides", "pop", "suicides/100k", "c-y", "hdi", "gdp_year", "gdp_capita", "cohort"]



#Drop unnecessary columns. Suicides per 100k population is a rate which can be recalculated with reaggregated tables.

df.drop(labels=["suicides/100k", "c-y"], axis=1, inplace=True)
#Data recasting & cleanup

df["year"] = df["year"].astype(str)

df["decade"] = df["year"].apply(lambda x: x[:-1] + "0")



df["age"] = df["age"].str.replace(pat=" years", 

                                  repl="")



df["gdp_year"] = pd.to_numeric(df["gdp_year"].str.replace(pat=r"[^\d]", repl="", regex=True), 

                               errors="coerce")
us = df[df["country"]=="United States"].copy()



us.reset_index(drop=True, inplace=True)
#Creating manually ordered groupings. To be used with bokeh visualizations.

sorted_decades = ["1980", "1990", "2000", "2010"]

sorted_cohorts = ["G.I. Generation", "Silent", "Boomers", "Generation X", "Millenials", "Generation Z"]

sorted_ages = ["5-14", "15-24", "25-34", "35-54", "55-74", "75+"]



sorted_dec_age = [(d, a) for d in sorted_decades for a in sorted_ages]

sorted_coh_age = [(c, a) for c in sorted_cohorts for a in sorted_ages]
#Aggregations used for visualization purposes.

aggs ={"suicides": "sum",

       "pop": "sum",

       "hdi": "mean",

       "gdp_capita": "mean"

      }



us_dec_age = us.groupby(["decade", "age"]).agg(aggs)

us_dec_age["suicides_per_100k"] = us_dec_age["suicides"]/(us_dec_age["pop"]/100000)



us_coh_age = us.groupby(["cohort", "age"]).agg(aggs)

us_coh_age["suicides_per_100k"] = us_coh_age["suicides"]/(us_coh_age["pop"]/100000)
#Visualizing Suicides by decade and age bracket



p1 = figure(title="Suicides in the United States by Decade and Age Bracket",

            x_range=FactorRange(*sorted_dec_age),

            x_axis_label="Age Groups by Decade",

            y_axis_label="Suicides in the US",

            y_axis_type="linear",

            tools="pan,tap,wheel_zoom,undo,reset",

            toolbar_location="below",

            width=1350

           )



p1.vbar(source=ColumnDataSource(us_dec_age),

        x="decade_age",

        top="suicides",

        width=0.5,

        hover_line_color="orange",

        hover_fill_color="orange"

)



p1_tooltip = [("Decade & Age Group", "@decade_age"), 

              ("# of Suicides", "@suicides{0,0}")]



p1.add_tools(HoverTool(tooltips=p1_tooltip))



p1.yaxis.formatter = formatters.NumeralTickFormatter(format="0a")



show(p1)
#Visualizing Suicides by Cohort and Age Bracket



p2 = figure(title="Suicides in the United States by Cohort and Age Bracket",

            x_range=FactorRange(*sorted_coh_age),

            x_axis_label="Age Groups by Cohort",

            y_axis_label="Suicides in the US",

            y_axis_type="linear",

            tools="pan,tap,wheel_zoom,undo,reset",

            toolbar_location="below",

            width=1350

           )



p2.vbar(source=ColumnDataSource(us_coh_age),

        x="cohort_age",

        top="suicides",

        width=0.5,

        hover_line_color="orange",

        hover_fill_color="orange"

)



p2_tooltip = [("Cohort & Age Group", "@cohort_age"), 

              ("# of Suicides", "@suicides{0,0}")]



p2.add_tools(HoverTool(tooltips=p2_tooltip))



p2.yaxis.formatter = formatters.NumeralTickFormatter(format="0a")



show(p2)
us_3554 = us[us["age"]=="35-54"].copy()



us_3554.reset_index(drop=True, inplace=True)
us_3554_pt = us_3554.groupby(["year", "cohort", "sex"]).agg(aggs)



us_3554_pt["suicides_per_100k"] = us_3554_pt["suicides"]/(us_3554_pt["pop"]/100000)
#Visualizing Suicides by Gender and Cohort over time.



p3 = figure(title="Suicides in the United States by Cohort and Age Bracket",

            x_range=FactorRange(*ColumnDataSource(us_3554_pt).data["year_cohort_sex"]),

            x_axis_label="Cohort and Gender Over Time",

            y_axis_label="Suicides in the US",

            y_axis_type="linear",

            tools="pan,tap,wheel_zoom,undo,reset",

            toolbar_location="below",

            width=1350

           )



p3.vbar(source=ColumnDataSource(us_3554_pt),

        x="year_cohort_sex",

        top="suicides",

        width=0.5,

        hover_line_color="orange",

        hover_fill_color="orange"

)



p3_tooltip = [("Year, Cohort, Sex", "@year_cohort_sex"), 

              ("# of Suicides", "@suicides{0,0}")]



p3.add_tools(HoverTool(tooltips=p3_tooltip))



p3.yaxis.formatter = formatters.NumeralTickFormatter(format="0a")



show(p3)
#Visualizing Suicides by Gender and Cohort over time.



p4 = figure(title="Suicides in the United States by Cohort and Age Bracket",

            x_range=FactorRange(*ColumnDataSource(us_3554_pt).data["year_cohort_sex"]),

            x_axis_label="Cohort and Gender Over Time",

            y_axis_label="Suicides per 100k in the US",

            y_axis_type="linear",

            tools="pan,tap,wheel_zoom,undo,reset",

            toolbar_location="below",

            width=1350

           )



p4.vbar(source=ColumnDataSource(us_3554_pt),

        x="year_cohort_sex",

        top="suicides_per_100k",

        width=0.5,

        hover_line_color="orange",

        hover_fill_color="orange"

)



p4_tooltip = [("Year, Cohort, Sex", "@year_cohort_sex"), 

              ("# of Suicides per 100k", "@suicides_per_100k{0.00a}")]



p4.add_tools(HoverTool(tooltips=p4_tooltip))



p4.yaxis.formatter = formatters.NumeralTickFormatter(format="0a")



show(p4)