# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from bokeh.plotting import figure

from bokeh.io import output_file , show

from bokeh.plotting import ColumnDataSource

from bokeh.layouts import gridplot,row,column



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
lit_birth_rate=pd.read_csv('/kaggle/input/visualizeusingbokeh/literacy_birth_rate.csv')

africa_df=lit_birth_rate[lit_birth_rate['Continent']=='AF']

america_df=lit_birth_rate[lit_birth_rate['Continent']=='LAT']

asia_df=lit_birth_rate[lit_birth_rate['Continent']=='ASI']

europe_df=lit_birth_rate[lit_birth_rate['Continent']=='EUR']



fertility_africa=africa_df['fertility']

female_literacy_africa=africa_df['female literacy']

fertility_latinamerica=america_df['fertility']

female_literacy_latinamerica=america_df['female literacy']



fertility_asia=asia_df['fertility']

female_literacy_asia=asia_df['female literacy']



fertility_europe=europe_df['fertility']

female_literacy_europe=europe_df['female literacy']
source=ColumnDataSource(africa_df)

# Create a blank figure: p1

p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)',title="Africa")



# Add circle scatter to the figure p1

p1.circle('fertility', 'female literacy',source=source)



p2 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)',title="Latin America")



# Add circle scatter to the figure p1

p2.circle(fertility_latinamerica, female_literacy_latinamerica)



p3 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)',title="Asia")



# Add circle scatter to the figure p1

p3.circle(fertility_asia, female_literacy_asia)





p4 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)',title="Europe")



# Add circle scatter to the figure p1

p4.circle(fertility_europe, female_literacy_europe)
# Link the x_range of p2 to p1: p2.x_range

p2.x_range = p1.x_range



# Link the y_range of p2 to p1: p2.y_range

p2.y_range = p1.y_range



# Link the x_range of p3 to p1: p3.x_range

p3.x_range=p1.x_range



# Link the y_range of p4 to p1: p4.y_range

p4.y_range=p1.y_range



layout=gridplot([[p1,p2],[p3,p4]])

# Specify the name of the output_file and show the result

output_file('linked_range.html')

show(layout)

# Create ColumnDataSource: source

source = ColumnDataSource(lit_birth_rate)



# Create the first figure: p1

p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female literacy (% population)',

            tools='box_select,lasso_select')



# Add a circle glyph to p1

p1.circle('fertility','female literacy',source=source)



# Create the second figure: p2

p2 = figure(x_axis_label='fertility (children per woman)', y_axis_label='population (millions)',

            tools='box_select,lasso_select' )



# Add a circle glyph to p2

p2.circle('fertility','population',source=source)



# Create row layout of figures p1 and p2: layout

layout = row(p1,p2)



# Specify the name of the output_file and show the result

output_file('linked_brush.html')

show(layout)
latin_america=ColumnDataSource(america_df)

africa=ColumnDataSource(africa_df)
p=figure(x_axis_label='Fertility',y_axis_label='Literacy rate')
# Add the first circle glyph to the figure p

p.circle('fertility', 'female literacy', source=latin_america, size=10, color='red', legend_label='Latin America')



# Add the second circle glyph to the figure p

p.circle('fertility', 'female literacy', source=africa, size=10, color='blue', legend_label='Africa')



# Specify the name of the output_file and show the result

output_file('fert_lit_groups.html')

show(p)

from bokeh.io import curdoc

from bokeh.plotting import figure



# Create a new plot: plot

plot = figure(x_axis_label=" ",y_axis_label=" ")



# Add a line to the plot

plot.line([1,2,3,4,5],[2,5,4,6,7])



# Add the plot to the current document

curdoc().add_root(plot)
from bokeh.layouts import Column

from bokeh.models import Slider



# Create a slider: slider

slider = Slider(title='my slider', start=0, end=10, step=0.1, value=2)



# Create a widgetbox layout: layout

layout = Column(slider)



# Add the layout to the current document

curdoc().add_root(layout)




# Create first slider: slider1

slider1 = Slider(title="slider1",start=0,end=10,step=0.1, value=2)



# Create second slider: slider2

slider2 = Slider(title="slider2",start=10,end=100,step=1, value=20)



# Add slider1 and slider2 to a widgetbox

layout = Column(slider1,slider2)



# Add the layout to the current document

curdoc().add_root(layout)
x=[  0.3, 0.33244147  , 0.36488294  , 0.39732441  , 0.42976589

,0.46220736,0.49464883,0.5270903, 0.55953177,0.59197324

,0.62441472,0.65685619,0.68929766,0.72173913,0.7541806

,0.78662207,0.81906355,0.85150502,0.88394649,0.91638796

,0.94882943,0.9812709, 1.01371237,1.04615385,1.07859532

,1.11103679,1.14347826,1.17591973,1.2083612, 1.24080268

,1.27324415,1.30568562,1.33812709,1.37056856,1.40301003

,1.43545151,1.46789298,1.50033445,1.53277592,1.56521739

,1.59765886,1.63010033,1.66254181,1.69498328,1.72742475

,1.75986622,1.79230769,1.82474916,1.85719064,1.88963211

,1.92207358,1.95451505,1.98695652,2.01939799,2.05183946

,2.08428094,2.11672241,2.14916388,2.18160535,2.21404682

,2.24648829,2.27892977,2.31137124,2.34381271,2.37625418

,2.40869565,2.44113712,2.4735786, 2.50602007,2.53846154

,2.57090301,2.60334448,2.63578595,2.66822742,2.7006689

,2.73311037,2.76555184,2.79799331,2.83043478,2.86287625

,2.89531773,2.9277592, 2.96020067,2.99264214,3.02508361

,3.05752508,3.08996656,3.12240803,3.1548495, 3.18729097

,3.21973244,3.25217391,3.28461538,3.31705686,3.34949833

,3.3819398, 3.41438127,3.44682274,3.47926421,3.51170569

,3.54414716,3.57658863,3.6090301, 3.64147157,3.67391304

,3.70635452,3.73879599,3.77123746,3.80367893,3.8361204

,3.86856187,3.90100334,3.93344482,3.96588629,3.99832776

,4.03076923,4.0632107, 4.09565217,4.12809365,4.16053512

,4.19297659,4.22541806,4.25785953,4.290301,  4.32274247

,4.35518395,4.38762542,4.42006689,4.45250836,4.48494983

,4.5173913, 4.54983278,4.58227425,4.61471572,4.64715719

,4.67959866,4.71204013,4.74448161,4.77692308,4.80936455

,4.84180602,4.87424749,4.90668896,4.93913043,4.97157191

,5.00401338,5.03645485,5.06889632,5.10133779,5.13377926

,5.16622074,5.19866221,5.23110368,5.26354515,5.29598662

,5.32842809,5.36086957,5.39331104,5.42575251,5.45819398

,5.49063545,5.52307692,5.55551839,5.58795987,5.62040134

,5.65284281,5.68528428,5.71772575,5.75016722,5.7826087

,5.81505017,5.84749164,5.87993311,5.91237458,5.94481605

,5.97725753,6.009699,  6.04214047,6.07458194,6.10702341

,6.13946488,6.17190635,6.20434783,6.2367893, 6.26923077

,6.30167224,6.33411371,6.36655518,6.39899666,6.43143813

,6.4638796, 6.49632107,6.52876254,6.56120401,6.59364548

,6.62608696,6.65852843,6.6909699, 6.72341137,6.75585284

,6.78829431,6.82073579,6.85317726,6.88561873,6.9180602

,6.95050167,6.98294314,7.01538462,7.04782609,7.08026756

,7.11270903,7.1451505, 7.17759197,7.21003344,7.24247492

,7.27491639,7.30735786,7.33979933,7.3722408, 7.40468227

,7.43712375,7.46956522,7.50200669,7.53444816,7.56688963

,7.5993311, 7.63177258,7.66421405,7.69665552,7.72909699

,7.76153846,7.79397993,7.8264214, 7.85886288,7.89130435

,7.92374582,7.95618729,7.98862876,8.02107023,8.05351171

,8.08595318,8.11839465,8.15083612,8.18327759,8.21571906

,8.24816054,8.28060201,8.31304348,8.34548495,8.37792642

,8.41036789,8.44280936,8.47525084,8.50769231,8.54013378

,8.57257525,8.60501672,8.63745819,8.66989967,8.70234114

,8.73478261,8.76722408,8.79966555,8.83210702,8.86454849

,8.89698997,8.92943144,8.96187291,8.99431438,9.02675585

,9.05919732,9.0916388, 9.12408027,9.15652174,9.18896321

,9.22140468,9.25384615,9.28628763,9.3187291, 9.35117057

,9.38361204,9.41605351,9.44849498,9.48093645,9.51337793

,9.5458194, 9.57826087,9.61070234,9.64314381,9.67558528

,9.70802676,9.74046823,9.7729097, 9.80535117,9.83779264

,9.87023411,9.90267559,9.93511706,9.96755853 , 10  ]
y=[-0.19056796,0.13314778,0.39032789,0.58490071,0.72755027,0.82941604

,0.90008145,0.94719898,0.97667411,0.99299073,0.99952869,0.99882928

,0.99280334,0.98288947,0.97017273,0.95547297,0.93941048,0.92245495

,0.90496191,0.88720012,0.86937208,0.85162961,0.83408561,0.81682308

,0.79990193,0.78336433,0.76723876,0.75154314,0.7362873, 0.72147487

,0.70710477,0.69317237,0.67967038,0.66658956,0.65391928,0.64164796

,0.62976339,0.61825301,0.60710407,0.59630386,0.58583975,0.57569933

,0.56587047,0.55634135,0.5471005, 0.53813683,0.52943965,0.52099866

,0.51280394,0.50484599,0.49711569,0.48960429,0.48230342,0.47520507

,0.46830157,0.4615856, 0.45505012,0.44868845,0.44249417,0.43646114

,0.43058352,0.42485569,0.4192723, 0.41382821,0.40851854,0.40333859

,0.39828387,0.39335008,0.38853312,0.38382904,0.37923407,0.37474459

,0.37035715,0.36606841,0.3618752, 0.35777446,0.35376325,0.34983877

,0.34599831,0.34223928,0.33855919,0.33495564,0.33142632,0.32796903

,0.32458163,0.32126208,0.3180084, 0.3148187, 0.31169115,0.30862399

,0.30561552,0.30266411,0.29976818,0.29692621,0.29413673,0.29139834

,0.28870966,0.28606938,0.28347622,0.28092895,0.27842639,0.27596739

,0.27355084,0.27117567,0.26884083,0.26654532,0.26428818,0.26206846

,0.25988525,0.25773767,0.25562487,0.25354602,0.25150031,0.24948698

,0.24750527,0.24555444,0.24363379,0.24174264,0.23988032,0.23804617

,0.23623958,0.23445993,0.23270663,0.2309791, 0.2292768, 0.22759917

,0.22594568,0.22431583,0.22270912,0.22112506,0.21956318,0.21802302

,0.21650414,0.2150061, 0.21352848,0.21207087,0.21063286,0.20921408

,0.20781413,0.20643266,0.20506929,0.20372368,0.20239549,0.20108438

,0.19979003,0.19851212,0.19725034,0.19600439,0.19477398,0.19355882

,0.19235862,0.19117313,0.19000206,0.18884517,0.18770219,0.18657288

,0.18545699,0.1843543, 0.18326456,0.18218756,0.18112306,0.18007087

,0.17903076,0.17800253,0.17698598,0.17598091,0.17498713,0.17400446

,0.1730327, 0.17207168,0.17112122,0.17018115,0.1692513, 0.16833151

,0.16742161,0.16652145,0.16563087,0.16474972,0.16387786,0.16301513

,0.16216139,0.16131651,0.16048035,0.15965278,0.15883366,0.15802286

,0.15722027,0.15642575,0.15563919,0.15486047,0.15408947,0.15332608

,0.15257018,0.15182167,0.15108044,0.15034639,0.14961941,0.14889939

,0.14818625,0.14747988,0.14678019,0.14608708,0.14540046,0.14472024

,0.14404634,0.14337866,0.14271712,0.14206163,0.14141212,0.1407685

,0.14013069,0.13949862,0.1388722, 0.13825137,0.13763605,0.13702616

,0.13642163,0.1358224, 0.13522839,0.13463954,0.13405578,0.13347705

,0.13290327,0.1323344, 0.13177035,0.13121109,0.13065653,0.13010663

,0.12956133,0.12902056,0.12848428,0.12795242,0.12742494,0.12690177

,0.12638288,0.12586819,0.12535768,0.12485127,0.12434893,0.12385061

,0.12335625,0.12286581,0.12237925,0.12189652,0.12141757,0.12094236

,0.12047084,0.12000298,0.11953873,0.11907805,0.1186209, 0.11816724

,0.11771703,0.11727022,0.11682679,0.11638669,0.11594988,0.11551634

,0.11508601,0.11465888,0.11423489,0.11381403,0.11339624,0.11298151

,0.11256979,0.11216106,0.11175527,0.11135241,0.11095243,0.11055531

,0.11016102,0.10976953,0.1093808, 0.10899481,0.10861153,0.10823093

,0.10785298,0.10747766,0.10710493,0.10673478,0.10636717,0.10600208

,0.10563948,0.10527936,0.10492167,0.1045664, 0.10421352,0.10386302

,0.10351486,0.10316902,0.10282548,0.10248422,0.10214521,0.10180843

,0.10147386,0.10114148,0.10081127,0.1004832, 0.10015726,0.09983342]
source = ColumnDataSource(data={'x': x, 'y': y})



# Add a line to the plot

plot.line('x', 'y', source=source)



# Create a column layout: layout

layout = column(Column(slider), plot)



# Add the layout to the current document

curdoc().add_root(layout)
source = ColumnDataSource(data={'x': x, 'y': y})



# Add a line to the plot

plot.line('x', 'y', source=source)



# Define a callback function: callback

def callback(attr, old, new):



    # Read the current value of the slider: scale

    scale = slider.value



    # Compute the updated y using np.sin(scale/x): new_y

    new_y = np.sin(scale/x)



    # Update source with the new data values

    source.data = {'x': x, 'y': new_y}



# Attach the callback to the 'value' property of slider

slider.on_change('value',callback)



# Create layout and add to current document

layout = column(Column(slider), plot)

curdoc().add_root(layout)
# Perform necessary imports

from bokeh.models import ColumnDataSource, Select



# Create ColumnDataSource: source

source = ColumnDataSource(lit_birth_rate)



# Create a new plot: plot

plot = figure()



# Add circles to the plot

plot.circle('fertility', 'female literacy', source=source)



# Define a callback function: update_plot

def update_plot(attr, old, new):

    # If the new Selection is 'female_literacy', update 'y' to female_literacy

    if new == 'female_literacy': 

        source.data = {

            'x' :lit_birth_rate[' fertility'],

            'y' : lit_birth_rate['female literacy']

        }

    # Else, update 'y' to population

    else:

        source.data = {

            'x' : lit_birth_rate[' fertility'],

            'y' : lit_birth_rate[' population']

        }



# Create a dropdown Select widget: select    

select = Select(title="distribution", options=['female_literacy', 'population'], value='female_literacy')



# Attach the update_plot callback to the 'value' property of select

select.on_change('value', update_plot)



# Create layout and add to current document

layout = row(select, plot)

curdoc().add_root(layout) 
# Create two dropdown Select widgets: select1, select2

select1 = Select(title='First', options=['A', 'B'], value='A')

select2 = Select(title='Second', options=['1', '2', '3'], value='1')



# Define a callback function: callback

def callback(attr, old, new):

    # If select1 is 'A' 

    if select1.value == 'A':

        # Set select2 options to ['1', '2', '3']

        select2.options = ['1', '2', '3']



        # Set select2 value to '1'

        select2.value = '1'

    else:

        # Set select2 options to ['100', '200', '300']

        select2.options = ['100', '200', '300']



        # Set select2 value to '100'

        select2.value = '100'



# Attach the callback to the 'value' property of select1

select1.on_change('value', callback)



# Create layout and add to current document

layout = Column(select1, select2)

curdoc().add_root(layout)
from bokeh.models import Button

# Create a Button with label 'Update Data'

button = Button(label='Update Data')



# Define an update callback with no arguments: update

def update():



    # Compute new y values: y

    y = np.sin(x) + np.random.random(N)



    # Update the ColumnDataSource data dictionary

    source.data = {'x': x, 'y': y}



# Add the update callback to the button

button.on_click(update)



# Create layout and add to current document

layout = column(Column(button), plot)

curdoc().add_root(layout)
# Import CheckboxGroup, RadioGroup, Toggle from bokeh.models

from bokeh.models import CheckboxGroup,RadioGroup,Toggle



# Add a Toggle: toggle

toggle = Toggle(button_type='success',label='Toggle button')



# Add a CheckboxGroup: checkbox

checkbox = CheckboxGroup(labels=['Option 1', 'Option 2', 'Option 3'])



# Add a RadioGroup: radio

radio = RadioGroup(labels=['Option 1', 'Option 2', 'Option 3'])



# Add widgetbox(toggle, checkbox, radio) to the current document

curdoc().add_root(Column(toggle, checkbox, radio))
gapminder_df=pd.read_csv('/kaggle/input/visualizeusingbokeh/gapminder_tidy.csv')
# Perform necessary imports

from bokeh.io import output_file, show

from bokeh.plotting import figure

from bokeh.models import HoverTool, ColumnDataSource



# Make the ColumnDataSource: source

source = ColumnDataSource(data={

    'x'       : gapminder_df[gapminder_df['Year']==1970].fertility,

    'y'       : gapminder_df[gapminder_df['Year']==1970].life,

    'country' : gapminder_df[gapminder_df['Year']==1970].Country,

})



# Create the figure: p

p = figure(title='1970', x_axis_label='Fertility (children per woman)', y_axis_label='Life Expectancy (years)',

           plot_height=400, plot_width=700,

           tools=[HoverTool(tooltips='@country')])



# Add a circle glyph to the figure p

p.circle(x='x', y='y', source=source)



# Output the file and show the figure

output_file('gapminder.html')

show(p)
# Import the necessary modules

from bokeh.io import curdoc

from bokeh.models import ColumnDataSource

from bokeh.plotting import figure



# Make the ColumnDataSource: source

source = ColumnDataSource(data={

    'x'       : gapminder_df[gapminder_df['Year']==1970].fertility,

    'y'       : gapminder_df[gapminder_df['Year']==1970].life,

    'country'      : gapminder_df[gapminder_df['Year']==1970].Country,

    'pop'      : (gapminder_df[gapminder_df['Year']==1970].population / 20000000) + 2,

    'region'      : gapminder_df[gapminder_df['Year']==1970].region,

})



# Save the minimum and maximum values of the fertility column: xmin, xmax

xmin, xmax = min(gapminder_df.fertility), max(gapminder_df.fertility)



# Save the minimum and maximum values of the life expectancy column: ymin, ymax

ymin, ymax = min(gapminder_df.life), max(gapminder_df.life)



# Create the figure: plot

plot = figure(title='Gapminder Data for 1970', plot_height=400, plot_width=700,

              x_range=(xmin, xmax), y_range=(ymin, ymax))



# Add circle glyphs to the plot

plot.circle(x='x', y='y', fill_alpha=0.8, source=source)



# Set the x-axis label

plot.xaxis.axis_label ='Fertility (children per woman)'



# Set the y-axis label

plot.yaxis.axis_label = 'Life Expectancy (years)'



# Add the plot to the current document and add a title

curdoc().add_root(plot)

curdoc().title = 'Gapminder'



# Make a list of the unique values from the region column: regions_list

regions_list = gapminder_df.region.unique().tolist()



# Import CategoricalColorMapper from bokeh.models and the Spectral6 palette from bokeh.palettes

from bokeh.models import CategoricalColorMapper

from bokeh.palettes import Spectral6



# Make a color mapper: color_mapper

color_mapper = CategoricalColorMapper(factors=regions_list,palette=Spectral6)



# Add the color mapper to the circle glyph

plot.circle(x='x', y='y', fill_alpha=0.8, source=source,

            color=dict(field='region', transform=color_mapper), legend_label='region')



# Set the legend.location attribute of the plot to 'top_right'

plot.legend.location = 'top_right'



# Add the plot to the current document and add the title

curdoc().add_root(plot)

curdoc().title = 'Gapminder'
# Import the necessary modules

from bokeh.layouts import Column,row

from bokeh.models import Slider



# Define the callback function: update_plot

def update_plot(attr,old,new):

    # Set the yr name to slider.value and new_data to source.data

    yr = slider.value

    new_data = {

        'x'       : gapminder_df[gapminder_df['Year']==yr].fertility,

        'y'       :  gapminder_df[gapminder_df['Year']==yr].life,

        'country' :  gapminder_df[gapminder_df['Year']==yr].Country,

        'pop'     : ( gapminder_df[gapminder_df['Year']==yr].population / 20000000) + 2,

        'region'  :  gapminder_df[gapminder_df['Year']==yr].region,

    }

    source.data = new_data

    plot.title.text = 'Gapminder data for %d' % yr





# Make a slider object: slider

slider = Slider(start=1970,end=2010,step=1,value=1970,title='Year')



# Attach the callback to the 'value' property of slider

slider.on_change('value',update_plot)



# Make a row layout of widgetbox(slider) and plot and add it to the current document

layout = row(Column(slider), plot)

curdoc().add_root(layout)