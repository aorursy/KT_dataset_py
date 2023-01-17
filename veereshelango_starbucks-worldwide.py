# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import plotly

from plotly.graph_objs import *

import plotly.plotly as py

plotly.offline.init_notebook_mode(connected=True)



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/directory.csv")

df.head()
df.shape
to_plot = df['Country'].value_counts().to_frame().reset_index().rename(columns={"index":"Country","Country":"Count"})

to_plot.head()
country_code = [('Afghanistan','AF','AFG'),('Aland Islands','AX','ALA'),('Albania','AL','ALB'),('Algeria','DZ','DZA'),('American Samoa','AS','ASM'),('Andorra','AD','AND'),('Angola','AO','AGO'),('Anguilla','AI','AIA'),('Antarctica','AQ','ATA'),('Antigua and Barbuda','AG','ATG'),('Argentina','AR','ARG'),('Armenia','AM','ARM'),('Aruba','AW','ABW'),('Australia','AU','AUS'),('Austria','AT','AUT'),('Azerbaijan','AZ','AZE'),('Bahamas','BS','BHS'),('Bahrain','BH','BHR'),('Bangladesh','BD','BGD'),('Barbados','BB','BRB'),('Belarus','BY','BLR'),('Belgium','BE','BEL'),('Belize','BZ','BLZ'),('Benin','BJ','BEN'),('Bermuda','BM','BMU'),('Bhutan','BT','BTN'),('Bolivia','BO','BOL'),('Bosnia and Herzegovina','BA','BIH'),('Botswana','BW','BWA'),('Bouvet Island','BV','BVT'),('Brazil','BR','BRA'),('British Virgin Islands','VG','VGB'),('British Indian Ocean Territory','IO','IOT'),('Brunei Darussalam','BN','BRN'),('Bulgaria','BG','BGR'),('Burkina Faso','BF','BFA'),('Burundi','BI','BDI'),('Cambodia','KH','KHM'),('Cameroon','CM','CMR'),('Canada','CA','CAN'),('Cape Verde','CV','CPV'),('Cayman Islands','KY','CYM'),('Central African Republic','CF','CAF'),('Chad','TD','TCD'),('Chile','CL','CHL'),('China','CN','CHN'),('Hong Kong, Special Administrative Region of China','HK','HKG'),('Macao, Special Administrative Region of China','MO','MAC'),('Christmas Island','CX','CXR'),('Cocos (Keeling) Islands','CC','CCK'),('Colombia','CO','COL'),('Comoros','KM','COM'),('Congo (Brazzaville)','CG','COG'),('Congo, Democratic Republic of the','CD','COD'),('Cook Islands','CK','COK'),('Costa Rica','CR','CRI'),('Côte dIvoire','CI','CIV'),('Croatia','HR','HRV'),('Cuba','CU','CUB'),('Cyprus','CY','CYP'),('Czech Republic','CZ','CZE'),('Denmark','DK','DNK'),('Djibouti','DJ','DJI'),('Dominica','DM','DMA'),('Dominican Republic','DO','DOM'),('Ecuador','EC','ECU'),('Egypt','EG','EGY'),('El Salvador','SV','SLV'),('Equatorial Guinea','GQ','GNQ'),('Eritrea','ER','ERI'),('Estonia','EE','EST'),('Ethiopia','ET','ETH'),('Falkland Islands (Malvinas)','FK','FLK'),('Faroe Islands','FO','FRO'),('Fiji','FJ','FJI'),('Finland','FI','FIN'),('France','FR','FRA'),('French Guiana','GF','GUF'),('French Polynesia','PF','PYF'),('French Southern Territories','TF','ATF'),('Gabon','GA','GAB'),('Gambia','GM','GMB'),('Georgia','GE','GEO'),('Germany','DE','DEU'),('Ghana','GH','GHA'),('Gibraltar','GI','GIB'),('Greece','GR','GRC'),('Greenland','GL','GRL'),('Grenada','GD','GRD'),('Guadeloupe','GP','GLP'),('Guam','GU','GUM'),('Guatemala','GT','GTM'),('Guernsey','GG','GGY'),('Guinea','GN','GIN'),('Guinea-Bissau','GW','GNB'),('Guyana','GY','GUY'),('Haiti','HT','HTI'),('Heard Island and Mcdonald Islands','HM','HMD'),('Holy See (Vatican City State)','VA','VAT'),('Honduras','HN','HND'),('Hungary','HU','HUN'),('Iceland','IS','ISL'),('India','IN','IND'),('Indonesia','ID','IDN'),('Iran, Islamic Republic of','IR','IRN'),('Iraq','IQ','IRQ'),('Ireland','IE','IRL'),('Isle of Man','IM','IMN'),('Israel','IL','ISR'),('Italy','IT','ITA'),('Jamaica','JM','JAM'),('Japan','JP','JPN'),('Jersey','JE','JEY'),('Jordan','JO','JOR'),('Kazakhstan','KZ','KAZ'),('Kenya','KE','KEN'),('Kiribati','KI','KIR'),("Korea, Democratic People's Republic of",'KP','PRK'),('Korea, Republic of','KR','KOR'),('Kuwait','KW','KWT'),('Kyrgyzstan','KG','KGZ'),('Lao PDR','LA','LAO'),('Latvia','LV','LVA'),('Lebanon','LB','LBN'),('Lesotho','LS','LSO'),('Liberia','LR','LBR'),('Libya','LY','LBY'),('Liechtenstein','LI','LIE'),('Lithuania','LT','LTU'),('Luxembourg','LU','LUX'),

                ('Macedonia, Republic of','MK','MKD'),('Madagascar','MG','MDG'),('Malawi','MW','MWI'),('Malaysia','MY','MYS'),('Maldives','MV','MDV'),('Mali','ML','MLI'),('Malta','MT','MLT'),('Marshall Islands','MH','MHL'),('Martinique','MQ','MTQ'),('Mauritania','MR','MRT'),('Mauritius','MU','MUS'),('Mayotte','YT','MYT'),('Mexico','MX','MEX'),('Micronesia, Federated States of','FM','FSM'),('Moldova','MD','MDA'),('Monaco','MC','MCO'),('Mongolia','MN','MNG'),('Montenegro','ME','MNE'),('Montserrat','MS','MSR'),('Morocco','MA','MAR'),('Mozambique','MZ','MOZ'),('Myanmar','MM','MMR'),('Namibia','NA','NAM'),('Nauru','NR','NRU'),('Nepal','NP','NPL'),('Netherlands','NL','NLD'),('Netherlands Antilles','AN','ANT'),('New Caledonia','NC','NCL'),('New Zealand','NZ','NZL'),('Nicaragua','NI','NIC'),('Niger','NE','NER'),('Nigeria','NG','NGA'),('Niue','NU','NIU'),('Norfolk Island','NF','NFK'),('Northern Mariana Islands','MP','MNP'),('Norway','NO','NOR'),('Oman','OM','OMN'),('Pakistan','PK','PAK'),('Palau','PW','PLW'),('Palestinian Territory, Occupied','PS','PSE'),('Panama','PA','PAN'),('Papua New Guinea','PG','PNG'),('Paraguay','PY','PRY'),('Peru','PE','PER'),('Philippines','PH','PHL'),('Pitcairn','PN','PCN'),('Poland','PL','POL'),('Portugal','PT','PRT'),('Puerto Rico','PR','PRI'),('Qatar','QA','QAT'),('Réunion','RE','REU'),('Romania','RO','ROU'),('Russian Federation','RU','RUS'),('Rwanda','RW','RWA'),('Saint-Barthélemy','BL','BLM'),('Saint Helena','SH','SHN'),('Saint Kitts and Nevis','KN','KNA'),('Saint Lucia','LC','LCA'),('Saint-Martin (French part)','MF','MAF'),('Saint Pierre and Miquelon','PM','SPM'),('Saint Vincent and Grenadines','VC','VCT'),('Samoa','WS','WSM'),('San Marino','SM','SMR'),('Sao Tome and Principe','ST','STP'),('Saudi Arabia','SA','SAU'),('Senegal','SN','SEN'),('Serbia','RS','SRB'),('Seychelles','SC','SYC'),('Sierra Leone','SL','SLE'),('Singapore','SG','SGP'),('Slovakia','SK','SVK'),('Slovenia','SI','SVN'),('Solomon Islands','SB','SLB'),('Somalia','SO','SOM'),('South Africa','ZA','ZAF'),('South Georgia and the South Sandwich Islands','GS','SGS'),('South Sudan','SS','SSD'),('Spain','ES','ESP'),('Sri Lanka','LK','LKA'),('Sudan','SD','SDN'),('Suriname *','SR','SUR'),('Svalbard and Jan Mayen Islands','SJ','SJM'),('Swaziland','SZ','SWZ'),('Sweden','SE','SWE'),('Switzerland','CH','CHE'),('Syrian Arab Republic (Syria)','SY','SYR'),('Taiwan, Republic of China','TW','TWN'),('Tajikistan','TJ','TJK'),('Tanzania *, United Republic of','TZ','TZA'),('Thailand','TH','THA'),('Timor-Leste','TL','TLS'),('Togo','TG','TGO'),('Tokelau','TK','TKL'),('Tonga','TO','TON'),('Trinidad and Tobago','TT','TTO'),('Tunisia','TN','TUN'),('Turkey','TR','TUR'),('Turkmenistan','TM','TKM'),('Turks and Caicos Islands','TC','TCA'),('Tuvalu','TV','TUV'),('Uganda','UG','UGA'),('Ukraine','UA','UKR'),('United Arab Emirates','AE','ARE'),('United Kingdom','GB','GBR'),('United States of America','US','USA'),('United States Minor Outlying Islands','UM','UMI'),('Uruguay','UY','URY'),('Uzbekistan','UZ','UZB'),('Vanuatu','VU','VUT'),('Venezuela (Bolivarian Republic of)','VE','VEN'),('Viet Nam','VN','VNM'),('Virgin Islands, US','VI','VIR'),('Wallis and Futuna Islands','WF','WLF'),('Western Sahara','EH','ESH'),

                ('Yemen','YE','YEM'),('Zambia','ZM','ZMB'),('Zimbabwe','ZW','ZWE')]

labels = ["CountryName","Country","CountryCode"]

cc_df = pd.DataFrame.from_records(country_code,columns=labels)
plots = pd.merge(to_plot,cc_df,on="Country",how="left")

plots.head()
data = [ dict(

        type='choropleth',

        autocolorscale = False,

        locations = plots["CountryCode"],

        z = plots["Count"],

        text = plots["CountryName"],

        marker = dict(

            line = dict (

                color = 'rgb(255,255,255)',

                width = 2

            ) ),

        colorbar = dict(

            title = "Count")

        ) ]



layout = dict(

        title= 'Number of Starbucks in Countries',

        geo = dict(

            showframe = False,            

            scope='world',

            projection=dict( type='Mercator' ),

            showlakes = True,

            lakecolor = 'rgb(255, 255, 255)'),

             )

    

fig = dict( data=data, layout=layout )

plotly.offline.iplot( fig,validate=False, filename='d3-cloropleth-map' )