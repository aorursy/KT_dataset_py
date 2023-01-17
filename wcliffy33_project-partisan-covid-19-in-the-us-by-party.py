import pandas as pd



import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots



from us_state_abbrev_dictionary import us_state_abbrev, abbrev_us_state, regions, r_names

from new_cases_utils import new_cases, new_cases1, new_cases_per_capita
covid_data = pd.read_csv('../input/us-covid19-dataset-live-hourlydaily-updates/US-counties_historical.csv')



def st_county_helper(row):    # This will come in handy when trying to query by county

    return us_state_abbrev[row.state] + ', ' + row.county



covid_data['st_county'] = covid_data.apply(st_county_helper, axis='columns')
ny_cases, us_cases = new_cases(covid_data.query('state == "New York"'), covid_data)



f1 = make_subplots(rows=1, cols=2, subplot_titles=("New York", "United States"))

f1.update_layout(width=800, height=400, showlegend=False, bargap=0, title_text="New COVID-19 Cases")

f1.add_trace(go.Bar(x=ny_cases.index, y=ny_cases.values), 1, 1)

f1.add_trace(go.Bar(x=us_cases.index, y=us_cases.values), 1, 2)



f1.show()
f2 = make_subplots(rows=2, cols=2, subplot_titles=r_names, shared_yaxes=True)

f2.update_layout(width=800, height=700, showlegend=False, bargap=0, yaxis=dict(range=[0,45000]), title_text="New COVID-19 Cases by Region")



for i in range(len(regions)):

    rc = new_cases1(covid_data.query('state in @regions[@i]'), 'cases')

    f2.add_trace(go.Bar(x=rc.index, y=rc.values), ((i//2)%2)+1, (i%2)+1)   # little indexing trick I learned in Algo



f2.show()
pres_data = pd.read_csv('../input/2016uspresidentialvotebycounty/pres16results.csv')

census_data = pd.read_csv('../input/us-county-population-estimates-20102019/co-est2019-annres.csv').set_index('County').transpose()
grouped = pres_data.groupby(['st', 'county', 'cand']).sum()

dt_counties = grouped.apply(lambda x: [x.name[0], x.name[1], x.name[2]] if x.pct >= 0.5 else None, axis='columns')



djt_win = [] 

hrc_win = []



for row in dt_counties:

    if row is None:

        continue

    if row[2] == 'Donald Trump':

        djt_win.append(row[0] + ', ' + row[1].replace(' County', ''))

    elif row[2] == 'Hillary Clinton':

        hrc_win.append(row[0] + ', ' + row[1].replace(' County', ''))

    else:

        print(row[2])    # No other candidates received the majority of a county in the general election
h = covid_data.query('st_county in @hrc_win')

d = covid_data.query('st_county in @djt_win')



hrc_t,  djt_t  = new_cases(h, d)

hrc_pc, djt_pc = new_cases_per_capita(h, d, census_data)

print(hrc_pc)
print(covid_data.county.unique())
f3 = make_subplots(rows=2, cols=2, subplot_titles=("Hillary Clinton - New Cases", "Donald Trump - New Cases", "Hillary Clinton - New Cases Per Capita","Donald Trump - New Cases Per Capita"), shared_yaxes=True)

f3.update_layout(width=800, height=800, showlegend=False, xaxis_showgrid=False, yaxis_showgrid=False, bargap=0, title_text="New Cases by 2016 Presidential Election Results (by County)")



f3.add_trace(go.Bar(x=hrc_t.index,  y=hrc_t.values), 1, 1)

f3.add_trace(go.Bar(x=djt_t.index,  y=djt_t.values), 1, 2)

f3.add_trace(go.Bar(x=hrc_pc.index, y=hrc_pc.values), 2, 1)

f3.add_trace(go.Bar(x=djt_pc.index, y=djt_pc.values), 2, 2)



f3.show()
gov2016 = pd.read_csv('../input/us-state-office-election-returns-2016/stateoffices2016.csv',encoding="ISO-8859-1").query('office == "Governor" or office == "Governor and Lt. Governor"')

gov2017 = None

gov2018 = pd.read_csv('../input/us-state-office-election-returns-2018/county_2018.csv',encoding="ISO-8859-1").query('office == "Governor" or office == "Governor and Lt. Governor"')

gov2019 = None
cand_mov1 = gov2016.groupby(['state', 'party']).apply(lambda r: r.candidatevotes.sum() / r.totalvotes).drop_duplicates()

cand_mov2 = gov2018.groupby(['state', 'party']).apply(lambda r: r.candidatevotes / r.totalvotes).drop_duplicates()



cand_mov = pd.concat([cand_mov1, cand_mov2], sort=False)



st_races = {}

for st in [*gov2016.state.unique(), *gov2018.state.unique()]:

    st_races[st] = {}

    

for r in cand_mov.index:

    st_races[r[0]][r[1]] = cand_mov[r]
rep_states = []

dem_states = []



i = 1

for k in st_races.keys():

    t1 = max(st_races[k], key=st_races[k].get)

    if t1 == 'republican':

        rep_states.append(k)

    elif t1 == 'democrat':

        dem_states.append(k)

    else:

        print('Non-bipartisan party majority:', k, st_races[k])   # Issue with NY, we'll have to fix later
rep, dem = new_cases(covid_data.query('state in @rep_states'), covid_data.query('state in @dem_states'))



f4 = make_subplots(rows=2, cols=2, subplot_titles=("Democratic Governors", "Republican Governors", "PLACEHOLDER", "PLACEHOLDER"), shared_yaxes=True)



f4.add_trace(go.Bar(x=dem.index, y=dem.values), 1, 1)

f4.add_trace(go.Bar(x=rep.index, y=rep.values), 1, 2)

f4.add_trace(go.Bar(x=dem.index, y=dem.values), 2, 1)

f4.add_trace(go.Bar(x=rep.index, y=rep.values), 2, 2)



f4.update_layout(width=800, height=800, 

                 showlegend=False, bargap=0)



f4.update_yaxes(title_text="New Cases - Total", row=1, col=1)

f4.update_yaxes(title_text="New Cases - Per Capita", row=2, col=1)



print("This graph is misleading until I get all states in - I believe I'm missing mostly democratic states, which would make their cases seem significantly lower then it should be.")

# f4.show()
rep2, dem2 = new_cases(covid_data.query('state in @rep_states'), covid_data.query('state in @dem_states'))



f5 = make_subplots(rows=2, cols=2, subplot_titles=("Democratic Candidates", "Republican Candidates", "PLACEHOLDER", "PLACEHOLDER"), shared_yaxes=True)



f5.add_trace(go.Bar(x=dem2.index, y=dem2.values), 1, 1)

f5.add_trace(go.Bar(x=rep2.index, y=rep2.values), 1, 2)

f5.add_trace(go.Bar(x=dem2.index, y=dem2.values), 2, 1)

f5.add_trace(go.Bar(x=rep2.index, y=rep2.values), 2, 2)



f5.update_layout(width=800, height=800, 

                 showlegend=False, bargap=0)



f5.update_yaxes(title_text="New Cases - Total", row=1, col=1)

f5.update_yaxes(title_text="New Cases - Per Capita", row=2, col=1)



print()

# f5.show()