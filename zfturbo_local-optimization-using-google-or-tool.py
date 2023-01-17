import numpy as np
import pandas as pd
from sympy import sieve
import random
%matplotlib inline
original = pd.read_csv("../input/close-ends-chunks-optimization-aka-2-opt/1515651.377838111.csv")
cities = pd.read_csv("../input/traveling-santa-2018-prime-paths/cities.csv")
cities.rename(columns={"CityId":"Path"}, inplace=True)
baseline = original.merge(cities,how='left',on='Path')
pnums = list(sieve.primerange(0, baseline.shape[0]))
def score_it(df):
    df['step'] = np.sqrt((df.X - df.X.shift())**2 + (df.Y - df.Y.shift())**2)
    df['step_adj'] = np.where((df.index) % 10 != 0, df.step, df.step + 
                              df.step*0.1*(~df.Path.shift().isin(pnums)))
    return df.step_adj.sum()

display(score_it(baseline))
#%% imports
from scipy.spatial.distance import pdist, squareform
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

#%% functions
def create_mat(df):
#    print("building matrix")
    mat = pdist(df)
    return squareform(mat)

def create_distance_callback(dist_matrix):
    def distance_callback(from_node, to_node):
      return int(dist_matrix[from_node][to_node])
    return distance_callback

status_dict = {0: 'ROUTING_NOT_SOLVED', 
               1: 'ROUTING_SUCCESS', 
               2: 'ROUTING_FAIL',
               3: 'ROUTING_FAIL_TIMEOUT',
               4: 'ROUTING_INVALID'}

def optimize(df, startnode, stopnode, fixed):     
    num_nodes = df.shape[0]
    mat = create_mat(df)
    dist_callback = create_distance_callback(mat)
    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
    search_parameters.solution_limit = num_iters 
    search_parameters.first_solution_strategy = (
                                    routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION)
    search_parameters.local_search_metaheuristic = (
                            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)

    if fixed:
        routemodel = pywrapcp.RoutingModel(num_nodes, 1, [startnode], [stopnode])
    else:
        routemodel = pywrapcp.RoutingModel(num_nodes, 1, startnode)
    routemodel.SetArcCostEvaluatorOfAllVehicles(dist_callback)
    
    assignment = routemodel.SolveWithParameters(search_parameters)
    return routemodel, assignment
    
def get_route(df, startnode, stopnode, fixed): 
    routemodel, assignment = optimize(df, int(startnode), int(stopnode), fixed)
    route_number = 0
    node = routemodel.Start(route_number)
    route = []
    while not routemodel.IsEnd(node):
        route.append(node) 
        node = assignment.Value(routemodel.NextVar(node))
    return route
import random
pnums = list(sieve.primerange(0, baseline.shape[0]))
# df = dataframe
# m = range of cities to optimize
# n = number of optimizations to run

def run_opt(df,m,n):
    i = 0
    while i < n:
        startpoint = random.randint(0,df.shape[0])
        endpoint = min((startpoint + m),df.shape[0])
        
        district = df.iloc[startpoint:endpoint,:3].copy()
        district = district.reset_index()
        locations = district[['X', 'Y']].values
        
        segnodes = get_route(locations, 0, (m-1), fixed=True)
        ord_district = district.iloc[segnodes]
        segment = ord_district.index.tolist()
        
        temp = district.loc[segment, ['Path','X', 'Y']].reset_index()
        district_2 = district.copy()
        district_2.iloc[:(m-1),1:] = temp.copy()
        district = district.set_index('index')
        district_2 = district_2.set_index('index')
        
        district['step'] = np.sqrt((district.X - district.X.shift())**2 + (district.Y - district.Y.shift())**2)
        district['step_adj'] = np.where((district.index) % 10 != 0, district.step, district.step + 
                                        district.step*0.1*(~district.Path.shift().isin(pnums)))
        district_2['step'] = np.sqrt((district_2.X - district_2.X.shift())**2 + (district_2.Y - district_2.Y.shift())**2)
        district_2['step_adj'] = np.where((district_2.index) % 10 != 0, district_2.step, district_2.step + 
                                          district_2.step*0.1*(~district_2.Path.shift().isin(pnums)))
        
        check_dist = district.step_adj.sum() > district_2.step_adj.sum()
        print(i)
        print(district.step_adj.sum(), district_2.step_adj.sum())
        
        if check_dist:
            df.iloc[startpoint:endpoint,0:3] = district_2
        i += 1
%%time
num_iters = 250
run_opt(baseline, 50, num_iters)
def score_it(df):
    df['step'] = np.sqrt((df.X - df.X.shift())**2 + (df.Y - df.Y.shift())**2)
    df['step_adj'] = np.where((df.index) % 10 != 0, df.step, df.step + 
                              df.step*0.1*(~df.Path.shift().isin(pnums)))
    return df.step_adj.sum()

score = score_it(baseline)
display(score)
sub = pd.read_csv("../input/traveling-santa-2018-prime-paths/sample_submission.csv")
sub['Path'] = baseline['Path']
sub.to_csv('submission_{}.csv'.format(score), index=False)
sub.head()