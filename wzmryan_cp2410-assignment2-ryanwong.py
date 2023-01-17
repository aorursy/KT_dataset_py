class Node:

    def __init__(self, item):           # Initializing the class

        self.idNum = item               # A node should contain a list item [id, x, y]

        self.adjacent = {}              # And a dictionary of adjacent points



        self.previous = None



    def add_adjacency(self, id_num, distance):  # Method for adding adjacency's

        self.adjacent[id_num] = distance



class Graph:                                    # Defining the class for Graphs to be handled

    def __init__(self):

        self.NodeDict = {}

        self.NumNodes = 0



    def add_node(self, id_num):                 # Method for adding nodes

        self.NumNodes += 1

        new_node = Node(id_num)

        self.NodeDict[id_num] = new_node

        return new_node



    def get_node(self, node_data):              # Function for getting nodes

        if node_data in self.NodeDict:

            return self.NodeDict[node_data]

        else:

            return None



    def get_node_adjacents(self, node):         # Function for getting node adjacents

        return self.NodeDict[node].adjacent



    def add_edge(self, start_id, end_id, distance):                # Function for adding edges

        self.NodeDict[start_id].add_adjacency(end_id, distance)    # Adds adjacency to start node

        self.NodeDict[end_id].add_adjacency(start_id, distance)    # Adds adjacency to end node

import math



def find_dist(pt1, pt2):                    # Function to find the distance between 2 points

    pt1 = int(pt1)                          # Calculation below assumes points are on flat surface

    pt2 = int(pt2)

    distance = math.sqrt(

        ((points_list[pt1][1] - points_list[pt2][1]) ** 2) + ((points_list[pt1][2] - points_list[pt2][2]) ** 2))

    return distance



def find_closest_pts(curr_point, search_range):             # Recursive function to find closest point

    curr_x_index = points_list_sortedX.index(curr_point)    # Defines the index of current point on sorted list

    possible_points_1 = []

    possible_points_2 = []

    possible_points_3 = []

    index_holder = []

    if len(points_list_sortedX) == 1:                               # Checks if this is the last point is list

        return False                                                # Returns False if yes

    if curr_x_index > 0:                                            # Checks if any points behind current point

        for element in points_list_sortedX[curr_x_index - 1::-1]:   # If yes, iterate backwards to find points

            if abs(curr_point[1] - element[1]) < search_range:

                possible_points_1.append(element)                   # Saves points if x coords in range

            else:

                break                                               # Ends loop if no more points in range

    for element in points_list_sortedX[curr_x_index + 1:]:          # Now iterate forwards to find points

        if abs(curr_point[1] - element[1]) < search_range:

            possible_points_1.append(element)                       # Saves points to list1 if x coords in range

        else:

            break                                                   # Ends loop if no more points in range

    if curr_point in possible_points_1:

        possible_points_1.remove(curr_point)

    if len(possible_points_1) > 1:                                  # While there are >1 possible points

        possible_points_1.sort(key=lambda x: x[2])                  # Sort points by Y coords

        for point2 in possible_points_1:

            if abs(curr_point[2] - point2[2]) < search_range:       # Check if points Y coords are also in range

                possible_points_2.append(point2)                    # Saves points to list2 if yes



        if len(possible_points_2) > 1:                              # If there are still >1 possible points

            for point3 in possible_points_2:                        # saves all point's distance

                possible_points_3.append([point3, find_dist(point3[0], curr_point[0])])

            possible_points_3.sort(key=lambda x: x[1])              # sort points by distance

            index_holder.append(possible_points_3[0])

            index_holder.append(possible_points_3[1])               # save only the closest 2 points

            return index_holder                                     # Returns list of lists, containing:

                                                                    # [[id1, distance], [id2, distance]]



    return find_closest_pts(curr_point, (search_range + 100) * 2)   # Re-curves with bigger range if no points



def find_path_length(path):                             # Function to calculate the total length of a given path

    length = 0                                          # Initiate counter

    for point in path:                                  # For each point

        if point == path[-1]:                           # If this is the last point

            length += find_dist(point[0], 0)            # Add distance from here back to home (return journey)

        else:                                                       # Else

            length += find_dist(point[0], path.index(point) + 1)    # add distance from here to next point

    return length

            
def BuildGraphNodes(Graph, RawData):        # Function to place these points on the graph as nodes

    for data in RawData:                    # RawData is expected to be a list of lists,

        Graph.add_node(data)                # so data is a list of [id num, x-coord, y-coord]

    print("Nodes built successfully")



def BuildGraphEdges(Graph, RawData):        # Function to build edges of the graph

    progress = 0                            # Counter to show progress as this will likely be slow

    prog10per = 0

    prog25per = 0

    prog50per = 0

    prog75per = 0

    prog90per = 0

    for data in Graph.NodeDict:

        closest_points = find_closest_pts(data, 100).copy()

        closest_point1 = closest_points[0][0]

        distance1 = closest_points[0][1]

        closest_point2 = closest_points[1][0]

        distance2 = closest_points[1][1]

        Graph.add_edge(data, closest_point1, distance1)

        Graph.add_edge(data, closest_point2, distance2)

        progress += 1

        completion = progress/len(Graph.NodeDict)

        if completion > 0.1 and prog10per == 0:

            prog10per = 1

            print("Edges 10% built")

        if completion > 0.25 and prog25per == 0:

            prog25per = 1

            print("Edges 25% built")

        if completion > 0.50 and prog50per == 0:

            prog50per = 1

            print("Edges 50% built")

        if completion > 0.75 and prog75per == 0:

            prog75per = 1

            print("Edges 75% built")

        if completion > 0.90 and prog90per == 0:

            prog90per = 1

            print("Edges 90% built")

    print("Edges built successfully")
import csv



def get_CSV(filename, points_list):

    with open(filename, 'r') as csvfile:      # reading the csv file

        csvreader = csv.reader(csvfile)         # Creating a reader

        col_names = next(csvreader)             # Extract column names

        for point_STR in csvreader:             # Add data points into internal list

            point_FLT = []                      # Convert the data points from String to Float

            for i in point_STR:

                if len(point_FLT) == 0:

                    point_FLT.append(int(i)+1)  # Saves id number as an integer+1, so that first data has id 1

                else:

                    point_FLT.append(float(i))  # Saves coordinates as float

            points_list.append(tuple(point_FLT))

            

def export_CSV(output_filename, output_data):

    with open(output_filename, 'w', newline='') as file:

        writer = csv.writer(file)

        writer.writerows(output_data)
Test_Graph1 = Graph()            # Initiate a graph

input_data = "../input/a2-data/first3.csv"       # Enter filename of input data here.

points_list = [(0, 0.0, 0.0)]   # Initiate the list of points, starting with start coordinates



get_CSV(input_data, points_list)                # Get the data from specified CSV

points_list_sortedX = points_list.copy()        # Create a duplicate list thats sorted by x coords

points_list_sortedX.sort(key=lambda x: x[1])



BuildGraphNodes(Test_Graph1, points_list)     # Build graphs nodes

BuildGraphEdges(Test_Graph1, points_list)     # Build graphs edges
print("node dictionary: ", Test_Graph1.NodeDict)
print("node 0's adjacency's: ", Test_Graph1.get_node_adjacents(points_list[0]))
def DFS(Graph, start):

    destinations = [start]                  # Use the keys of the dictionary here

    path = []                               # When nodes are visited, log their keys here

    while len(destinations) >0:

        current = destinations[-1]

        destinations.pop()

        if current in path:

            continue

        for adjacent in Graph.get_node_adjacents(current):

            destinations.append(adjacent)

        path.append(current)

    return path
print("path is: ", DFS(Test_Graph1, points_list[0]))
Test_Graph2 = Graph()           # Initiate a graph

input_data = "../input/a2-data/cities10per.csv"  # Enter filename of input data here.

points_list = [(0, 0.0, 0.0)]   # Initiate the list of points, starting with start coordinates



get_CSV(input_data, points_list)                # Get the data from specified CSV

points_list_sortedX = points_list.copy()        # Create a duplicate list thats sorted by x coords

points_list_sortedX.sort(key=lambda x: x[1])



BuildGraphNodes(Test_Graph2, points_list)     # Build graphs nodes

BuildGraphEdges(Test_Graph2, points_list)     # Build graphs edges
print("path is: ", DFS(Test_Graph2, points_list[0]))
def DFS_Repeat(Graph, start):

    path = DFS(Graph, start)                                # Apply DFS on starting island

    for point in path:                                      # Remove visited points from sorted list

        points_list_sortedX.remove(point)

    points_list_sortedX.append(path[-1])                    # Add the ending point of the DFS back in

    points_list_sortedX.sort(key=lambda x: x[1])

    hop_start = path[-1]

    prog10per = 0                            # Counter to show progress as this will likely be slow

    prog25per = 0

    prog50per = 0

    prog75per = 0

    prog90per = 0

    while len(path) != len(Graph.NodeDict):                 # While ther are still unvisited points

        destinations = find_closest_pts(hop_start, 100)     # Find closest unvisited point to hop to

        points_list_sortedX.remove(hop_start)

        new_visits = DFS(Graph, destinations[0][0])         # Run and save DFS on new island

        points_list_sortedX.append(new_visits[-1])

        points_list_sortedX.sort(key=lambda x: x[1])

        for point in new_visits:                             # Remove visited points from points_list

            points_list_sortedX.remove(point)

        hop_start = new_visits[-1]

        path += new_visits

        completion = len(path)/len(Graph.NodeDict)

        if completion > 0.1 and prog10per == 0:

            prog10per = 1

            print("Path 10% built")

        if completion > 0.25 and prog25per == 0:

            prog25per = 1

            print("Path 25% built")

        if completion > 0.50 and prog50per == 0:

            prog50per = 1

            print("Path 50% built")

        if completion > 0.75 and prog75per == 0:

            prog75per = 1

            print("Path 75% built")

        if completion > 0.90 and prog90per == 0:

            prog90per = 1

            print("Path 90% built")

    print("Path built successfully")

    return path
path = DFS_Repeat(Test_Graph2, points_list[0])

points_list_sortedX = points_list.copy()        # Fixes the list after algorithm is done

points_list_sortedX.sort(key=lambda x: x[1])

export_CSV("dfspath.csv", path)

print("Distance for repeated DFS: ", find_path_length(path))
Test_Graph3 = Graph()           # Initiate a graph

input_data = "../input/a2-data/cities.csv"       # Enter filename of input data here.

points_list = [(0, 0.0, 0.0)]   # Initiate the list of points, starting with start coordinates



get_CSV(input_data, points_list)                # Get the data from specified CSV

points_list_sortedX = points_list.copy()        # Create a duplicate list thats sorted by x coords

points_list_sortedX.sort(key=lambda x: x[1])



BuildGraphNodes(Test_Graph3, points_list)     # Build graphs nodes

BuildGraphEdges(Test_Graph3, points_list)     # Build graphs edges
path = DFS_Repeat(Test_Graph3, points_list[0])

points_list_sortedX = points_list.copy()        # Fixes the list after algorithm is done

points_list_sortedX.sort(key=lambda x: x[1])

export_CSV("dfsrptpath.csv", path)

print("Distance for repeated DFS: ", find_path_length(path))
from queue import Queue



def BFS(Graph, start):

    destinations = Queue(0)                 # Use the keys of the dictionary here

    destinations.put(start)

    path = []                               # When nodes are visited, log their keys here

    while destinations.empty() == False:

        current = destinations.get()

        if current in path:

            continue

        for adjacent in Graph.get_node_adjacents(current):

            destinations.put(adjacent)

        path.append(current)

    return path



def BFS_Repeat(Graph, start):

    path = BFS(Graph, start)                                # Apply BFS on starting island

    for point in path:                                      # Remove visited points from sorted list

        points_list_sortedX.remove(point)

    points_list_sortedX.append(path[-1])                    # Add the ending point of the BFS back in

    points_list_sortedX.sort(key=lambda x: x[1])

    hop_start = path[-1]

    prog10per = 0                            # Counter to show progress as this will likely be slow

    prog25per = 0

    prog50per = 0

    prog75per = 0

    prog90per = 0

    while len(path) != len(Graph.NodeDict):                 # While ther are still unvisited points

        destinations = find_closest_pts(hop_start, 100)     # Find closest unvisited point to hop to

        points_list_sortedX.remove(hop_start)

        new_visits = BFS(Graph, destinations[0][0])         # Run and save BFS on new island

        points_list_sortedX.append(new_visits[-1])

        points_list_sortedX.sort(key=lambda x: x[1])

        for point in new_visits:                            # Remove visited points from points_list

            points_list_sortedX.remove(point)

        hop_start = new_visits[-1]

        path += new_visits

        completion = len(path)/len(Graph.NodeDict)

        if completion > 0.1 and prog10per == 0:

            prog10per = 1

            print("Path 10% built")

        if completion > 0.25 and prog25per == 0:

            prog25per = 1

            print("Path 25% built")

        if completion > 0.50 and prog50per == 0:

            prog50per = 1

            print("Path 50% built")

        if completion > 0.75 and prog75per == 0:

            prog75per = 1

            print("Path 75% built")

        if completion > 0.90 and prog90per == 0:

            prog90per = 1

            print("Path 90% built")

    print("Path built successfully")

    return path
path2 = BFS_Repeat(Test_Graph3, points_list[0])

points_list_sortedX = points_list.copy()        # Fixes the list after algorithm is done

points_list_sortedX.sort(key=lambda x: x[1])

export_CSV("bfsrptpath.csv", path2)

print("Distance for repeated BFS: ", find_path_length(path2))