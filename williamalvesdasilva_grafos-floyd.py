import matplotlib.pyplot as plt

from IPython.display import clear_output

from ipywidgets import interact, widgets

import networkx as nx

import numpy as np
button_add = widgets.Button(description="Add Node")

button_update = widgets.Button(description="Update Node") 

range_edge = widgets.IntSlider(description="Edge value between nodes", continuous_update=True)

button_distance = widgets.Button(description="Distance") 

 

output_graph = widgets.Output()

output_buttons = widgets.Output()

output_dropdowns = widgets.Output()

output_slider = widgets.Output()

output_matrix = widgets.Output()

output_matrix_m = widgets.Output()

output_matrix_r = widgets.Output()

output_distance = widgets.Output()

output_caldist = widgets.Output()

 

graph = nx.MultiDiGraph()

nodes = []

idx = 1

 

matrix = []

matrix_m = []

matrix_r = []

xvalue = 1

zvalue = 1

wvalue = 0

 

def fw(v_start, v_end):

  global matrix_r, matrix_m

 

  v_start -= 1  

  v_end -= 1

  v_orig = v_start

  plen = 0  

 

  while (v_end != v_start):        

    print('Caminho percorrido: ', end= ' ')

    print(v_start + 1, ' -> ', end='')

    v_start = int(matrix_r[v_start][v_end]) - 1              

    print('Distância percorrida: ' , end=' ')

    plen += matrix_m[v_start][v_end]  

  

  print(v_start + 1)

  print(matrix_m[v_orig][v_end])

 

 

def floyd(G, b):

  global idx  

  distance = list(map(lambda i: list(map(lambda j: j, i)), G))    

  travel = np.zeros((idx - 1, idx - 1))

 

  for i in range(idx - 1):

    travel[:, i] = i + 1

 

  for k in range(idx-1):

    for i in range(idx-1):

      for j in range(idx-1):  

        if distance[i][j] > (distance[i][k] + distance[k][j]):                    

          distance[i][j] = distance[i][k] + distance[k][j]

          travel[i][j] = k + 1

  return distance if b else travel

 

def on_changew(change):  

  global graph, wvalue

  wvalue = change['new']  

 

def on_changex(change):  

  global graph, xvalue

  xvalue = change['new']  

 

def on_changez(change):  

  global graph, zvalue

  zvalue = change['new']  

 

def distance_nodes(x):

  global graph, xvalue, zvalue, wvalue, matrix, matrix_m, matrix_r

 

  with output_caldist:

    clear_output()  

    if len(nodes) > 1:

      display(button_distance)

 

    if xvalue != zvalue:

      fw(xvalue, zvalue)

 

def update_node(x):

  global graph, xvalue, zvalue, wvalue, matrix, matrix_m, matrix_r

 

  with output_graph:

    if xvalue != zvalue:

      graph.add_edge(xvalue, zvalue, weight=wvalue)

    clear_output()             

    plt.show(nx.draw(graph))        

 

  with output_matrix:

    clear_output()

    if xvalue != zvalue:      

      matrix[xvalue-1][zvalue-1] = range_edge.value #wvalue

 

    fig, ax = plt.subplots()    

    ax.matshow(matrix, alpha=0)

    ax.set_xticklabels([' '] + nodes)

    ax.set_yticklabels([' '] + nodes)

 

    for (i, j), z in np.ndenumerate(matrix):

        ax.text(j, i, '{:0.0f}'.format(z), ha='center', va='center')        

    plt.show()

    

  with output_matrix_r:

    clear_output()

    matrix_r = floyd(matrix, False)

    fig_r, ax_r = plt.subplots()    

    ax_r.matshow(matrix_r, alpha=0)

    ax_r.set_xticklabels([' '] + nodes)

    ax_r.set_yticklabels([' '] + nodes)

 

    for (i, j), z in np.ndenumerate(matrix_r):

        ax_r.text(j, i, '{:0.0f}'.format(z), ha='center', va='center')        

    plt.show()

    

  with output_matrix_m:

    clear_output()

    matrix_m = floyd(matrix, True)

    fig_m, ax_m = plt.subplots()    

    ax_m.matshow(matrix_m, alpha=0)

    ax_m.set_xticklabels([' '] + nodes)

    ax_m.set_yticklabels([' '] + nodes)

 

    for (i, j), z in np.ndenumerate(matrix_m):

        ax_m.text(j, i, '{:0.0f}'.format(z), ha='center', va='center')        

    plt.show()

 

 

def add_node(x):

  global vertices, graph, idx, xvalue, zvalue

 

  graph.add_node(idx)         

  nodes.append(idx)

  idx += 1

 

  with output_buttons:

    clear_output()  

    if len(nodes) > 1:

      display(button_update)

 

  with output_dropdowns:    

    clear_output()     

    node_in = widgets.Dropdown(options=nodes, description="Node:", continuous_update=True)

    node_out = widgets.Dropdown(options=nodes, description="->:", continuous_update=True)

 

    display(node_in)

    display(node_out)

 

    node_in.observe(on_changex, 'value')

    node_out.observe(on_changez, 'value')

 

  with output_slider:

    clear_output()       

    display(range_edge)

    range_edge.observe(on_changew, 'value')

     

 

  with output_graph:

    clear_output()

    plt.show(nx.draw(graph))

 

  with output_matrix:    

    global matrix

    clear_output()

 

    matrix = np.zeros((idx - 1, idx - 1))

    matrix[:][:] = np.inf

    for i in range(idx-1):

      matrix[i][i] = 0

    

    fig, ax = plt.subplots()    

    ax.matshow(matrix, alpha=0)

    ax.set_xticklabels([' '] + nodes)

    ax.set_yticklabels([' '] + nodes)

 

    for (i, j), z in np.ndenumerate(matrix):

        ax.text(j, i, '{:0.0f}'.format(z), ha='center', va='center')

        

    plt.show()

    

  with output_distance:    

    clear_output()     

    node_in = widgets.Dropdown(options=nodes, description="Node:", continuous_update=True)

    node_out = widgets.Dropdown(options=nodes, description="->:", continuous_update=True)

 

    display(node_in)

    display(node_out)

 

    node_in.observe(on_changex, 'value')

    node_out.observe(on_changez, 'value')

 

  with output_caldist:

    clear_output()  

    if len(nodes) > 1:

      display(button_distance)

 

 

display(button_add)

display(output_dropdowns)

display(output_slider)

display(output_buttons)

 

button_add.on_click(add_node)

button_update.on_click(update_node)

button_distance.on_click(distance_nodes)
display(output_matrix)
display(output_matrix_m)
display(output_matrix_r)
display(output_distance)

display(output_caldist)
display(output_graph)