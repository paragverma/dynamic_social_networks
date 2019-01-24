import networkx as nx
import os
import community
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

fname = "step_1"
fname2 = "step_2"
fname3 = "step_3"
graph = nx.read_gpickle(fname)
graph2 = nx.read_gpickle(fname2)
graph3 = nx.read_gpickle(fname3)


communities = community.best_partition(graph)
communities2 = community.best_partition(graph2)
communities3 = community.best_partition(graph3)

cl_dict = {}
cl_list = []
for node in communities:
  cnum = communities[node]
  if cnum not in cl_dict:
    cl_dict[cnum] = (random.random(), random.random(), random.random())

for node in graph.nodes:
  cl_list.append(cl_dict[communities[node]])


"""cl_list2 = []
cl_dict2 = {}
for node in communities2:
  cnum = communities2[node]
  if node in communities:
    cnumold = communities[node]
    
  if cnum not in cl_dict2:
    if cnum not in cl_dict:
      cl_dict2[cnum] = (random.random(), random.random(), random.random())
    
    cl_dict2[cnum] = cl_dict[communities[node]]"""

cl_dict2 = {}
for node in graph2.nodes:
  current_cnum = communities2[node]
  if node in graph.nodes:
    if current_cnum not in cl_dict2:
      cl_dict2[current_cnum] = cl_dict[communities[node]]
  elif current_cnum not in cl_dict2:
    cl_dict2[current_cnum] = (random.random(), random.random(), random.random())
    
cl_list2 = []
for node in graph2.nodes:
  cl_list2.append(cl_dict2[communities2[node]])

cl_dict3 = {}
for node in graph3.nodes:
  current_cnum = communities3[node]
  if node in graph2.nodes:
    if current_cnum not in cl_dict3:
      cl_dict3[current_cnum] = cl_dict2[communities2[node]]
  elif node in graph.nodes:
    if current_cnum not in cl_dict3:
      cl_dict3[current_cnum] = cl_dict[communities[node]]
  elif current_cnum not in cl_dict3:
    cl_dict3[current_cnum] = (random.random(), random.random(), random.random())
    
cl_list3 = []
for node in graph3.nodes:
  cl_list3.append(cl_dict3[communities3[node]])

"""for i in range(1, 201):
  if len(graph[i]) == 0:
    graph.remove_node(i)"""

pos = nx.drawing.nx_pydot.graphviz_layout(graph, prog='dot')
nx.draw_networkx(graph, pos=pos, node_size=2000, with_labels=False, node_color=cl_list, width=[graph[u][v]['weight'] / 2 for u,v in graph.edges])

plt.figure()
pos = nx.drawing.nx_pydot.graphviz_layout(graph2, prog='dot')
nx.draw_networkx(graph2, pos=pos, node_size=2000, with_labels=False, node_color=cl_list2, width=[graph2[u][v]['weight'] / 2 for u,v in graph2.edges])

plt.figure()
pos = nx.drawing.nx_pydot.graphviz_layout(graph3, prog='dot')
nx.draw_networkx(graph3, pos=pos, node_size=2000, with_labels=False, node_color=cl_list3, width=[graph3[u][v]['weight'] / 5 for u,v in graph3.edges])

plt.show()

plt.savefig("Graph.png", format="PNG")
