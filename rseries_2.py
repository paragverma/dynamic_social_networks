import networkx as nx
from datetime import datetime, timedelta
import community
import math
import matplotlib.pyplot as plt
import random
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import entropy
from argparse import ArgumentParser
from recursive_binning import RecursiveOptimalBin
from scipy.stats import binned_statistic
#wi and ki are weights

w = 10
e = 0.0692
step_size = 100 #days
#Events in wall posts file are sorted by timestamp
#Links are sorted by user Id
#Links may have \N

"""
Loop in a timestep
timestep can be adjusted

1st wall post - (14, 9, 2004)
Last wall post - (21, 1, 2009)

905565 link establishments time unknown. Can't use

Weight entries will always be from smaller to larger node number
"""

      

def getCurrentGraph(link_weight_entries, currentDate, maxuser=0, userlist=[]):
  graph = nx.Graph()
  
  #if maxuser > 0:
    #for i in range(1, maxuser + 1):
      #graph.add_node(i)
  #for u in userlist:
    #graph.add_node(u)
  
  max_weight = -float('inf')
  min_weight = float('inf')
  for node in link_weight_entries:
    for target in link_weight_entries[node]:
      weight = 0
      for ts in link_weight_entries[node][target]:
        dt_ts = datetime.fromtimestamp(ts)
        weight += w * math.exp(e * -(currentDate - dt_ts).days)
      #print(link_weight_entries[node][target])
      if weight > max_weight:
        max_weight = weight
      if weight < min_weight:
        min_weight = weight
      if weight > 1:
        graph.add_weighted_edges_from([(node, target, weight)])

  return graph

def clusterPlot(graph, steps):
  communities = community.best_partition(graph)
  cl_dict = {}
  cl_list = []
  for node in communities:
    if node not in cl_dict:
      cl_dict[node] = (random.random(), random.random(), random.random())
  
  for node in graph.nodes:
    cl_list.append(cl_dict[node])
  
  #nx.draw_networkx(graph, cl_list)
  #print(graph.nodes)
  nx.write_gpickle(graph, "step_" + str(steps))
  #nx.write_weighted_edgelist(graph, "step_" + str(steps))
  return communities

def node_weights(graph):
  ret = {}
  
  for n in graph.nodes:
    sumn = 0.0
    for link in graph[n]:
      sumn += graph[n][link]['weight']
    ret[n] = sumn
  
  return ret

def graph_properties(graph, prevgraph):
  num_edges = graph.number_of_edges()
  sum_weights_symmetric = np.sum(nx.to_numpy_array(graph)) / 2
  ewpl_wij = []
  spl = []
  prev_sum_weights_symmetric = np.sum(nx.to_numpy_array(prevgraph)) / 2
  nodeweights = node_weights(graph)

  for t in graph.edges:
    sum1 = nodeweights[t[0]]
    
    sum2 = nodeweights[t[1]]
    
    wij = graph[t[0]][t[1]]['weight']
    
    ewpl_wij.append((wij, ((sum1 - wij) * (sum2 - wij)) ))
  
  for n in graph.nodes:
    spl.append((nodeweights[n], graph.degree[n]))
  
  largest_eigenvalue = max(np.linalg.eigvals(nx.to_numpy_array(graph)))
  
  diffweights = abs(sum_weights_symmetric - prev_sum_weights_symmetric)
  
  return (num_edges, sum_weights_symmetric, ewpl_wij, spl, largest_eigenvalue, diffweights)
  
  
def log_linear_reg_analysis(X, y):
  
  X = np.array(X).reshape(-1, 1) + 1e-5
  y = np.array(y).reshape(-1, 1) + 1e-5
  X = np.log10(X).astype(float)
  y = np.log10(y).astype(float)

  
  model = LinearRegression()
  
  model.fit(X, y)
  
  return [model.coef_, model.intercept_, model.score(X, y)]

def log_linear_reg_analysis_with_bin(X, y):
  X = np.array(X).reshape(-1, 1) + 1e-5
  y = np.array(y).reshape(-1, 1) + 1e-5
  X = np.log10(X)
  y = np.log10(y)
  
  recur_bin = RecursiveOptimalBin(max_bins=10)
  bins = recur_bin.optimal_binning(X.ravel().tolist(), y.ravel().tolist())

  #print(type(bins))
  bin_means, bin_edges, binnumber = binned_statistic(X.ravel().tolist(), y.ravel().tolist(), statistic='median', bins=bins)
  
  model = LinearRegression()
  
  i = 0
  while(True):
    if i >= len(bin_means):
      break
    if np.isnan(bin_means[i]):
      bin_means = np.delete(bin_means, [i])
      bins.pop(i + 1)
    else:
      i += 1
  #print(bins, bin_means[1])
  model.fit(np.array(bins[1:]).reshape(-1, 1), bin_means.reshape(-1, 1))
  
  return [model.coef_, model.intercept_, model.score(np.array(bins[1:]).reshape(-1, 1), bin_means.reshape(-1, 1))]

def analyse_snapshot_type_property(arr):
  X = []
  y = []
  for a in arr:
    X.append(a[0])
    y.append(a[1])
  
  return log_linear_reg_analysis_with_bin(X, y)



def analyse_properties(properties_array):
  num_edges_series = []
  total_weight_series = []
  diffweights = []
  eigenvalues = []
  
  ewpl_stats = []
  spl_stats = []

  for prop in properties_array:
    print(properties_array.index(prop))
    num_edges_series.append(prop[0])
    total_weight_series.append(prop[1])
    
    if prop[0] > 0:
      ewpl_stats.append(analyse_snapshot_type_property(prop[2]))
    
    if prop[0] > 0:
      spl_stats.append(analyse_snapshot_type_property(prop[3]))
    #analyseeqpl
    #analysespl
    eigenvalues.append(prop[4])
    diffweights.append(prop[5])
  
  wpl_stats = log_linear_reg_analysis(total_weight_series, num_edges_series)
  eigenvalue_pl_stats = log_linear_reg_analysis(eigenvalues, num_edges_series)
  
  return wpl_stats, ewpl_stats, spl_stats, entropy(diffweights), eigenvalue_pl_stats

print(__name__)

if __name__ == "__main__":
  
  """parser = ArgumentParser(description='Get arguments')
  parser.add_argument('--step-size', type=int, nargs=1, default=100, dest='step_size')
  parser.add_argument('-e', type=float, nargs=1, default=0.01, dest='e')
  parser.add_argument('-w', type=float, nargs=1, default=10.0, dest='w')
  parser.add_argument('--num-users', type=int, nargs=1, default=200, dest='num_users')
  args = parser.parse_args()
  print("yo")
  print(args)
  
  if type(args.e) == float:
    e = args.e
  else:
    e = args.e[0]
  if type(args.w) == float:
    w = args.w
  else:
    w = args.w[0]
  
  if type(args.step_size) == int:
    step_size = args.step_size
  else:
    step_size = args.step_size[0]
  
  if type(args.num_users) == int:
    num_users = args.num_users
  else:
    num_users = args.num_users[0]"""

  
  link_weight_entries = {}
  """
  link_weight_entries=
  {
    node1:{
          node2: [ts1, ts2, ts3,...],
          noden: [ts1, ts2, ts3,...],
          }
  }
  """
  
  with open("users_com3.pkl", "rb") as f:
    users_com3 = pickle.load(f)
    users_com3 = users_com3[0:1000]
  
  """users_com3 = [i for i in range(1, num_users + 1)]"""
  fw = open("facebook-wall.txt.anon", "r")
  
  ref_date = datetime.fromtimestamp(1095135831)
  timestep = timedelta(days=step_size)
  ograph = nx.Graph()
  
  ref_date_arr = [ref_date]
  nb = 10000000
  
  steps = 0
  firstsnaphot = True
  maxuser = 63891
  
  properties_array = []
  for line in fw.readlines():
    
    nb -= 1
    if nb == 0:
      break
    
    post = list(map(int, line.strip().split()))
    
    if datetime.fromtimestamp(post[2]) > (ref_date + timestep):
      
      ref_date = ref_date + timestep
      ref_date_arr.append(ref_date)
      #Update graph and visualize stuff
      newgraph = getCurrentGraph(link_weight_entries, ref_date, maxuser=maxuser, userlist=users_com3)
      
      if firstsnaphot:
        firstsnaphot = False
      else:
        if len(newgraph.nodes) > 0:
          properties_array.append(graph_properties(newgraph, graph))
      
      graph = newgraph
      
        
      clusterPlot(graph, steps)
      steps += 1
      #print(graph)
      #break
  
    user1 = post[0]
    user2 = post[1]
    
    if user1 not in users_com3 and user2 not in users_com3:
      continue
    
    #print(user1, user2, post[2])
    if user1 == user2:
      continue
    if user1 > user2:
      user1, user2 = user2, user1
    
    if user1 not in link_weight_entries:
      link_weight_entries[user1] = {}
    
    if user2 not in link_weight_entries[user1]:
      link_weight_entries[user1][user2] = []
    
    link_weight_entries[user1][user2].append(int(post[2]))
    
  
  result = analyse_properties(properties_array)
  #fl = open("facebook-links.txt.anon", "r")
  
  print("WPL: Coefficient=" + str(result[0][0][0][0]) + " Score=" + str(result[0][2]))
  print("LWPL: Coefficient=" + str(result[4][0][0][0]) + " Score=" + str(result[4][2]))
  
  spl_coeffs = []
  spl_scores = []
  
  for res in result[1]:
    spl_coeffs.append(res[0][0][0])
    spl_scores.append(res[2])
  
  #print(spl_coeffs)
  #print(spl_scores)
  spl_coeffs = spl_coeffs[int(0.25 * len(spl_coeffs)):]
  spl_scores = spl_scores[int(0.25 * len(spl_scores)):]
  print("Average SPL: Coefficient=" + str(sum(spl_coeffs) / len(spl_coeffs)) + " Score=" + str(sum(spl_scores) / len(spl_scores)))
  
  ewpl_coeffs = []
  ewpl_scores = []
  
  for res in result[2]:
    ewpl_coeffs.append(res[0][0][0])
    ewpl_scores.append(res[2])
  
  #print(spl_coeffs)
  #print(spl_scores)
  
  ewpl_coeffs = ewpl_coeffs[int(0.25 * len(ewpl_coeffs)):]
  ewpl_scores = ewpl_scores[int(0.25 * len(ewpl_scores)):]
  print("Average EWPL: Coefficient=" + str(sum(ewpl_coeffs) / len(ewpl_coeffs)) + " Score=" + str(sum(ewpl_scores) / len(ewpl_scores)))


"""with open("sample.csv", "w") as f:
  am = nx.convert_matrix.to_numpy_array(graph, nodelist=sorted(graph.nodes))
  np.savetxt("sample.csv", X=am, delimiter=",", fmt="%.2f")"""