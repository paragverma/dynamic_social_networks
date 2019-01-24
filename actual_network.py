import networkx as nx
import community
import pickle 
from collections import Counter

g = nx.Graph()



with open("facebook-wall.txt.anon", "r") as f:
  for line in f.readlines():
    sl = line.strip().split()

    g.add_edge(int(sl[0]), int(sl[1]))

communities = community.best_partition(g)

with open("full_network_community.pkl", "wb") as f:
  pickle.dump(communities, f)

numcomms = max(communities.values())

vals = Counter(communities.values())

users_com3 = []

for ci in communities:
    if communities[ci] == 3:
        users_com3.append(ci)

with open("users_com3.pkl", "wb") as f:
  pickle.dump(users_com3, f)