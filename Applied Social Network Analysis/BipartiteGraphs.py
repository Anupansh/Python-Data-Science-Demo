# Bipartite graphs is a graph in which graph is divided into two parts or sets in which each set
# performs a relation/edge to another set only but there are no relations between the same sets. Like
# there are two sets one of fans and other of football team. In which a team may have a fan and a fan may have
# a team but no fan or team has a relation with each other.

import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt

B = nx.Graph()  # No separate class for bipartite graphs
B.add_nodes_from(['A', 'B', 'C', 'D', 'E'], bipartite=0)  # label one set of nodes 0
B.add_nodes_from([1, 2, 3, 4], bipartite=1)  # label other set of nodes 1
B.add_edges_from([('A', 1), ('B', 1), ('C', 1), ('D', 2), ('C', 3), ('E', 3), ('E', 4)])
print(bipartite.is_bipartite(B))  # Check if B is bipartite
B.add_edge('A', 'B')
print(bipartite.is_bipartite(B))
B.remove_edge('A', 'B')

# Checking if a set of nodes is a bipartition of a graph
X = {1, 2, 3, 4}
print(bipartite.is_bipartite_node_set(B, X))
X = {'A', 'B', 'C', 'D', 'E'}
print(bipartite.is_bipartite_node_set(B, X))
X = {'A', 'B', 'C', 'D', 'E', 1}
print(bipartite.is_bipartite_node_set(B, X))

# Getting each set of nodes of a bipartite graph
# print(bipartite.sets(B))g
# B.add_edge('A', 'B')
# print(bipartite.sets(B)) # Will give error
# B.remove_edge('A', 'B')


# ------------------------------------------   Projected Graphs  ----------------------------------------
# L-Bipartite graph projection: Network of nodes in group L, where a pair of nodes is connected if they have a
# common neighbor in R in the bipartite graph.
# Similar definition for RBipartite graph projection
B = nx.Graph()
B.add_edges_from([('A', 1), ('B', 1), ('C', 1), ('D', 1), ('H', 1), ('B', 2), ('C', 2), ('D', 2), ('E', 2), ('G', 2),
                  ('E', 3), ('F', 3), ('H', 3), ('J', 3), ('E', 4), ('I', 4), ('J', 4)])
X = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'}
P = bipartite.projected_graph(B, X)

B = nx.Graph()
B.add_edges_from([('A', 1), ('B', 1), ('C', 1), ('D', 1), ('H', 1), ('B', 2), ('C', 2), ('D', 2), ('E', 2),
                  ('G', 2), ('E', 3), ('F', 3), ('H', 3), ('J', 3), ('E', 4), ('I', 4), ('J', 4)])
X = {1, 2, 3, 4}
P = bipartite.projected_graph(B, X)

plt.figure()
X = {'A','B','C','D','E'}
P = bipartite.weighted_projected_graph(B, X)
nx.draw(P,with_labels=True)
plt.show()

B = nx.Graph()
B.add_nodes_from([1, 2, 3, 4], bipartite=0)  # Add the node attribute "bipartite"
B.add_nodes_from(['a', 'b', 'c'], bipartite=1)
B.add_edges_from([(1, 'a'), (1, 'b'), (2, 'b'), (2, 'c'), (3, 'c'), (4, 'a')])
print(bipartite.sets(B))
