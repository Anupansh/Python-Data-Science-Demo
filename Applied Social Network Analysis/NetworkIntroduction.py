# Network (or Graph): A representation of connections among a set of items.
# Items are called nodes (or vertices)
# Connections are called edges (or link or ties)

# Undirected graph - A normal graph with no direction and symmetric relationship between nodes

import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
G.add_edge('A', 'B')
G.add_edge('B', 'C')
nx.draw(G, with_labels=True)

# Directed Graphs - Asymmetric relationships that is edges have a direction like in food chain from bottom
# to top an animal at bottom will be eaten by an animal at top

plt.figure()
G = nx.DiGraph()
G.add_edge('B', 'A')
G.add_edge('B', 'C')
nx.draw(G, with_labels=True)

# Weighted Networks - A network where weights are assigned to edges like number of times co-workers had
# lunch together
plt.figure()
G = nx.Graph()
G.add_edge('A', 'B', weight=6)
G.add_edge('B', 'C', weight=13)
nx.draw(G, with_labels=True)

# Signed Network - A network where edges are assigned some signs like positive or negative sign like a node is a
# friend (+) of other or enemy (-) of other
plt.figure()
G = nx.Graph()
G.add_edge('A', 'B', sign='+')
G.add_edge('B', 'C', sign='-')
G.add_edge('A', 'D', sign='-')
G.add_edge('A', 'E', sign='+')
nx.draw(G, with_labels=True)

# Other edge attributes - Edges can carry many other label and attributes
plt.figure()
G = nx.Graph()
G.add_edge('A', 'B', relation='friend')
G.add_edge('B', 'C', relation='coworker')
G.add_edge('D', 'E', relation='family')
G.add_edge('E', 'I', relation='neighbor')
nx.draw(G, with_labels=True)

# Multigraphs - A pair of nodes can have different types of relationships simultaneously. A network where multiple
# edges can connect the same nodes (parallel edges).
plt.figure()
G = nx.MultiGraph()
G.add_edge('A', 'B', relation='friend')
G.add_edge('A', 'B', relation='neighbor')
G.add_edge('G', 'F', relation='family')
G.add_edge('G', 'F', relation='coworker')
nx.draw(G, with_labels=True)

# --------------------------- Accessing the edges and vertices of graphs -------------------------------------

plt.figure()
G = nx.Graph()
G.add_edge('A', 'B', weight=6, relation='family')
G.add_edge('B', 'C', weight=13, relation='friend')
print(G.edges)
print(G.edges(data=True))  # List of all edges with attributes
print(G.edges(data='relation'))  # List of all edges with attribute ‘relation’
# plt.show()

# Accessing attributes of a specific edge:

print(G.edges['A', 'B'])
print(G.edges['B', 'C']['weight'])
print(G.edges['C', 'B']['weight'])  # Undirected graph. Order does not matter

# ----------------------------------  Directed Graph  -------------------------------------------------------

G = nx.DiGraph()
G.add_edge('A', 'B', weight=6, relation='family')
G.add_edge('C', 'B', weight=13, relation='friend')

print(G.edges['C', 'B']['weight'])  # Accessing edge
# print(G.edge['B']['C']['weight']) # Will give error since graph is directed


# ------------------------------------- Multigraphs ---------------------------------------------------------

G = nx.MultiGraph()
G.add_edge('A', 'B', weight=6, relation='family')
G.add_edge('A', 'B', weight=18, relation='friend')
G.add_edge('C', 'B', weight=13, relation='friend')
print(G['A']['B'])  # One dictionary of attributes per (A,B) edge
print(G['A']['B'][0]['weight'])  # undirected graph, order does not matter

# Directed Multigraphs

G = nx.MultiDiGraph()
G.add_edge('A', 'B', weight=6, relation='family')
G.add_edge('A', 'B', weight=18, relation='friend')
G.add_edge('C', 'B', weight=13, relation='friend')
print(G['A']['B'][0]['weight'])  # Accessing edge attributes
# print(G['B']['A'][0]['weight'])  # Will give error since its directed

# -------------------------------- Node Attributes -------------------------------------------------------

G = nx.Graph()
G.add_edge('A', 'B', weight=6, relation='family')
G.add_edge('B', 'C', weight=13, relation='friend')

# Adding node attributes:
G.add_node('A', role='trader')
G.add_node('B', role='trader')
G.add_node('C', role='manager')

# Accessing node attributes

print(G.nodes())  # list of all nodes
print(G.nodes(data=True))  # list of all nodes with attributes
print(G.nodes['C']['role'])
