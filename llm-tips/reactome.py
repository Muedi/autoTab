# %%
# Dependencies
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import networkx as nx

# %%
# Import the Reactome pathway relations.
# This includes which pathways are parents of which other pathways, in format parent -> child.
# However it does not specify anything about genes specifically;
# that is information found instead in the gmt files with pathway definitions.
data_dir = pathlib.Path('../lib/cancer-net/data/reactome')
reactome_file = data_dir / 'ReactomePathwaysRelation.txt'
pathway_rels = pd.read_csv(reactome_file, header=None, sep='\t', names=['Parent', 'Child'])

# %%
# Convert pathway relations into a directed graph
G = nx.DiGraph()
G.add_edges_from(pathway_rels.values)
# Count the number of nodes and edges overall
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
# Count the number of root nodes
root_nodes = [node for node in G.nodes() if G.in_degree(node) == 0]
num_root_nodes = len(root_nodes)
print("Number of root nodes:", num_root_nodes)
# Count the number of leaf nodes
leaf_nodes = [node for node in G.nodes() if G.out_degree(node) == 0]
num_leaf_nodes = len(leaf_nodes)
print("Number of leaf nodes:", num_leaf_nodes)

# %%
# Visualize the graph
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
nx.draw_networkx_edges(G, pos, edge_color='gray', arrowsize=10)
nx.draw_networkx_labels(G, pos, font_size=8)
plt.title('Reactome Pathway Relations')
plt.axis('off')
plt.show()

# %%
