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
# Select a subset of 20 root nodes and their children
subset_root_nodes = root_nodes[:10]
subset_child_nodes = [child for node in subset_root_nodes for child in G.successors(node)]
subset_nodes = subset_root_nodes + subset_child_nodes
subset_edges = [(parent, child) for parent in subset_root_nodes for child in G.successors(parent)]
# Draw the subset of nodes and edges
G2 = G.subgraph(subset_nodes)
#pos = nx.spring_layout(G2, seed=42)
# Create a bipartite layout for root nodes vs others
pos = nx.shell_layout(G2, nlist=[subset_root_nodes, subset_child_nodes])

# Color the root nodes differently
node_colors = ['red' if node in subset_root_nodes else 'blue' for node in G2.nodes()]

nx.draw_networkx_nodes(G2, pos, node_size=500, node_color=node_colors)
nx.draw_networkx_edges(G2, pos, edge_color='red', arrowsize=10)
nx.draw_networkx_labels(G2, pos, font_size=8)

plt.title('Reactome Pathway Relations')
plt.axis('off')
# %%
