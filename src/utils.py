import torch
import numpy as np

from sklearn.metrics import roc_curve, auc
from typing import Iterable, Tuple, Optional
import random
import matplotlib.pyplot as plt
import networkx as nx

def get_metrics(
    model: torch.nn.Module,
    loader: Iterable,
    seed: Optional[int] = 1,
    exp: bool = True,
    takeLast: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    
    """Run a model on the a dataset and calculate ROC and AUC.

    The model output values are by default exponentiated before calculating the ROC.
    XXX Why?

    When the model outputs a list of values instead of a tensor, the last element in the
    sequence is used by this function.

    :param model: model to test
    :param loader: data loader
    :param seed: PyTorch random seed; set to `None` to avoid setting the seed
    :param exp: if `True`, exponential model outputs before calculating ROC
    :param takeLast: Some architectures produce multiple outputs from intermediate layers.
                    If True, take the final prediction, if False take average of all predictions.
    :return: a tuple `(fpr, tpr, auc_value, ys, outs)`, where `(fpr, tpr)` are vectors
        representing the ROC curve; `auc_value` is the AUC; `ys` and `outs` are the
        expected (ground-truth) outputs and the (exponentiated) model outputs,
        respectively
    """
    if seed is not None:
        # keep everything reproducible!
        torch.manual_seed(seed)

    # make sure the model is in evaluation mode
    model.eval()

    outs = []
    ys = []
    device = next(iter(model.parameters())).device
    with torch.no_grad():
        for tb in loader:
            if hasattr(tb,"subject_id"):
                tb = tb.to(device)
                y=tb.y
            else:
                x,y=tb
                tb=x
            
            output = model(tb)
    
            # handle multiple outputs
            if not torch.is_tensor(output):
                assert hasattr(output, "__getitem__")
                ## Either take last prediction
                if takeLast:
                    output = output[-1].cpu().numpy()
                ## Or average over all
                else:
                    output = np.mean(np.array(output),axis=0)
            else:
                output = output.cpu().numpy()
                
            if exp:
                output = np.exp(output)
            outs.append(output)
    
            ys.append(y.cpu().numpy())

    outs = np.concatenate(outs)
    ys = np.concatenate(ys)
    if len(outs.shape) == 1 or outs.shape[0] == 1 or outs.shape[1] == 1:
        outs = np.column_stack([1 - outs, outs])
    fpr, tpr, _ = roc_curve(ys, outs[:, 1])
    auc_value = auc(fpr, tpr)

    return fpr, tpr, auc_value, ys, outs


# graph shenanigans

# def create_random_graph(original_graph):
#     num_nodes = len(original_graph.nodes)
#     num_edges = len(original_graph.edges)

#     # Create an empty graph
#     random_graph = nx.DiGraph()

#     # Add nodes
#     random_graph.add_nodes_from(range(num_nodes))

#     # Add random edges
#     while len(random_graph.edges) < num_edges:
#         source = random.randint(0, num_nodes - 1)
#         target = random.randint(0, num_nodes - 1)
#         if source != target and not random_graph.has_edge(source, target):
#             random_graph.add_edge(source, target)

#     return random_graph


def remove_edges_by_pathway(graph, pathway_name):
    edges_to_remove = [(u, v) for u, v, d in graph.edges(data=True) if pathway_name in d.get('pathway', '')]
    graph.remove_edges_from(edges_to_remove)

def visualize_graph(graph, title="Graph"):
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=50, font_size=8, node_color='skyblue', edge_color='gray')
    plt.title(title)
    plt.show()

def visualize_tree(tree, title="Tree"):
    pos = nx.spring_layout(tree)
    nx.draw(tree, pos, with_labels=True, node_size=50, font_size=8, node_color='lightgreen', edge_color='gray')
    plt.title(title)
    plt.show()

# Example usage
# random_graph = create_random_graph(reactome.netx)
# visualize_graph(random_graph, title="Random Graph")
# visualize_graph(reactome.netx, title="Reactome Graph")

# copy_graph = reactome.netx.copy()
# pathway_name_to_remove = "R-HSA-69306"
# remove_edges_by_pathway(copy_graph, pathway_name_to_remove)
# visualize_graph(copy_graph, title="Reactome Graph")

# visualize_tree(reactome.get_tree(), title="Reactome Tree")

def compare_graph_metrics(original_graph, randomized_graph):
    metrics = {
        "Degree Distribution": nx.degree_histogram,
        "Clustering Coefficient": nx.average_clustering,
        "Betweenness Centrality": nx.betweenness_centrality,
    }

    results = {}
    for metric_name, metric_func in metrics.items():
        original_metric = metric_func(original_graph)
        randomized_metric = metric_func(randomized_graph)
        results[metric_name] = (original_metric, randomized_metric)

    return results

def plot_degree_distribution(original_graph, randomized_graph):
    original_degrees = nx.degree_histogram(original_graph)
    randomized_degrees = nx.degree_histogram(randomized_graph)

    plt.figure(figsize=(10, 5))
    plt.plot(original_degrees, label="Original Graph")
    plt.plot(randomized_degrees, label="Randomized Graph")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title("Degree Distribution")
    plt.legend()
    plt.show()

def calculate_edge_overlap(original_graph, randomized_graph):
    original_edges = set(original_graph.edges)
    randomized_edges = set(randomized_graph.edges)
    overlap = original_edges.intersection(randomized_edges)
    overlap_ratio = len(overlap) / len(original_edges)
    return overlap_ratio

# # Compare graph metrics
# original_graph = reactome.netx

# metrics_comparison = compare_graph_metrics(original_graph, randomized_reactome.netx)
# plot_degree_distribution(original_graph, randomized_reactome.netx)

# # Calculate edge overlap
# overlap_ratio = calculate_edge_overlap(original_graph, randomized_reactome.netx)
# print(f"Edge Overlap Ratio: {overlap_ratio:.2f}")

# # Visualize graphs
# visualize_graph(original_graph, title="Original Reactome Graph")
# visualize_graph(randomized_reactome.netx, title="Randomized Reactome Graph")