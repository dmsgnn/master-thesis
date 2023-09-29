import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool

# -------- Dataset download --------
dataset = PygGraphPropPredDataset(name='ogbg-molhiv')

# Print information about the dataset
print(f'Dataset: {dataset}')
print('--------------------')
print(f'Number of graphs\t: {len(dataset)}')
print(f'Number of nodes\t\t: {dataset[0].x.shape[0]}')
print(f'Number of features\t: {dataset.num_features}')
print(f'Number of classes\t: {dataset.num_classes}')

print("### dataset")
print(dataset.data)
# x is the node feature, it contains the number of nodes multiplied by the number of node features
# y is the node labeling, it tells to which category the node belongs

### example of graph 0
print("### data of graph 0")
print(dataset[0])
print(dataset[0].edge_index)  # edge index specifies which node is connected to which other node

# -------- 3D Plot --------
G = to_networkx(dataset[0], to_undirected=True)

# 3D spring layout
pos = nx.spring_layout(G, dim=3, seed=0)

# Extract node and edge positions from the layout
node_xyz = np.array([pos[v] for v in sorted(G)])
edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

# Create the 3D figure
fig = plt.figure(figsize=(16, 16))
ax = fig.add_subplot(111, projection="3d")

# Suppress tick labels
for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
    dim.set_ticks([])

# Plot the nodes - alpha is scaled by "depth" automatically
ax.scatter(*node_xyz.T, s=500, c="#0A047A")

# Plot the edges
for vizedge in edge_xyz:
    ax.plot(*vizedge.T, color="tab:gray")

# fig.tight_layout()
plt.show()
