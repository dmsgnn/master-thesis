import torch
from ogb.graphproppred.mol_encoder import BondEncoder, AtomEncoder
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, \
    Set2Set
import torch.nn.functional as F

from gnn import GNN

import argparse
import numpy as np
from tqdm import tqdm

from torch_geometric.nn.models import GIN

# importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


### GIN convolution along the graph structure
class GINConv(MessagePassing):
    propagate_type = {'x': torch.Tensor, 'edge_attr': torch.Tensor}

    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                       torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        edge_embedding = self.bond_encoder(edge_attr)
        assert isinstance(edge_embedding, torch.Tensor)
        return self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding, size=None))

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GNN(torch.nn.Module):

    def __init__(self, num_tasks=10, num_layer=5, emb_dim=300,
                 gnn_type='gin', residual=False, drop_ratio=0.5, JK="last", graph_pooling="mean"):
        """
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        """

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        ### GNN to generate node embeddings
        self.gnn_node = GNN_node(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual,
                                 gnn_type=gnn_type)

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                            torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2 * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch) -> torch.Tensor:
        """
        forward function used for prediction
        """

        h_node = self.gnn_node(x, edge_index, edge_attr)

        h_graph = self.pool(h_node, batch)

        return self.graph_pred_linear(h_graph)


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layer, emb_dim, drop_ratio=0.5, JK="last", residual=False, gnn_type='gin'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            self.convs.append(GINConv(emb_dim).jittable())

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        ### computing input node embedding

        h_list = [self.atom_encoder(x)]

        for layer, (conv, norm) in enumerate(zip(self.convs, self.batch_norms)):
            assert isinstance(h_list[layer], torch.Tensor)
            h = conv(h_list[layer], edge_index, edge_attr)
            h = norm(h)

            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        ### if self.JK == "last":
        ###    node_representation = h_list[-1]
        ### elif self.JK == "sum":
        ###     node_representation = 0
        ###     for layer in range(self.num_layer + 1):
        ###         node_representation += h_list[layer]

        ### upper lines changed with:
        node_representation = h_list[-1]

        return node_representation


def train(model, device, loader, optimizer, task_type):
    """
    function used to train a model
    """
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        x, edge_index, edge_attr, batch_f = batch.x, batch.edge_index, batch.edge_attr, batch.batch

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(x, edge_index, edge_attr, batch_f)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type:
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()


def eval(model, device, loader, evaluator):
    """
    function used to evaluate the model and make inference
    """
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        x, edge_index, edge_attr, batch_f = batch.x, batch.edge_index, batch.edge_attr, batch.batch

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(x, edge_index, edge_attr, batch_f)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def main():
    """
    main function which allow the user to train a new model and make inference (graph classification)
    """
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    args = parser.parse_args()

    device = torch.device("cpu")

    ### automatic data loading and splitting
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv")

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator("ogbg-molhiv")

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = GNN(gnn_type='gin', num_tasks=dataset.num_tasks, num_layer=args.num_layer, emb_dim=args.emb_dim,
                drop_ratio=args.drop_ratio).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    valid_curve = []
    test_curve = []
    train_curve = []



    for epoch in range(1, 1+1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer, dataset.task_type)

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        # test_perf = eval(model, device, test_loader, evaluator)

        # print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        # train_curve.append(train_perf[dataset.eval_metric])
        # valid_curve.append(valid_perf[dataset.eval_metric])
        # test_curve.append(test_perf[dataset.eval_metric])

    # if 'classification' in dataset.task_type:
    #    best_val_epoch = np.argmax(np.array(valid_curve))
    #    best_train = max(train_curve)
    # else:
    #    best_val_epoch = np.argmin(np.array(valid_curve))
    #    best_train = min(train_curve)

    print('Finished training!')
    # print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    # print('Test score: {}'.format(test_curve[best_val_epoch]))

    model_scripted = torch.jit.script(model)  # Export to TorchScript
    model_scripted.save("gin-script.pt")  # Save

    model_scripted = torch.jit.load("gin-script.pt")
    ## print(model_scripted)
    ## model.eval()
    ## model_load.eval()

    model_scripted.eval()

    for step, batch in enumerate(tqdm(test_loader, desc="Iteration")):
        batch = batch.to(device)
        x, edge_index, edge_attr, batch_f = batch.x, batch.edge_index, batch.edge_attr, batch.batch

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(x, edge_index, edge_attr, batch_f)
                scripted_pred = model_scripted(x, edge_index, edge_attr, batch_f)

            print('=====>> Inference normal model')
            print(pred)
            print('=====>> Inference scripted model')
            print(scripted_pred)

            break


if __name__ == "__main__":
    main()
