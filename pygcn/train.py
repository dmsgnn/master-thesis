from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import pickle

from utils import load_data, accuracy
from models import GCN
import torch.utils.benchmark as benchmark

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

print(model)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()

    with torch.autograd.profiler.profile(record_shapes=True, with_flops=True) as prof:
        output = model(features, adj)
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))

    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


def save_binary_info():
    data_file = open("data.log", "w")

    print(type(adj[0][0].item()))

    numpy_labels = labels.numpy().astype(np.int32)
    labels_dim = numpy_labels.shape[0]

    numpy_features = features.numpy().astype(np.float32)
    features_row, features_col = numpy_features.shape

    dense_adj = adj.to_dense()
    numpy_adj = dense_adj.numpy().astype(np.float32)
    adj_row, adj_col = numpy_adj.shape

    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    numpy_out = output.detach().numpy().astype(np.float32)
    out_row, out_col = numpy_out.shape

    ## save log info about data
    data_file.write("*************************************************************\n")
    data_file.write("*** Log data of trained GCN model for accuracy comparison ***\n")
    data_file.write("*************************************************************\n")
    data_file.write(f">> Labels shape: {labels_dim}\n")
    data_file.write(f"{labels}\n")
    data_file.write(f">> Features shape: {features_row}x{features_col}\n")
    data_file.write(f"{features} \n")
    data_file.write(f">> Adjacency matrix shape: {adj_row}x{adj_col}\n")
    data_file.write(f"{adj} \n")
    data_file.write(f">> Output shape: {out_row}x{out_col}\n")
    data_file.write(f"{output} \n")

    data_file.write("Test set results:\n")
    data_file.write("Loss= {:.4f}\n".format(loss_test.item()))
    data_file.write("Accuracy= {:.4f}\n".format(acc_test.item()))

    # Save the tensor dimensions and data to a binary file
    with open('data.bin', 'wb') as file:
        file.write(labels_dim.to_bytes(4, byteorder='little'))
        numpy_labels.tofile(file)
        file.write(features_row.to_bytes(4, byteorder='little'))
        file.write(features_col.to_bytes(4, byteorder='little'))
        numpy_features.tofile(file)
        file.write(adj_row.to_bytes(4, byteorder='little'))
        file.write(adj_col.to_bytes(4, byteorder='little'))
        numpy_adj.tofile(file)
        file.write(out_row.to_bytes(4, byteorder='little'))
        file.write(out_col.to_bytes(4, byteorder='little'))
        numpy_out.tofile(file)


def profile():
    num_threads = torch.get_num_threads()
    model.eval()
    t0 = benchmark.Timer(
        stmt='model(features, adj)',
        setup='import torch',
        globals={'model': model, 'features': features, 'adj': adj},
        num_threads=num_threads)
    print(t0.timeit(1000000))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

## model_script = torch.jit.script(model)
## profile()
## save_binary_info()

# Testing
testing_start = time.time()
test()
testing_end = time.time()
print("Number of classified nodes: " + str(len(idx_test)))
print("Testing time : {:.6f}s".format(testing_end - testing_start))
print("Inference time per node: {:.4f} seconds".format((testing_end - testing_start) / len(idx_test)))
