# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import sys

from PIL import Image
import requests

import sys

sys.path.insert(1, '/Users/dvlpr/torch-mlir/build/tools/torch-mlir/python_packages/torch_mlir/')

import torch_mlir
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend



import torch
from torchvision import transforms
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from gnn import GNN
import torch.optim as optim

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator


def train(model, device, loader, optimizer, task_type):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
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
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def predictions(torch_model, jit_model):
    pytorch_prediction = eval(torch_model, "cpu", test_loader, evaluator)
    print("PyTorch prediction")
    print(pytorch_prediction)
    mlir_prediction = eval(jit_model, "cpu", test_loader, evaluator)
    print("torch-mlir prediction")
    print(mlir_prediction)


cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

### automatic data loading and splitting
dataset = PygGraphPropPredDataset(name="ogbg-molhiv")

split_idx = dataset.get_idx_split()

### automatic evaluator. takes dataset name as input
evaluator = Evaluator("ogbg-molhiv")

train_loader = DataLoader(dataset[split_idx["train"]], batch_size=1, shuffle=True,
                          num_workers=0)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=1, shuffle=False,
                          num_workers=0)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=1, shuffle=False,
                         num_workers=0)

gin = GNN(gnn_type='gin', num_tasks=dataset.num_tasks, num_layer=5, emb_dim=300,
          drop_ratio=0.5, virtual_node=False).to("cpu")

optimizer = optim.Adam(gin.parameters(), lr=0.001)
train(gin, "cpu", train_loader, optimizer, dataset.task_type)

module = torch_mlir.compile(gin, torch.ones(1, 3, 224, 224), output_type="linalg-on-tensors")
backend = refbackend.RefBackendLinalgOnTensorsBackend()
compiled = backend.compile(module)
jit_module = backend.load(compiled)

predictions(gin, jit_module)
