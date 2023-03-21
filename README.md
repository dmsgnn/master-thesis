# __GNN acceleration Master thesis__

## How to run

### training

Training can be done by running the main script without the inference args parameter. The model will be automatically saved.

```shell
python main_pyg.py --dataset ogbg-molhiv --gnn gin --epochs 1 --batch_size 1 
```

### inference

Inference can be done using the inference args parameter set to true.

```shell
python main_pyg.py --dataset ogbg-molhiv --gnn gin --epochs 1 --batch_size 1 --inference true
```


## Conda environment

The conda environment (macOS) used for the execution of the script is the following.

```shell
conda create -n pygeometric python=3.7

conda activate pygeometric

conda install pytorch torchvision -c pytorch

conda install -y clang_osx-64 clangxx_osx-64 gfortran_osx-64

MACOSX_DEPLOYMENY_TARGET=10.9 CC=clang CXX=clang++

pip install torch_scatter

pip install torch_sparse

pip install torch_cluster

pip install torch-spline-conv

pip install torch_geometric

pip install ogb

pip install rdkit

pip install networkx

pip install matplotlib
```
