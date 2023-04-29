import numpy as np
import torch
from torch_scatter import segment_csr
from torch_sparse import storage

import sys
sys.path.insert(1, '/Users/dvlpr/torch-mlir/build/tools/torch-mlir/python_packages/torch_mlir/')
import torch_mlir
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend


graph = """
graph(%self.1 : __torch__.___torch_mangle_0.GNN,
      %x.1 : Tensor,
      %edge_index.1 : Tensor,
      %edge_attr.2 : Tensor,
      %batch.1 : Tensor):
  %1657 : Float(1, strides=[1], requires_grad=0, device=cpu) = prim::Constant[value={1.06454}]()
  %1646 : Float(1, strides=[1], requires_grad=0, device=cpu) = prim::Constant[value={1.0462}]()
  %1635 : Float(1, strides=[1], requires_grad=0, device=cpu) = prim::Constant[value={0.993363}]()
  %1624 : Float(1, strides=[1], requires_grad=0, device=cpu) = prim::Constant[value={0.90717}]()
  %1619 : int[] = prim::Constant[value=[1]]()
  %1613 : int[] = prim::Constant[value=[0, 1, 3, 4]]()
  %1611 : Float(1, strides=[1], requires_grad=0, device=cpu) = prim::Constant[value={0.875899}]()
  %43 : int = prim::Constant[value=0]() # /Users/dvlpr/PyCharmProjects/gnn-acceleration-master-thesis/gnn/gin.py:147:8
  %42 : int = prim::Constant[value=1]() # /Users/dvlpr/PyCharmProjects/gnn-acceleration-master-thesis/gnn/gin.py:152:41
  %41 : int = prim::Constant[value=2]() # /Users/dvlpr/PyCharmProjects/gnn-acceleration-master-thesis/gnn/gin.py:147:8
  %40 : int = prim::Constant[value=3]() # /Users/dvlpr/PyCharmProjects/gnn-acceleration-master-thesis/gnn/gin.py:147:8
  %39 : int = prim::Constant[value=4]() # /Users/dvlpr/PyCharmProjects/gnn-acceleration-master-thesis/gnn/gin.py:147:8
  %38 : int = prim::Constant[value=-1]() # /Users/dvlpr/PyCharmProjects/gnn-acceleration-master-thesis/gnn/gin.py:172:37
  %36 : int = prim::Constant[value=8]() # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:29:48
  %35 : int = prim::Constant[value=7]() # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:28:48
  %34 : int = prim::Constant[value=6]() # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:27:48
  %30 : str = prim::Constant[value="expected 2D or 3D input (got {}D input)"]() # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:302:16
  %27 : bool = prim::Constant[value=1]() # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:81:11
  %25 : str = prim::Constant[value="Expected \'edge_index\' to be of integer type (got \'{}\')"]() # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:58:33
  %24 : str = prim::Constant[value="Expected \'edge_index\' to be two-dimensional (got {} dimensions)"]() # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:61:33
  %23 : str = prim::Constant[value="Expected \'edge_index\' to have size \'2\' in the first dimension (got \'{}\')"]() # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:64:33
  %21 : str = prim::Constant[value="builtins.ValueError"]() # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:44:22
  %20 : str = prim::Constant[value="Encountered tensor with size {} in dimension {}, but expected size {}."]() # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:246:17
  %16 : str = prim::Constant[value="The `dim` argument must lay between 0 and {} (got {})"]() # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:52:29
  %15 : str = prim::Constant[value="The `index` argument must be one-dimensional (got {} dimensions)"]() # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:46:29
  %6 : NoneType = prim::Constant() # :0:0
  %self.graph_pred_linear.bias : Float(1, strides=[1], requires_grad=0, device=cpu) = prim::Constant[value={-0.109641}]()
  %self.graph_pred_linear.weight : Float(1, 300, strides=[300, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.convs.4.bond_encoder.bond_embedding_list.2.weight : Float(2, 300, strides=[300, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.convs.4.bond_encoder.bond_embedding_list.1.weight : Float(6, 300, strides=[300, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.convs.4.bond_encoder.bond_embedding_list.0.weight : Float(5, 300, strides=[300, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.convs.3.bond_encoder.bond_embedding_list.2.weight : Float(2, 300, strides=[300, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.convs.3.bond_encoder.bond_embedding_list.1.weight : Float(6, 300, strides=[300, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.convs.3.bond_encoder.bond_embedding_list.0.weight : Float(5, 300, strides=[300, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.convs.2.bond_encoder.bond_embedding_list.2.weight : Float(2, 300, strides=[300, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.convs.2.bond_encoder.bond_embedding_list.1.weight : Float(6, 300, strides=[300, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.convs.2.bond_encoder.bond_embedding_list.0.weight : Float(5, 300, strides=[300, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.convs.1.bond_encoder.bond_embedding_list.2.weight : Float(2, 300, strides=[300, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.convs.1.bond_encoder.bond_embedding_list.1.weight : Float(6, 300, strides=[300, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.convs.1.bond_encoder.bond_embedding_list.0.weight : Float(5, 300, strides=[300, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.num_layer : int = prim::Constant[value=5]()
  %self.gnn_node.convs.0.mlp.1.training : bool = prim::Constant[value=0]()
  %self.gnn_node.convs.0.node_dim : int = prim::Constant[value=-2]()
  %self.gnn_node.convs.0.bond_encoder.bond_embedding_list.2.weight : Float(2, 300, strides=[300, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.convs.0.bond_encoder.bond_embedding_list.1.weight : Float(6, 300, strides=[300, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.convs.0.bond_encoder.bond_embedding_list.0.weight : Float(5, 300, strides=[300, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.atom_encoder.atom_embedding_list.8.weight : Float(2, 300, strides=[300, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.atom_encoder.atom_embedding_list.7.weight : Float(2, 300, strides=[300, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.atom_encoder.atom_embedding_list.6.weight : Float(6, 300, strides=[300, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.atom_encoder.atom_embedding_list.5.weight : Float(6, 300, strides=[300, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.atom_encoder.atom_embedding_list.4.weight : Float(10, 300, strides=[300, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.atom_encoder.atom_embedding_list.3.weight : Float(12, 300, strides=[300, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.atom_encoder.atom_embedding_list.2.weight : Float(12, 300, strides=[300, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.atom_encoder.atom_embedding_list.1.weight : Float(5, 300, strides=[300, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.atom_encoder.atom_embedding_list.0.weight : Float(119, 300, strides=[300, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %47 : Tensor = aten::slice(%x.1, %43, %6, %6, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:21:51
  %48 : Tensor = aten::select(%47, %42, %43) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:21:51
  %50 : Tensor = aten::embedding(%self.gnn_node.atom_encoder.atom_embedding_list.0.weight, %48, %38, %self.gnn_node.convs.0.mlp.1.training, %self.gnn_node.convs.0.mlp.1.training) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:2210:11
  %x_embedding.1 : Tensor = aten::add(%50, %43, %42) # <string>:5:9
  %54 : Tensor = aten::slice(%x.1, %43, %6, %6, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:22:51
  %55 : Tensor = aten::select(%54, %42, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:22:51
  %57 : Tensor = aten::embedding(%self.gnn_node.atom_encoder.atom_embedding_list.1.weight, %55, %38, %self.gnn_node.convs.0.mlp.1.training, %self.gnn_node.convs.0.mlp.1.training) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:2210:11
  %x_embedding0.1 : Tensor = aten::add_(%x_embedding.1, %57, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:22:8
  %61 : Tensor = aten::slice(%x.1, %43, %6, %6, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:23:51
  %62 : Tensor = aten::select(%61, %42, %41) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:23:51
  %64 : Tensor = aten::embedding(%self.gnn_node.atom_encoder.atom_embedding_list.2.weight, %62, %38, %self.gnn_node.convs.0.mlp.1.training, %self.gnn_node.convs.0.mlp.1.training) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:2210:11
  %x_embedding1.1 : Tensor = aten::add_(%x_embedding0.1, %64, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:23:8
  %68 : Tensor = aten::slice(%x.1, %43, %6, %6, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:24:51
  %69 : Tensor = aten::select(%68, %42, %40) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:24:51
  %71 : Tensor = aten::embedding(%self.gnn_node.atom_encoder.atom_embedding_list.3.weight, %69, %38, %self.gnn_node.convs.0.mlp.1.training, %self.gnn_node.convs.0.mlp.1.training) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:2210:11
  %x_embedding2.1 : Tensor = aten::add_(%x_embedding1.1, %71, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:24:8
  %75 : Tensor = aten::slice(%x.1, %43, %6, %6, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:25:51
  %76 : Tensor = aten::select(%75, %42, %39) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:25:51
  %78 : Tensor = aten::embedding(%self.gnn_node.atom_encoder.atom_embedding_list.4.weight, %76, %38, %self.gnn_node.convs.0.mlp.1.training, %self.gnn_node.convs.0.mlp.1.training) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:2210:11
  %x_embedding3.1 : Tensor = aten::add_(%x_embedding2.1, %78, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:25:8
  %82 : Tensor = aten::slice(%x.1, %43, %6, %6, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:26:51
  %83 : Tensor = aten::select(%82, %42, %self.gnn_node.num_layer) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:26:51
  %85 : Tensor = aten::embedding(%self.gnn_node.atom_encoder.atom_embedding_list.5.weight, %83, %38, %self.gnn_node.convs.0.mlp.1.training, %self.gnn_node.convs.0.mlp.1.training) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:2210:11
  %x_embedding4.1 : Tensor = aten::add_(%x_embedding3.1, %85, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:26:8
  %89 : Tensor = aten::slice(%x.1, %43, %6, %6, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:27:51
  %90 : Tensor = aten::select(%89, %42, %34) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:27:51
  %92 : Tensor = aten::embedding(%self.gnn_node.atom_encoder.atom_embedding_list.6.weight, %90, %38, %self.gnn_node.convs.0.mlp.1.training, %self.gnn_node.convs.0.mlp.1.training) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:2210:11
  %x_embedding5.1 : Tensor = aten::add_(%x_embedding4.1, %92, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:27:8
  %96 : Tensor = aten::slice(%x.1, %43, %6, %6, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:28:51
  %97 : Tensor = aten::select(%96, %42, %35) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:28:51
  %99 : Tensor = aten::embedding(%self.gnn_node.atom_encoder.atom_embedding_list.7.weight, %97, %38, %self.gnn_node.convs.0.mlp.1.training, %self.gnn_node.convs.0.mlp.1.training) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:2210:11
  %x_embedding6.1 : Tensor = aten::add_(%x_embedding5.1, %99, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:28:8
  %103 : Tensor = aten::slice(%x.1, %43, %6, %6, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:29:51
  %104 : Tensor = aten::select(%103, %42, %36) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:29:51
  %106 : Tensor = aten::embedding(%self.gnn_node.atom_encoder.atom_embedding_list.8.weight, %104, %38, %self.gnn_node.convs.0.mlp.1.training, %self.gnn_node.convs.0.mlp.1.training) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:2210:11
  %107 : Tensor = aten::add_(%x_embedding6.1, %106, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:29:8
  %h_list.1 : Tensor[] = prim::ListConstruct(%107)
  %121 : Tensor = aten::__getitem__(%h_list.1, %43) # /Users/dvlpr/PyCharmProjects/gnn-acceleration-master-thesis/gnn/gin.py:149:21
  %125 : Tensor = aten::slice(%edge_attr.2, %43, %6, %6, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:49:54
  %126 : Tensor = aten::select(%125, %42, %43) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:49:54
  %128 : Tensor = aten::embedding(%self.gnn_node.convs.0.bond_encoder.bond_embedding_list.0.weight, %126, %38, %self.gnn_node.convs.0.mlp.1.training, %self.gnn_node.convs.0.mlp.1.training) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:2210:11
  %bond_embedding.2 : Tensor = aten::add(%128, %43, %42) # <string>:5:9
  %132 : Tensor = aten::slice(%edge_attr.2, %43, %6, %6, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:50:54
  %133 : Tensor = aten::select(%132, %42, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:50:54
  %135 : Tensor = aten::embedding(%self.gnn_node.convs.0.bond_encoder.bond_embedding_list.1.weight, %133, %38, %self.gnn_node.convs.0.mlp.1.training, %self.gnn_node.convs.0.mlp.1.training) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:2210:11
  %bond_embedding0.2 : Tensor = aten::add_(%bond_embedding.2, %135, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:50:8
  %139 : Tensor = aten::slice(%edge_attr.2, %43, %6, %6, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:51:54
  %140 : Tensor = aten::select(%139, %42, %41) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:51:54
  %142 : Tensor = aten::embedding(%self.gnn_node.convs.0.bond_encoder.bond_embedding_list.2.weight, %140, %38, %self.gnn_node.convs.0.mlp.1.training, %self.gnn_node.convs.0.mlp.1.training) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:2210:11
  %edge_embedding.2 : Tensor = aten::add_(%bond_embedding0.2, %142, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:51:8
  %147 : Tensor = aten::mul(%1611, %121) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:206:25
  %the_size.3 : int?[] = prim::ListConstruct(%6, %6)
  %149 : bool = prim::Uninitialized() # :0:0
  %150 : bool = prim::isinstance[types=[Tensor]](%edge_index.1)
  %151 : bool, %152 : bool = prim::If(%150) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:63:4
    block0():
      %153 : int = prim::layout(%edge_index.1) # :0:0
      %154 : bool = aten::eq(%153, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:64:11
      %155 : bool, %156 : bool = prim::If(%154) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:64:8
        block0():
          -> (%27, %27)
        block1():
          %157 : int = prim::layout(%edge_index.1) # :0:0
          %158 : bool = aten::eq(%157, %41) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:66:11
          %159 : bool, %160 : bool = prim::If(%158) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:66:8
            block0():
              -> (%27, %27)
            block1():
              %161 : int = prim::layout(%edge_index.1) # :0:0
              %162 : bool = aten::eq(%161, %39) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:68:11
              %163 : bool = prim::If(%162) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:68:8
                block0():
                  -> (%27)
                block1():
                  -> (%149)
              -> (%162, %163)
          -> (%159, %160)
      -> (%155, %156)
    block1():
      -> (%self.gnn_node.convs.0.mlp.1.training, %149)
  %164 : bool = prim::If(%151) # :0:0
    block0():
      -> (%152)
    block1():
      -> (%self.gnn_node.convs.0.mlp.1.training)
  %165 : bool = prim::If(%164) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:81:11
    block0():
      -> (%27)
    block1():
      %166 : bool = prim::isinstance[types=[__torch__.torch_sparse.tensor.SparseTensor]](%edge_index.1)
      -> (%166)
   = prim::If(%165) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:42:8
    block0():
      %169 : int = aten::size(%edge_index.1, %42) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:51:26
      %170 : int?[] = aten::_set_item(%the_size.3, %43, %169) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:51:12
      %171 : int = aten::size(%edge_index.1, %43) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:52:26
      %172 : int?[] = aten::_set_item(%the_size.3, %42, %171) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:52:12
      -> ()
    block1():
      %173 : int = prim::dtype(%edge_index.1) # :0:0
      %175 : bool = aten::__contains__(%1613, %173) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:57:15
      %176 : bool = aten::__not__(%175) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:57:15
       = prim::If(%176) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:57:12
        block0():
          %177 : int = prim::dtype(%edge_index.1) # :0:0
          %178 : str = aten::format(%25, %177) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:58:33
           = prim::RaiseException(%178, %21) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:58:16
          -> ()
        block1():
          -> ()
      %179 : int = aten::dim(%edge_index.1) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:60:15
      %180 : bool = aten::ne(%179, %41) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:60:15
       = prim::If(%180) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:60:12
        block0():
          %181 : int = aten::dim(%edge_index.1) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:62:42
          %182 : str = aten::format(%24, %181) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:61:33
           = prim::RaiseException(%182, %21) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:61:16
          -> ()
        block1():
          -> ()
      %183 : int = aten::size(%edge_index.1, %43) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:63:15
      %184 : bool = aten::ne(%183, %41) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:63:15
       = prim::If(%184) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:63:12
        block0():
          %185 : int = aten::size(%edge_index.1, %43) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:66:37
          %186 : str = aten::format(%23, %185) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:64:33
           = prim::RaiseException(%186, %21) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:64:16
          -> ()
        block1():
          -> ()
      -> ()
  %the_size.5 : int? = aten::__getitem__(%the_size.3, %43) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:241:19
  %195 : bool = aten::__is__(%the_size.5, %6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:242:11
   = prim::If(%195) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:242:8
    block0():
      %197 : int = aten::size(%121, %self.gnn_node.convs.0.node_dim) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:243:24
      %198 : int?[] = aten::_set_item(%the_size.3, %43, %197) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:243:12
      -> ()
    block1():
      %the_size0.2 : int = prim::unchecked_cast(%the_size.5)
      %201 : int = aten::size(%121, %self.gnn_node.convs.0.node_dim) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:244:25
      %202 : bool = aten::ne(%the_size0.2, %201) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:244:13
       = prim::If(%202) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:244:8
        block0():
          %204 : int = aten::size(%121, %self.gnn_node.convs.0.node_dim) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:246:49
          %206 : str = aten::format(%20, %204, %self.gnn_node.convs.0.node_dim, %the_size0.2) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:246:17
           = prim::RaiseException(%206, %21) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:245:12
          -> ()
        block1():
          -> ()
      -> ()
  %index.2 : Tensor = aten::select(%edge_index.1, %43, %43) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:89:20
  %x_j0.2 : Tensor = aten::index_select(%121, %self.gnn_node.convs.0.node_dim, %index.2) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:90:19
  %edge_index_i.2 : Tensor = aten::select(%edge_index.1, %43, %42) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:141:27
  %212 : int? = aten::__getitem__(%the_size.3, %42) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:160:28
  %213 : bool = aten::__isnot__(%212, %6) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:160:28
  %size_i.2 : int? = prim::If(%213) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:160:17
    block0():
      %size_i.4 : int? = aten::__getitem__(%the_size.3, %42) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:160:17
      -> (%size_i.4)
    block1():
      %size_i.6 : int? = aten::__getitem__(%the_size.3, %43) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:160:53
      -> (%size_i.6)
  %220 : Tensor = aten::add(%x_j0.2, %edge_embedding.2, %42) # /Users/dvlpr/PyCharmProjects/gnn-acceleration-master-thesis/gnn/gin.py:47:22
  %out.2 : Tensor = aten::relu(%220) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:1457:17
  %226 : bool = aten::__isnot__(%6, %6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/aggr/base.py:150:11
  %out0.2 : Tensor = prim::If(%226) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/aggr/base.py:150:8
    block0():
      %ptr0.2 : Tensor = prim::unchecked_cast(%6)
      %229 : int = aten::dim(%out.2) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/aggr/base.py:151:45
      %232 : int = aten::add(%229, %self.gnn_node.convs.0.node_dim) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/aggr/base.py:186:19
      %ptr1.2 : Tensor = prim::Loop(%232, %27, %ptr0.2) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/aggr/base.py:186:4
        block0(%234 : int, %ptr0.10 : Tensor):
          %ptr0.12 : Tensor = aten::unsqueeze(%ptr0.10, %43) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/aggr/base.py:187:14
          -> (%27, %ptr0.12)
      %237 : Tensor = torch_scatter::segment_sum_csr(%out.2, %ptr1.2, %6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_scatter/segment_csr.py:8:11
      -> (%237)
    block1():
      %238 : int = aten::dim(%edge_index_i.2) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:45:11
      %239 : bool = aten::ne(%238, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:45:11
       = prim::If(%239) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:45:8
        block0():
          %240 : int = aten::dim(%edge_index_i.2) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:47:37
          %241 : str = aten::format(%15, %240) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:46:29
           = prim::RaiseException(%241, %21) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:46:12
          -> ()
        block1():
          -> ()
      %244 : int = aten::dim(%out.2) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:49:14
      %dim0.10 : int = aten::add(%244, %self.gnn_node.convs.0.node_dim) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:49:14
      %246 : bool = aten::lt(%dim0.10, %43) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:51:11
      %247 : bool = prim::If(%246) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:51:11
        block0():
          -> (%27)
        block1():
          %248 : int = aten::dim(%out.2) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:51:29
          %249 : bool = aten::ge(%dim0.10, %248) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:51:22
          -> (%249)
       = prim::If(%247) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:51:8
        block0():
          %250 : int = aten::dim(%out.2) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:53:32
          %251 : int = aten::sub(%250, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:53:32
          %252 : str = aten::format(%16, %251, %dim0.10) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:52:29
           = prim::RaiseException(%252, %21) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:52:12
          -> ()
        block1():
          -> ()
      %253 : bool = aten::__is__(%size_i.2, %6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:55:11
      %dim_size0.1 : int = prim::If(%253) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:55:8
        block0():
          %255 : int = aten::numel(%edge_index_i.2) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:56:47
          %256 : bool = aten::gt(%255, %43) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:56:47
          %dim_size1.1 : int = prim::If(%256) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:56:23
            block0():
              %258 : Tensor = aten::max(%edge_index_i.2) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:56:27
              %259 : int = aten::Int(%258) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:56:23
              %260 : int = aten::add(%259, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:56:23
              -> (%260)
            block1():
              -> (%43)
          -> (%dim_size1.1)
        block1():
          %dim_size0.5 : int = prim::unchecked_cast(%size_i.2)
          -> (%dim_size0.5)
      %262 : int[] = aten::size(%out.2) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:68:20
      %size.11 : int[] = aten::list(%262) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:68:15
      %264 : int[] = aten::_set_item(%size.11, %dim0.10, %dim_size0.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:69:8
      %266 : int = aten::dim(%out.2) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:19:21
      %size.13 : int[] = aten::mul(%1619, %266) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:19:15
      %268 : int[] = aten::_set_item(%size.13, %dim0.10, %38) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:20:8
      %269 : Tensor = aten::view(%edge_index_i.2, %size.13) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:21:15
      %index0.5 : Tensor = aten::expand_as(%269, %out.2) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:21:15
      %271 : Tensor = aten::new_zeros(%out.2, %size.11, %6, %6, %6, %6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:74:19
      %272 : Tensor = aten::scatter_add_(%271, %dim0.10, %index0.5, %out.2) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:74:19
      -> (%272)
  %273 : Tensor = aten::add(%147, %out0.2, %42) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmprb27wjk9.py:206:25
  %self.gnn_node.convs.0.mlp.0.weight_fused_bn : Float(600, 300, strides=[300, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.convs.0.mlp.0.bias_fused_bn : Float(600, strides=[1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %input0.2 : Tensor = aten::linear(%273, %self.gnn_node.convs.0.mlp.0.weight_fused_bn, %self.gnn_node.convs.0.mlp.0.bias_fused_bn) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/linear.py:114:15
  %280 : int = aten::dim(%input0.2) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:11
  %281 : bool = aten::ne(%280, %41) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:11
  %282 : bool = prim::If(%281) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:11
    block0():
      %284 : bool = aten::ne(%280, %40) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:32
      -> (%284)
    block1():
      -> (%self.gnn_node.convs.0.mlp.1.training)
   = prim::If(%282) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:8
    block0():
      %286 : str = aten::format(%30, %280) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:302:16
       = prim::RaiseException(%286, %21) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:301:12
      -> ()
    block1():
      -> ()
  %input2.2 : Tensor = aten::relu(%input0.2) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:1457:17
  %self.gnn_node.convs.0.mlp.3.weight_fused_bn : Float(300, 600, strides=[600, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.convs.0.mlp.3.bias_fused_bn : Float(300, strides=[1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %h.1 : Tensor = aten::linear(%input2.2, %self.gnn_node.convs.0.mlp.3.weight_fused_bn, %self.gnn_node.convs.0.mlp.3.bias_fused_bn) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/linear.py:114:15
  %312 : int = aten::dim(%h.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:11
  %313 : bool = aten::ne(%312, %41) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:11
  %314 : bool = prim::If(%313) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:11
    block0():
      %316 : bool = aten::ne(%312, %40) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:32
      -> (%316)
    block1():
      -> (%self.gnn_node.convs.0.mlp.1.training)
   = prim::If(%314) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:8
    block0():
      %318 : str = aten::format(%30, %312) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:302:16
       = prim::RaiseException(%318, %21) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:301:12
      -> ()
    block1():
      -> ()
  %result.6 : Tensor = aten::relu(%h.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:1457:17
  %363 : Tensor[] = aten::append(%h_list.1, %result.6) # /Users/dvlpr/PyCharmProjects/gnn-acceleration-master-thesis/gnn/gin.py:161:12
  %364 : Tensor = aten::__getitem__(%h_list.1, %42) # /Users/dvlpr/PyCharmProjects/gnn-acceleration-master-thesis/gnn/gin.py:149:21
  %368 : Tensor = aten::slice(%edge_attr.2, %43, %6, %6, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:49:54
  %369 : Tensor = aten::select(%368, %42, %43) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:49:54
  %371 : Tensor = aten::embedding(%self.gnn_node.convs.1.bond_encoder.bond_embedding_list.0.weight, %369, %38, %self.gnn_node.convs.0.mlp.1.training, %self.gnn_node.convs.0.mlp.1.training) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:2210:11
  %bond_embedding.4 : Tensor = aten::add(%371, %43, %42) # <string>:5:9
  %375 : Tensor = aten::slice(%edge_attr.2, %43, %6, %6, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:50:54
  %376 : Tensor = aten::select(%375, %42, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:50:54
  %378 : Tensor = aten::embedding(%self.gnn_node.convs.1.bond_encoder.bond_embedding_list.1.weight, %376, %38, %self.gnn_node.convs.0.mlp.1.training, %self.gnn_node.convs.0.mlp.1.training) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:2210:11
  %bond_embedding0.4 : Tensor = aten::add_(%bond_embedding.4, %378, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:50:8
  %382 : Tensor = aten::slice(%edge_attr.2, %43, %6, %6, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:51:54
  %383 : Tensor = aten::select(%382, %42, %41) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:51:54
  %385 : Tensor = aten::embedding(%self.gnn_node.convs.1.bond_encoder.bond_embedding_list.2.weight, %383, %38, %self.gnn_node.convs.0.mlp.1.training, %self.gnn_node.convs.0.mlp.1.training) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:2210:11
  %edge_embedding.4 : Tensor = aten::add_(%bond_embedding0.4, %385, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:51:8
  %390 : Tensor = aten::mul(%1624, %364) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:206:25
  %the_size.7 : int?[] = prim::ListConstruct(%6, %6)
  %393 : bool = prim::isinstance[types=[Tensor]](%edge_index.1)
  %394 : bool, %395 : bool = prim::If(%393) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:63:4
    block0():
      %396 : int = prim::layout(%edge_index.1) # :0:0
      %397 : bool = aten::eq(%396, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:64:11
      %398 : bool, %399 : bool = prim::If(%397) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:64:8
        block0():
          -> (%27, %27)
        block1():
          %400 : int = prim::layout(%edge_index.1) # :0:0
          %401 : bool = aten::eq(%400, %41) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:66:11
          %402 : bool, %403 : bool = prim::If(%401) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:66:8
            block0():
              -> (%27, %27)
            block1():
              %404 : int = prim::layout(%edge_index.1) # :0:0
              %405 : bool = aten::eq(%404, %39) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:68:11
              %406 : bool = prim::If(%405) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:68:8
                block0():
                  -> (%27)
                block1():
                  -> (%149)
              -> (%405, %406)
          -> (%402, %403)
      -> (%398, %399)
    block1():
      -> (%self.gnn_node.convs.0.mlp.1.training, %149)
  %407 : bool = prim::If(%394) # :0:0
    block0():
      -> (%395)
    block1():
      -> (%self.gnn_node.convs.0.mlp.1.training)
  %408 : bool = prim::If(%407) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:81:11
    block0():
      -> (%27)
    block1():
      %409 : bool = prim::isinstance[types=[__torch__.torch_sparse.tensor.SparseTensor]](%edge_index.1)
      -> (%409)
   = prim::If(%408) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:42:8
    block0():
      %412 : int = aten::size(%edge_index.1, %42) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:51:26
      %413 : int?[] = aten::_set_item(%the_size.7, %43, %412) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:51:12
      %414 : int = aten::size(%edge_index.1, %43) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:52:26
      %415 : int?[] = aten::_set_item(%the_size.7, %42, %414) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:52:12
      -> ()
    block1():
      %416 : int = prim::dtype(%edge_index.1) # :0:0
      %418 : bool = aten::__contains__(%1613, %416) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:57:15
      %419 : bool = aten::__not__(%418) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:57:15
       = prim::If(%419) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:57:12
        block0():
          %420 : int = prim::dtype(%edge_index.1) # :0:0
          %421 : str = aten::format(%25, %420) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:58:33
           = prim::RaiseException(%421, %21) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:58:16
          -> ()
        block1():
          -> ()
      %422 : int = aten::dim(%edge_index.1) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:60:15
      %423 : bool = aten::ne(%422, %41) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:60:15
       = prim::If(%423) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:60:12
        block0():
          %424 : int = aten::dim(%edge_index.1) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:62:42
          %425 : str = aten::format(%24, %424) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:61:33
           = prim::RaiseException(%425, %21) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:61:16
          -> ()
        block1():
          -> ()
      %426 : int = aten::size(%edge_index.1, %43) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:63:15
      %427 : bool = aten::ne(%426, %41) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:63:15
       = prim::If(%427) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:63:12
        block0():
          %428 : int = aten::size(%edge_index.1, %43) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:66:37
          %429 : str = aten::format(%23, %428) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:64:33
           = prim::RaiseException(%429, %21) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:64:16
          -> ()
        block1():
          -> ()
      -> ()
  %the_size.9 : int? = aten::__getitem__(%the_size.7, %43) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:241:19
  %438 : bool = aten::__is__(%the_size.9, %6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:242:11
   = prim::If(%438) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:242:8
    block0():
      %440 : int = aten::size(%364, %self.gnn_node.convs.0.node_dim) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:243:24
      %441 : int?[] = aten::_set_item(%the_size.7, %43, %440) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:243:12
      -> ()
    block1():
      %the_size0.4 : int = prim::unchecked_cast(%the_size.9)
      %444 : int = aten::size(%364, %self.gnn_node.convs.0.node_dim) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:244:25
      %445 : bool = aten::ne(%the_size0.4, %444) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:244:13
       = prim::If(%445) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:244:8
        block0():
          %447 : int = aten::size(%364, %self.gnn_node.convs.0.node_dim) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:246:49
          %449 : str = aten::format(%20, %447, %self.gnn_node.convs.0.node_dim, %the_size0.4) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:246:17
           = prim::RaiseException(%449, %21) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:245:12
          -> ()
        block1():
          -> ()
      -> ()
  %index.4 : Tensor = aten::select(%edge_index.1, %43, %43) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:89:20
  %x_j0.4 : Tensor = aten::index_select(%364, %self.gnn_node.convs.0.node_dim, %index.4) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:90:19
  %edge_index_i.4 : Tensor = aten::select(%edge_index.1, %43, %42) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:141:27
  %455 : int? = aten::__getitem__(%the_size.7, %42) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:160:28
  %456 : bool = aten::__isnot__(%455, %6) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:160:28
  %size_i.8 : int? = prim::If(%456) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:160:17
    block0():
      %size_i.10 : int? = aten::__getitem__(%the_size.7, %42) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:160:17
      -> (%size_i.10)
    block1():
      %size_i.12 : int? = aten::__getitem__(%the_size.7, %43) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:160:53
      -> (%size_i.12)
  %463 : Tensor = aten::add(%x_j0.4, %edge_embedding.4, %42) # /Users/dvlpr/PyCharmProjects/gnn-acceleration-master-thesis/gnn/gin.py:47:22
  %out.4 : Tensor = aten::relu(%463) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:1457:17
  %469 : bool = aten::__isnot__(%6, %6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/aggr/base.py:150:11
  %out0.4 : Tensor = prim::If(%469) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/aggr/base.py:150:8
    block0():
      %ptr0.14 : Tensor = prim::unchecked_cast(%6)
      %472 : int = aten::dim(%out.4) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/aggr/base.py:151:45
      %475 : int = aten::add(%472, %self.gnn_node.convs.0.node_dim) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/aggr/base.py:186:19
      %ptr1.4 : Tensor = prim::Loop(%475, %27, %ptr0.14) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/aggr/base.py:186:4
        block0(%477 : int, %ptr0.16 : Tensor):
          %ptr0.18 : Tensor = aten::unsqueeze(%ptr0.16, %43) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/aggr/base.py:187:14
          -> (%27, %ptr0.18)
      %480 : Tensor = torch_scatter::segment_sum_csr(%out.4, %ptr1.4, %6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_scatter/segment_csr.py:8:11
      -> (%480)
    block1():
      %481 : int = aten::dim(%edge_index_i.4) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:45:11
      %482 : bool = aten::ne(%481, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:45:11
       = prim::If(%482) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:45:8
        block0():
          %483 : int = aten::dim(%edge_index_i.4) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:47:37
          %484 : str = aten::format(%15, %483) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:46:29
           = prim::RaiseException(%484, %21) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:46:12
          -> ()
        block1():
          -> ()
      %487 : int = aten::dim(%out.4) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:49:14
      %dim0.14 : int = aten::add(%487, %self.gnn_node.convs.0.node_dim) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:49:14
      %489 : bool = aten::lt(%dim0.14, %43) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:51:11
      %490 : bool = prim::If(%489) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:51:11
        block0():
          -> (%27)
        block1():
          %491 : int = aten::dim(%out.4) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:51:29
          %492 : bool = aten::ge(%dim0.14, %491) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:51:22
          -> (%492)
       = prim::If(%490) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:51:8
        block0():
          %493 : int = aten::dim(%out.4) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:53:32
          %494 : int = aten::sub(%493, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:53:32
          %495 : str = aten::format(%16, %494, %dim0.14) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:52:29
           = prim::RaiseException(%495, %21) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:52:12
          -> ()
        block1():
          -> ()
      %496 : bool = aten::__is__(%size_i.8, %6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:55:11
      %dim_size0.7 : int = prim::If(%496) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:55:8
        block0():
          %498 : int = aten::numel(%edge_index_i.4) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:56:47
          %499 : bool = aten::gt(%498, %43) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:56:47
          %dim_size1.3 : int = prim::If(%499) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:56:23
            block0():
              %501 : Tensor = aten::max(%edge_index_i.4) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:56:27
              %502 : int = aten::Int(%501) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:56:23
              %503 : int = aten::add(%502, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:56:23
              -> (%503)
            block1():
              -> (%43)
          -> (%dim_size1.3)
        block1():
          %dim_size0.9 : int = prim::unchecked_cast(%size_i.8)
          -> (%dim_size0.9)
      %505 : int[] = aten::size(%out.4) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:68:20
      %size.15 : int[] = aten::list(%505) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:68:15
      %507 : int[] = aten::_set_item(%size.15, %dim0.14, %dim_size0.7) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:69:8
      %509 : int = aten::dim(%out.4) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:19:21
      %size.17 : int[] = aten::mul(%1619, %509) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:19:15
      %511 : int[] = aten::_set_item(%size.17, %dim0.14, %38) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:20:8
      %512 : Tensor = aten::view(%edge_index_i.4, %size.17) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:21:15
      %index0.7 : Tensor = aten::expand_as(%512, %out.4) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:21:15
      %514 : Tensor = aten::new_zeros(%out.4, %size.15, %6, %6, %6, %6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:74:19
      %515 : Tensor = aten::scatter_add_(%514, %dim0.14, %index0.7, %out.4) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:74:19
      -> (%515)
  %516 : Tensor = aten::add(%390, %out0.4, %42) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpai9vfe_r.py:206:25
  %self.gnn_node.convs.1.mlp.0.weight_fused_bn : Float(600, 300, strides=[300, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.convs.1.mlp.0.bias_fused_bn : Float(600, strides=[1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %input0.4 : Tensor = aten::linear(%516, %self.gnn_node.convs.1.mlp.0.weight_fused_bn, %self.gnn_node.convs.1.mlp.0.bias_fused_bn) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/linear.py:114:15
  %523 : int = aten::dim(%input0.4) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:11
  %524 : bool = aten::ne(%523, %41) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:11
  %525 : bool = prim::If(%524) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:11
    block0():
      %527 : bool = aten::ne(%523, %40) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:32
      -> (%527)
    block1():
      -> (%self.gnn_node.convs.0.mlp.1.training)
   = prim::If(%525) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:8
    block0():
      %529 : str = aten::format(%30, %523) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:302:16
       = prim::RaiseException(%529, %21) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:301:12
      -> ()
    block1():
      -> ()
  %input2.4 : Tensor = aten::relu(%input0.4) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:1457:17
  %self.gnn_node.convs.1.mlp.3.weight_fused_bn : Float(300, 600, strides=[600, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.convs.1.mlp.3.bias_fused_bn : Float(300, strides=[1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %h5.1 : Tensor = aten::linear(%input2.4, %self.gnn_node.convs.1.mlp.3.weight_fused_bn, %self.gnn_node.convs.1.mlp.3.bias_fused_bn) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/linear.py:114:15
  %555 : int = aten::dim(%h5.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:11
  %556 : bool = aten::ne(%555, %41) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:11
  %557 : bool = prim::If(%556) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:11
    block0():
      %559 : bool = aten::ne(%555, %40) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:32
      -> (%559)
    block1():
      -> (%self.gnn_node.convs.0.mlp.1.training)
   = prim::If(%557) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:8
    block0():
      %561 : str = aten::format(%30, %555) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:302:16
       = prim::RaiseException(%561, %21) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:301:12
      -> ()
    block1():
      -> ()
  %result.12 : Tensor = aten::relu(%h5.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:1457:17
  %606 : Tensor[] = aten::append(%h_list.1, %result.12) # /Users/dvlpr/PyCharmProjects/gnn-acceleration-master-thesis/gnn/gin.py:161:12
  %607 : Tensor = aten::__getitem__(%h_list.1, %41) # /Users/dvlpr/PyCharmProjects/gnn-acceleration-master-thesis/gnn/gin.py:149:21
  %611 : Tensor = aten::slice(%edge_attr.2, %43, %6, %6, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:49:54
  %612 : Tensor = aten::select(%611, %42, %43) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:49:54
  %614 : Tensor = aten::embedding(%self.gnn_node.convs.2.bond_encoder.bond_embedding_list.0.weight, %612, %38, %self.gnn_node.convs.0.mlp.1.training, %self.gnn_node.convs.0.mlp.1.training) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:2210:11
  %bond_embedding.6 : Tensor = aten::add(%614, %43, %42) # <string>:5:9
  %618 : Tensor = aten::slice(%edge_attr.2, %43, %6, %6, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:50:54
  %619 : Tensor = aten::select(%618, %42, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:50:54
  %621 : Tensor = aten::embedding(%self.gnn_node.convs.2.bond_encoder.bond_embedding_list.1.weight, %619, %38, %self.gnn_node.convs.0.mlp.1.training, %self.gnn_node.convs.0.mlp.1.training) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:2210:11
  %bond_embedding0.6 : Tensor = aten::add_(%bond_embedding.6, %621, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:50:8
  %625 : Tensor = aten::slice(%edge_attr.2, %43, %6, %6, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:51:54
  %626 : Tensor = aten::select(%625, %42, %41) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:51:54
  %628 : Tensor = aten::embedding(%self.gnn_node.convs.2.bond_encoder.bond_embedding_list.2.weight, %626, %38, %self.gnn_node.convs.0.mlp.1.training, %self.gnn_node.convs.0.mlp.1.training) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:2210:11
  %edge_embedding.6 : Tensor = aten::add_(%bond_embedding0.6, %628, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:51:8
  %633 : Tensor = aten::mul(%1635, %607) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:206:25
  %the_size.11 : int?[] = prim::ListConstruct(%6, %6)
  %636 : bool = prim::isinstance[types=[Tensor]](%edge_index.1)
  %637 : bool, %638 : bool = prim::If(%636) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:63:4
    block0():
      %639 : int = prim::layout(%edge_index.1) # :0:0
      %640 : bool = aten::eq(%639, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:64:11
      %641 : bool, %642 : bool = prim::If(%640) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:64:8
        block0():
          -> (%27, %27)
        block1():
          %643 : int = prim::layout(%edge_index.1) # :0:0
          %644 : bool = aten::eq(%643, %41) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:66:11
          %645 : bool, %646 : bool = prim::If(%644) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:66:8
            block0():
              -> (%27, %27)
            block1():
              %647 : int = prim::layout(%edge_index.1) # :0:0
              %648 : bool = aten::eq(%647, %39) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:68:11
              %649 : bool = prim::If(%648) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:68:8
                block0():
                  -> (%27)
                block1():
                  -> (%149)
              -> (%648, %649)
          -> (%645, %646)
      -> (%641, %642)
    block1():
      -> (%self.gnn_node.convs.0.mlp.1.training, %149)
  %650 : bool = prim::If(%637) # :0:0
    block0():
      -> (%638)
    block1():
      -> (%self.gnn_node.convs.0.mlp.1.training)
  %651 : bool = prim::If(%650) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:81:11
    block0():
      -> (%27)
    block1():
      %652 : bool = prim::isinstance[types=[__torch__.torch_sparse.tensor.SparseTensor]](%edge_index.1)
      -> (%652)
   = prim::If(%651) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:42:8
    block0():
      %655 : int = aten::size(%edge_index.1, %42) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:51:26
      %656 : int?[] = aten::_set_item(%the_size.11, %43, %655) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:51:12
      %657 : int = aten::size(%edge_index.1, %43) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:52:26
      %658 : int?[] = aten::_set_item(%the_size.11, %42, %657) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:52:12
      -> ()
    block1():
      %659 : int = prim::dtype(%edge_index.1) # :0:0
      %661 : bool = aten::__contains__(%1613, %659) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:57:15
      %662 : bool = aten::__not__(%661) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:57:15
       = prim::If(%662) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:57:12
        block0():
          %663 : int = prim::dtype(%edge_index.1) # :0:0
          %664 : str = aten::format(%25, %663) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:58:33
           = prim::RaiseException(%664, %21) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:58:16
          -> ()
        block1():
          -> ()
      %665 : int = aten::dim(%edge_index.1) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:60:15
      %666 : bool = aten::ne(%665, %41) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:60:15
       = prim::If(%666) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:60:12
        block0():
          %667 : int = aten::dim(%edge_index.1) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:62:42
          %668 : str = aten::format(%24, %667) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:61:33
           = prim::RaiseException(%668, %21) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:61:16
          -> ()
        block1():
          -> ()
      %669 : int = aten::size(%edge_index.1, %43) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:63:15
      %670 : bool = aten::ne(%669, %41) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:63:15
       = prim::If(%670) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:63:12
        block0():
          %671 : int = aten::size(%edge_index.1, %43) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:66:37
          %672 : str = aten::format(%23, %671) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:64:33
           = prim::RaiseException(%672, %21) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:64:16
          -> ()
        block1():
          -> ()
      -> ()
  %the_size.13 : int? = aten::__getitem__(%the_size.11, %43) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:241:19
  %681 : bool = aten::__is__(%the_size.13, %6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:242:11
   = prim::If(%681) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:242:8
    block0():
      %683 : int = aten::size(%607, %self.gnn_node.convs.0.node_dim) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:243:24
      %684 : int?[] = aten::_set_item(%the_size.11, %43, %683) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:243:12
      -> ()
    block1():
      %the_size0.6 : int = prim::unchecked_cast(%the_size.13)
      %687 : int = aten::size(%607, %self.gnn_node.convs.0.node_dim) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:244:25
      %688 : bool = aten::ne(%the_size0.6, %687) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:244:13
       = prim::If(%688) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:244:8
        block0():
          %690 : int = aten::size(%607, %self.gnn_node.convs.0.node_dim) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:246:49
          %692 : str = aten::format(%20, %690, %self.gnn_node.convs.0.node_dim, %the_size0.6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:246:17
           = prim::RaiseException(%692, %21) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:245:12
          -> ()
        block1():
          -> ()
      -> ()
  %index.6 : Tensor = aten::select(%edge_index.1, %43, %43) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:89:20
  %x_j0.6 : Tensor = aten::index_select(%607, %self.gnn_node.convs.0.node_dim, %index.6) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:90:19
  %edge_index_i.6 : Tensor = aten::select(%edge_index.1, %43, %42) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:141:27
  %698 : int? = aten::__getitem__(%the_size.11, %42) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:160:28
  %699 : bool = aten::__isnot__(%698, %6) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:160:28
  %size_i.14 : int? = prim::If(%699) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:160:17
    block0():
      %size_i.16 : int? = aten::__getitem__(%the_size.11, %42) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:160:17
      -> (%size_i.16)
    block1():
      %size_i.18 : int? = aten::__getitem__(%the_size.11, %43) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:160:53
      -> (%size_i.18)
  %706 : Tensor = aten::add(%x_j0.6, %edge_embedding.6, %42) # /Users/dvlpr/PyCharmProjects/gnn-acceleration-master-thesis/gnn/gin.py:47:22
  %out.6 : Tensor = aten::relu(%706) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:1457:17
  %712 : bool = aten::__isnot__(%6, %6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/aggr/base.py:150:11
  %out0.6 : Tensor = prim::If(%712) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/aggr/base.py:150:8
    block0():
      %ptr0.20 : Tensor = prim::unchecked_cast(%6)
      %715 : int = aten::dim(%out.6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/aggr/base.py:151:45
      %718 : int = aten::add(%715, %self.gnn_node.convs.0.node_dim) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/aggr/base.py:186:19
      %ptr1.6 : Tensor = prim::Loop(%718, %27, %ptr0.20) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/aggr/base.py:186:4
        block0(%720 : int, %ptr0.22 : Tensor):
          %ptr0.24 : Tensor = aten::unsqueeze(%ptr0.22, %43) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/aggr/base.py:187:14
          -> (%27, %ptr0.24)
      %723 : Tensor = torch_scatter::segment_sum_csr(%out.6, %ptr1.6, %6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_scatter/segment_csr.py:8:11
      -> (%723)
    block1():
      %724 : int = aten::dim(%edge_index_i.6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:45:11
      %725 : bool = aten::ne(%724, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:45:11
       = prim::If(%725) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:45:8
        block0():
          %726 : int = aten::dim(%edge_index_i.6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:47:37
          %727 : str = aten::format(%15, %726) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:46:29
           = prim::RaiseException(%727, %21) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:46:12
          -> ()
        block1():
          -> ()
      %730 : int = aten::dim(%out.6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:49:14
      %dim0.18 : int = aten::add(%730, %self.gnn_node.convs.0.node_dim) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:49:14
      %732 : bool = aten::lt(%dim0.18, %43) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:51:11
      %733 : bool = prim::If(%732) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:51:11
        block0():
          -> (%27)
        block1():
          %734 : int = aten::dim(%out.6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:51:29
          %735 : bool = aten::ge(%dim0.18, %734) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:51:22
          -> (%735)
       = prim::If(%733) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:51:8
        block0():
          %736 : int = aten::dim(%out.6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:53:32
          %737 : int = aten::sub(%736, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:53:32
          %738 : str = aten::format(%16, %737, %dim0.18) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:52:29
           = prim::RaiseException(%738, %21) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:52:12
          -> ()
        block1():
          -> ()
      %739 : bool = aten::__is__(%size_i.14, %6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:55:11
      %dim_size0.11 : int = prim::If(%739) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:55:8
        block0():
          %741 : int = aten::numel(%edge_index_i.6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:56:47
          %742 : bool = aten::gt(%741, %43) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:56:47
          %dim_size1.5 : int = prim::If(%742) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:56:23
            block0():
              %744 : Tensor = aten::max(%edge_index_i.6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:56:27
              %745 : int = aten::Int(%744) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:56:23
              %746 : int = aten::add(%745, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:56:23
              -> (%746)
            block1():
              -> (%43)
          -> (%dim_size1.5)
        block1():
          %dim_size0.13 : int = prim::unchecked_cast(%size_i.14)
          -> (%dim_size0.13)
      %748 : int[] = aten::size(%out.6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:68:20
      %size.19 : int[] = aten::list(%748) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:68:15
      %750 : int[] = aten::_set_item(%size.19, %dim0.18, %dim_size0.11) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:69:8
      %752 : int = aten::dim(%out.6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:19:21
      %size.21 : int[] = aten::mul(%1619, %752) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:19:15
      %754 : int[] = aten::_set_item(%size.21, %dim0.18, %38) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:20:8
      %755 : Tensor = aten::view(%edge_index_i.6, %size.21) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:21:15
      %index0.9 : Tensor = aten::expand_as(%755, %out.6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:21:15
      %757 : Tensor = aten::new_zeros(%out.6, %size.19, %6, %6, %6, %6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:74:19
      %758 : Tensor = aten::scatter_add_(%757, %dim0.18, %index0.9, %out.6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:74:19
      -> (%758)
  %759 : Tensor = aten::add(%633, %out0.6, %42) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp7z3h7jnc.py:206:25
  %self.gnn_node.convs.2.mlp.0.weight_fused_bn : Float(600, 300, strides=[300, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.convs.2.mlp.0.bias_fused_bn : Float(600, strides=[1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %input0.6 : Tensor = aten::linear(%759, %self.gnn_node.convs.2.mlp.0.weight_fused_bn, %self.gnn_node.convs.2.mlp.0.bias_fused_bn) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/linear.py:114:15
  %766 : int = aten::dim(%input0.6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:11
  %767 : bool = aten::ne(%766, %41) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:11
  %768 : bool = prim::If(%767) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:11
    block0():
      %770 : bool = aten::ne(%766, %40) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:32
      -> (%770)
    block1():
      -> (%self.gnn_node.convs.0.mlp.1.training)
   = prim::If(%768) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:8
    block0():
      %772 : str = aten::format(%30, %766) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:302:16
       = prim::RaiseException(%772, %21) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:301:12
      -> ()
    block1():
      -> ()
  %input2.6 : Tensor = aten::relu(%input0.6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:1457:17
  %self.gnn_node.convs.2.mlp.3.weight_fused_bn : Float(300, 600, strides=[600, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.convs.2.mlp.3.bias_fused_bn : Float(300, strides=[1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %h11.1 : Tensor = aten::linear(%input2.6, %self.gnn_node.convs.2.mlp.3.weight_fused_bn, %self.gnn_node.convs.2.mlp.3.bias_fused_bn) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/linear.py:114:15
  %798 : int = aten::dim(%h11.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:11
  %799 : bool = aten::ne(%798, %41) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:11
  %800 : bool = prim::If(%799) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:11
    block0():
      %802 : bool = aten::ne(%798, %40) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:32
      -> (%802)
    block1():
      -> (%self.gnn_node.convs.0.mlp.1.training)
   = prim::If(%800) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:8
    block0():
      %804 : str = aten::format(%30, %798) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:302:16
       = prim::RaiseException(%804, %21) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:301:12
      -> ()
    block1():
      -> ()
  %result.18 : Tensor = aten::relu(%h11.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:1457:17
  %849 : Tensor[] = aten::append(%h_list.1, %result.18) # /Users/dvlpr/PyCharmProjects/gnn-acceleration-master-thesis/gnn/gin.py:161:12
  %850 : Tensor = aten::__getitem__(%h_list.1, %40) # /Users/dvlpr/PyCharmProjects/gnn-acceleration-master-thesis/gnn/gin.py:149:21
  %854 : Tensor = aten::slice(%edge_attr.2, %43, %6, %6, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:49:54
  %855 : Tensor = aten::select(%854, %42, %43) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:49:54
  %857 : Tensor = aten::embedding(%self.gnn_node.convs.3.bond_encoder.bond_embedding_list.0.weight, %855, %38, %self.gnn_node.convs.0.mlp.1.training, %self.gnn_node.convs.0.mlp.1.training) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:2210:11
  %bond_embedding.8 : Tensor = aten::add(%857, %43, %42) # <string>:5:9
  %861 : Tensor = aten::slice(%edge_attr.2, %43, %6, %6, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:50:54
  %862 : Tensor = aten::select(%861, %42, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:50:54
  %864 : Tensor = aten::embedding(%self.gnn_node.convs.3.bond_encoder.bond_embedding_list.1.weight, %862, %38, %self.gnn_node.convs.0.mlp.1.training, %self.gnn_node.convs.0.mlp.1.training) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:2210:11
  %bond_embedding0.8 : Tensor = aten::add_(%bond_embedding.8, %864, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:50:8
  %868 : Tensor = aten::slice(%edge_attr.2, %43, %6, %6, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:51:54
  %869 : Tensor = aten::select(%868, %42, %41) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:51:54
  %871 : Tensor = aten::embedding(%self.gnn_node.convs.3.bond_encoder.bond_embedding_list.2.weight, %869, %38, %self.gnn_node.convs.0.mlp.1.training, %self.gnn_node.convs.0.mlp.1.training) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:2210:11
  %edge_embedding.8 : Tensor = aten::add_(%bond_embedding0.8, %871, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:51:8
  %876 : Tensor = aten::mul(%1646, %850) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:206:25
  %the_size.15 : int?[] = prim::ListConstruct(%6, %6)
  %879 : bool = prim::isinstance[types=[Tensor]](%edge_index.1)
  %880 : bool, %881 : bool = prim::If(%879) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:63:4
    block0():
      %882 : int = prim::layout(%edge_index.1) # :0:0
      %883 : bool = aten::eq(%882, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:64:11
      %884 : bool, %885 : bool = prim::If(%883) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:64:8
        block0():
          -> (%27, %27)
        block1():
          %886 : int = prim::layout(%edge_index.1) # :0:0
          %887 : bool = aten::eq(%886, %41) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:66:11
          %888 : bool, %889 : bool = prim::If(%887) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:66:8
            block0():
              -> (%27, %27)
            block1():
              %890 : int = prim::layout(%edge_index.1) # :0:0
              %891 : bool = aten::eq(%890, %39) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:68:11
              %892 : bool = prim::If(%891) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:68:8
                block0():
                  -> (%27)
                block1():
                  -> (%149)
              -> (%891, %892)
          -> (%888, %889)
      -> (%884, %885)
    block1():
      -> (%self.gnn_node.convs.0.mlp.1.training, %149)
  %893 : bool = prim::If(%880) # :0:0
    block0():
      -> (%881)
    block1():
      -> (%self.gnn_node.convs.0.mlp.1.training)
  %894 : bool = prim::If(%893) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:81:11
    block0():
      -> (%27)
    block1():
      %895 : bool = prim::isinstance[types=[__torch__.torch_sparse.tensor.SparseTensor]](%edge_index.1)
      -> (%895)
   = prim::If(%894) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:42:8
    block0():
      %898 : int = aten::size(%edge_index.1, %42) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:51:26
      %899 : int?[] = aten::_set_item(%the_size.15, %43, %898) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:51:12
      %900 : int = aten::size(%edge_index.1, %43) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:52:26
      %901 : int?[] = aten::_set_item(%the_size.15, %42, %900) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:52:12
      -> ()
    block1():
      %902 : int = prim::dtype(%edge_index.1) # :0:0
      %904 : bool = aten::__contains__(%1613, %902) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:57:15
      %905 : bool = aten::__not__(%904) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:57:15
       = prim::If(%905) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:57:12
        block0():
          %906 : int = prim::dtype(%edge_index.1) # :0:0
          %907 : str = aten::format(%25, %906) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:58:33
           = prim::RaiseException(%907, %21) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:58:16
          -> ()
        block1():
          -> ()
      %908 : int = aten::dim(%edge_index.1) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:60:15
      %909 : bool = aten::ne(%908, %41) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:60:15
       = prim::If(%909) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:60:12
        block0():
          %910 : int = aten::dim(%edge_index.1) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:62:42
          %911 : str = aten::format(%24, %910) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:61:33
           = prim::RaiseException(%911, %21) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:61:16
          -> ()
        block1():
          -> ()
      %912 : int = aten::size(%edge_index.1, %43) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:63:15
      %913 : bool = aten::ne(%912, %41) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:63:15
       = prim::If(%913) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:63:12
        block0():
          %914 : int = aten::size(%edge_index.1, %43) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:66:37
          %915 : str = aten::format(%23, %914) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:64:33
           = prim::RaiseException(%915, %21) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:64:16
          -> ()
        block1():
          -> ()
      -> ()
  %the_size.17 : int? = aten::__getitem__(%the_size.15, %43) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:241:19
  %924 : bool = aten::__is__(%the_size.17, %6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:242:11
   = prim::If(%924) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:242:8
    block0():
      %926 : int = aten::size(%850, %self.gnn_node.convs.0.node_dim) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:243:24
      %927 : int?[] = aten::_set_item(%the_size.15, %43, %926) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:243:12
      -> ()
    block1():
      %the_size0.8 : int = prim::unchecked_cast(%the_size.17)
      %930 : int = aten::size(%850, %self.gnn_node.convs.0.node_dim) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:244:25
      %931 : bool = aten::ne(%the_size0.8, %930) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:244:13
       = prim::If(%931) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:244:8
        block0():
          %933 : int = aten::size(%850, %self.gnn_node.convs.0.node_dim) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:246:49
          %935 : str = aten::format(%20, %933, %self.gnn_node.convs.0.node_dim, %the_size0.8) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:246:17
           = prim::RaiseException(%935, %21) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:245:12
          -> ()
        block1():
          -> ()
      -> ()
  %index.8 : Tensor = aten::select(%edge_index.1, %43, %43) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:89:20
  %x_j0.8 : Tensor = aten::index_select(%850, %self.gnn_node.convs.0.node_dim, %index.8) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:90:19
  %edge_index_i.8 : Tensor = aten::select(%edge_index.1, %43, %42) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:141:27
  %941 : int? = aten::__getitem__(%the_size.15, %42) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:160:28
  %942 : bool = aten::__isnot__(%941, %6) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:160:28
  %size_i.20 : int? = prim::If(%942) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:160:17
    block0():
      %size_i.22 : int? = aten::__getitem__(%the_size.15, %42) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:160:17
      -> (%size_i.22)
    block1():
      %size_i.24 : int? = aten::__getitem__(%the_size.15, %43) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:160:53
      -> (%size_i.24)
  %949 : Tensor = aten::add(%x_j0.8, %edge_embedding.8, %42) # /Users/dvlpr/PyCharmProjects/gnn-acceleration-master-thesis/gnn/gin.py:47:22
  %out.8 : Tensor = aten::relu(%949) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:1457:17
  %955 : bool = aten::__isnot__(%6, %6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/aggr/base.py:150:11
  %out0.8 : Tensor = prim::If(%955) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/aggr/base.py:150:8
    block0():
      %ptr0.26 : Tensor = prim::unchecked_cast(%6)
      %958 : int = aten::dim(%out.8) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/aggr/base.py:151:45
      %961 : int = aten::add(%958, %self.gnn_node.convs.0.node_dim) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/aggr/base.py:186:19
      %ptr1.8 : Tensor = prim::Loop(%961, %27, %ptr0.26) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/aggr/base.py:186:4
        block0(%963 : int, %ptr0.28 : Tensor):
          %ptr0.30 : Tensor = aten::unsqueeze(%ptr0.28, %43) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/aggr/base.py:187:14
          -> (%27, %ptr0.30)
      %966 : Tensor = torch_scatter::segment_sum_csr(%out.8, %ptr1.8, %6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_scatter/segment_csr.py:8:11
      -> (%966)
    block1():
      %967 : int = aten::dim(%edge_index_i.8) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:45:11
      %968 : bool = aten::ne(%967, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:45:11
       = prim::If(%968) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:45:8
        block0():
          %969 : int = aten::dim(%edge_index_i.8) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:47:37
          %970 : str = aten::format(%15, %969) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:46:29
           = prim::RaiseException(%970, %21) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:46:12
          -> ()
        block1():
          -> ()
      %973 : int = aten::dim(%out.8) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:49:14
      %dim0.22 : int = aten::add(%973, %self.gnn_node.convs.0.node_dim) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:49:14
      %975 : bool = aten::lt(%dim0.22, %43) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:51:11
      %976 : bool = prim::If(%975) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:51:11
        block0():
          -> (%27)
        block1():
          %977 : int = aten::dim(%out.8) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:51:29
          %978 : bool = aten::ge(%dim0.22, %977) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:51:22
          -> (%978)
       = prim::If(%976) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:51:8
        block0():
          %979 : int = aten::dim(%out.8) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:53:32
          %980 : int = aten::sub(%979, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:53:32
          %981 : str = aten::format(%16, %980, %dim0.22) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:52:29
           = prim::RaiseException(%981, %21) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:52:12
          -> ()
        block1():
          -> ()
      %982 : bool = aten::__is__(%size_i.20, %6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:55:11
      %dim_size0.15 : int = prim::If(%982) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:55:8
        block0():
          %984 : int = aten::numel(%edge_index_i.8) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:56:47
          %985 : bool = aten::gt(%984, %43) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:56:47
          %dim_size1.7 : int = prim::If(%985) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:56:23
            block0():
              %987 : Tensor = aten::max(%edge_index_i.8) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:56:27
              %988 : int = aten::Int(%987) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:56:23
              %989 : int = aten::add(%988, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:56:23
              -> (%989)
            block1():
              -> (%43)
          -> (%dim_size1.7)
        block1():
          %dim_size0.17 : int = prim::unchecked_cast(%size_i.20)
          -> (%dim_size0.17)
      %991 : int[] = aten::size(%out.8) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:68:20
      %size.23 : int[] = aten::list(%991) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:68:15
      %993 : int[] = aten::_set_item(%size.23, %dim0.22, %dim_size0.15) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:69:8
      %995 : int = aten::dim(%out.8) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:19:21
      %size.25 : int[] = aten::mul(%1619, %995) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:19:15
      %997 : int[] = aten::_set_item(%size.25, %dim0.22, %38) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:20:8
      %998 : Tensor = aten::view(%edge_index_i.8, %size.25) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:21:15
      %index0.11 : Tensor = aten::expand_as(%998, %out.8) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:21:15
      %1000 : Tensor = aten::new_zeros(%out.8, %size.23, %6, %6, %6, %6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:74:19
      %1001 : Tensor = aten::scatter_add_(%1000, %dim0.22, %index0.11, %out.8) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:74:19
      -> (%1001)
  %1002 : Tensor = aten::add(%876, %out0.8, %42) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmp5ihy2zn7.py:206:25
  %self.gnn_node.convs.3.mlp.0.weight_fused_bn : Float(600, 300, strides=[300, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.convs.3.mlp.0.bias_fused_bn : Float(600, strides=[1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %input0.8 : Tensor = aten::linear(%1002, %self.gnn_node.convs.3.mlp.0.weight_fused_bn, %self.gnn_node.convs.3.mlp.0.bias_fused_bn) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/linear.py:114:15
  %1009 : int = aten::dim(%input0.8) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:11
  %1010 : bool = aten::ne(%1009, %41) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:11
  %1011 : bool = prim::If(%1010) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:11
    block0():
      %1013 : bool = aten::ne(%1009, %40) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:32
      -> (%1013)
    block1():
      -> (%self.gnn_node.convs.0.mlp.1.training)
   = prim::If(%1011) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:8
    block0():
      %1015 : str = aten::format(%30, %1009) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:302:16
       = prim::RaiseException(%1015, %21) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:301:12
      -> ()
    block1():
      -> ()
  %input2.8 : Tensor = aten::relu(%input0.8) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:1457:17
  %self.gnn_node.convs.3.mlp.3.weight_fused_bn : Float(300, 600, strides=[600, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.convs.3.mlp.3.bias_fused_bn : Float(300, strides=[1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %h17.1 : Tensor = aten::linear(%input2.8, %self.gnn_node.convs.3.mlp.3.weight_fused_bn, %self.gnn_node.convs.3.mlp.3.bias_fused_bn) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/linear.py:114:15
  %1041 : int = aten::dim(%h17.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:11
  %1042 : bool = aten::ne(%1041, %41) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:11
  %1043 : bool = prim::If(%1042) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:11
    block0():
      %1045 : bool = aten::ne(%1041, %40) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:32
      -> (%1045)
    block1():
      -> (%self.gnn_node.convs.0.mlp.1.training)
   = prim::If(%1043) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:8
    block0():
      %1047 : str = aten::format(%30, %1041) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:302:16
       = prim::RaiseException(%1047, %21) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:301:12
      -> ()
    block1():
      -> ()
  %result.24 : Tensor = aten::relu(%h17.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:1457:17
  %1092 : Tensor[] = aten::append(%h_list.1, %result.24) # /Users/dvlpr/PyCharmProjects/gnn-acceleration-master-thesis/gnn/gin.py:161:12
  %1093 : Tensor = aten::__getitem__(%h_list.1, %39) # /Users/dvlpr/PyCharmProjects/gnn-acceleration-master-thesis/gnn/gin.py:149:21
  %1097 : Tensor = aten::slice(%edge_attr.2, %43, %6, %6, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:49:54
  %1098 : Tensor = aten::select(%1097, %42, %43) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:49:54
  %1100 : Tensor = aten::embedding(%self.gnn_node.convs.4.bond_encoder.bond_embedding_list.0.weight, %1098, %38, %self.gnn_node.convs.0.mlp.1.training, %self.gnn_node.convs.0.mlp.1.training) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:2210:11
  %bond_embedding.1 : Tensor = aten::add(%1100, %43, %42) # <string>:5:9
  %1104 : Tensor = aten::slice(%edge_attr.2, %43, %6, %6, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:50:54
  %1105 : Tensor = aten::select(%1104, %42, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:50:54
  %1107 : Tensor = aten::embedding(%self.gnn_node.convs.4.bond_encoder.bond_embedding_list.1.weight, %1105, %38, %self.gnn_node.convs.0.mlp.1.training, %self.gnn_node.convs.0.mlp.1.training) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:2210:11
  %bond_embedding0.1 : Tensor = aten::add_(%bond_embedding.1, %1107, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:50:8
  %1111 : Tensor = aten::slice(%edge_attr.2, %43, %6, %6, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:51:54
  %1112 : Tensor = aten::select(%1111, %42, %41) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:51:54
  %1114 : Tensor = aten::embedding(%self.gnn_node.convs.4.bond_encoder.bond_embedding_list.2.weight, %1112, %38, %self.gnn_node.convs.0.mlp.1.training, %self.gnn_node.convs.0.mlp.1.training) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:2210:11
  %edge_embedding.1 : Tensor = aten::add_(%bond_embedding0.1, %1114, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/ogb/graphproppred/mol_encoder.py:51:8
  %1119 : Tensor = aten::mul(%1657, %1093) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:206:25
  %the_size.2 : int?[] = prim::ListConstruct(%6, %6)
  %1122 : bool = prim::isinstance[types=[Tensor]](%edge_index.1)
  %1123 : bool, %1124 : bool = prim::If(%1122) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:63:4
    block0():
      %1125 : int = prim::layout(%edge_index.1) # :0:0
      %1126 : bool = aten::eq(%1125, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:64:11
      %1127 : bool, %1128 : bool = prim::If(%1126) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:64:8
        block0():
          -> (%27, %27)
        block1():
          %1129 : int = prim::layout(%edge_index.1) # :0:0
          %1130 : bool = aten::eq(%1129, %41) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:66:11
          %1131 : bool, %1132 : bool = prim::If(%1130) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:66:8
            block0():
              -> (%27, %27)
            block1():
              %1133 : int = prim::layout(%edge_index.1) # :0:0
              %1134 : bool = aten::eq(%1133, %39) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:68:11
              %1135 : bool = prim::If(%1134) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:68:8
                block0():
                  -> (%27)
                block1():
                  -> (%149)
              -> (%1134, %1135)
          -> (%1131, %1132)
      -> (%1127, %1128)
    block1():
      -> (%self.gnn_node.convs.0.mlp.1.training, %149)
  %1136 : bool = prim::If(%1123) # :0:0
    block0():
      -> (%1124)
    block1():
      -> (%self.gnn_node.convs.0.mlp.1.training)
  %1137 : bool = prim::If(%1136) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/sparse.py:81:11
    block0():
      -> (%27)
    block1():
      %1138 : bool = prim::isinstance[types=[__torch__.torch_sparse.tensor.SparseTensor]](%edge_index.1)
      -> (%1138)
   = prim::If(%1137) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:42:8
    block0():
      %1141 : int = aten::size(%edge_index.1, %42) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:51:26
      %1142 : int?[] = aten::_set_item(%the_size.2, %43, %1141) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:51:12
      %1143 : int = aten::size(%edge_index.1, %43) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:52:26
      %1144 : int?[] = aten::_set_item(%the_size.2, %42, %1143) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:52:12
      -> ()
    block1():
      %1145 : int = prim::dtype(%edge_index.1) # :0:0
      %1147 : bool = aten::__contains__(%1613, %1145) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:57:15
      %1148 : bool = aten::__not__(%1147) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:57:15
       = prim::If(%1148) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:57:12
        block0():
          %1149 : int = prim::dtype(%edge_index.1) # :0:0
          %1150 : str = aten::format(%25, %1149) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:58:33
           = prim::RaiseException(%1150, %21) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:58:16
          -> ()
        block1():
          -> ()
      %1151 : int = aten::dim(%edge_index.1) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:60:15
      %1152 : bool = aten::ne(%1151, %41) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:60:15
       = prim::If(%1152) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:60:12
        block0():
          %1153 : int = aten::dim(%edge_index.1) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:62:42
          %1154 : str = aten::format(%24, %1153) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:61:33
           = prim::RaiseException(%1154, %21) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:61:16
          -> ()
        block1():
          -> ()
      %1155 : int = aten::size(%edge_index.1, %43) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:63:15
      %1156 : bool = aten::ne(%1155, %41) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:63:15
       = prim::If(%1156) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:63:12
        block0():
          %1157 : int = aten::size(%edge_index.1, %43) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:66:37
          %1158 : str = aten::format(%23, %1157) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:64:33
           = prim::RaiseException(%1158, %21) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:64:16
          -> ()
        block1():
          -> ()
      -> ()
  %the_size.1 : int? = aten::__getitem__(%the_size.2, %43) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:241:19
  %1167 : bool = aten::__is__(%the_size.1, %6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:242:11
   = prim::If(%1167) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:242:8
    block0():
      %1169 : int = aten::size(%1093, %self.gnn_node.convs.0.node_dim) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:243:24
      %1170 : int?[] = aten::_set_item(%the_size.2, %43, %1169) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:243:12
      -> ()
    block1():
      %the_size0.1 : int = prim::unchecked_cast(%the_size.1)
      %1173 : int = aten::size(%1093, %self.gnn_node.convs.0.node_dim) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:244:25
      %1174 : bool = aten::ne(%the_size0.1, %1173) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:244:13
       = prim::If(%1174) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:244:8
        block0():
          %1176 : int = aten::size(%1093, %self.gnn_node.convs.0.node_dim) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:246:49
          %1178 : str = aten::format(%20, %1176, %self.gnn_node.convs.0.node_dim, %the_size0.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:246:17
           = prim::RaiseException(%1178, %21) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/conv/message_passing.py:245:12
          -> ()
        block1():
          -> ()
      -> ()
  %index.1 : Tensor = aten::select(%edge_index.1, %43, %43) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:89:20
  %x_j0.1 : Tensor = aten::index_select(%1093, %self.gnn_node.convs.0.node_dim, %index.1) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:90:19
  %edge_index_i.1 : Tensor = aten::select(%edge_index.1, %43, %42) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:141:27
  %1184 : int? = aten::__getitem__(%the_size.2, %42) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:160:28
  %1185 : bool = aten::__isnot__(%1184, %6) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:160:28
  %size_i : int? = prim::If(%1185) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:160:17
    block0():
      %size_i.1 : int? = aten::__getitem__(%the_size.2, %42) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:160:17
      -> (%size_i.1)
    block1():
      %size_i.3 : int? = aten::__getitem__(%the_size.2, %43) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:160:53
      -> (%size_i.3)
  %1192 : Tensor = aten::add(%x_j0.1, %edge_embedding.1, %42) # /Users/dvlpr/PyCharmProjects/gnn-acceleration-master-thesis/gnn/gin.py:47:22
  %out.3 : Tensor = aten::relu(%1192) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:1457:17
  %1198 : bool = aten::__isnot__(%6, %6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/aggr/base.py:150:11
  %out0.1 : Tensor = prim::If(%1198) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/aggr/base.py:150:8
    block0():
      %ptr0.1 : Tensor = prim::unchecked_cast(%6)
      %1201 : int = aten::dim(%out.3) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/aggr/base.py:151:45
      %1204 : int = aten::add(%1201, %self.gnn_node.convs.0.node_dim) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/aggr/base.py:186:19
      %ptr1.1 : Tensor = prim::Loop(%1204, %27, %ptr0.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/aggr/base.py:186:4
        block0(%1206 : int, %ptr0.9 : Tensor):
          %ptr0.3 : Tensor = aten::unsqueeze(%ptr0.9, %43) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/aggr/base.py:187:14
          -> (%27, %ptr0.3)
      %1209 : Tensor = torch_scatter::segment_sum_csr(%out.3, %ptr1.1, %6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_scatter/segment_csr.py:8:11
      -> (%1209)
    block1():
      %1210 : int = aten::dim(%edge_index_i.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:45:11
      %1211 : bool = aten::ne(%1210, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:45:11
       = prim::If(%1211) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:45:8
        block0():
          %1212 : int = aten::dim(%edge_index_i.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:47:37
          %1213 : str = aten::format(%15, %1212) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:46:29
           = prim::RaiseException(%1213, %21) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:46:12
          -> ()
        block1():
          -> ()
      %1216 : int = aten::dim(%out.3) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:49:14
      %dim0.13 : int = aten::add(%1216, %self.gnn_node.convs.0.node_dim) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:49:14
      %1218 : bool = aten::lt(%dim0.13, %43) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:51:11
      %1219 : bool = prim::If(%1218) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:51:11
        block0():
          -> (%27)
        block1():
          %1220 : int = aten::dim(%out.3) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:51:29
          %1221 : bool = aten::ge(%dim0.13, %1220) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:51:22
          -> (%1221)
       = prim::If(%1219) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:51:8
        block0():
          %1222 : int = aten::dim(%out.3) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:53:32
          %1223 : int = aten::sub(%1222, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:53:32
          %1224 : str = aten::format(%16, %1223, %dim0.13) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:52:29
           = prim::RaiseException(%1224, %21) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:52:12
          -> ()
        block1():
          -> ()
      %1225 : bool = aten::__is__(%size_i, %6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:55:11
      %dim_size0 : int = prim::If(%1225) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:55:8
        block0():
          %1227 : int = aten::numel(%edge_index_i.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:56:47
          %1228 : bool = aten::gt(%1227, %43) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:56:47
          %dim_size1 : int = prim::If(%1228) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:56:23
            block0():
              %1230 : Tensor = aten::max(%edge_index_i.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:56:27
              %1231 : int = aten::Int(%1230) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:56:23
              %1232 : int = aten::add(%1231, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:56:23
              -> (%1232)
            block1():
              -> (%43)
          -> (%dim_size1)
        block1():
          %dim_size0.4 : int = prim::unchecked_cast(%size_i)
          -> (%dim_size0.4)
      %1234 : int[] = aten::size(%out.3) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:68:20
      %size.12 : int[] = aten::list(%1234) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:68:15
      %1236 : int[] = aten::_set_item(%size.12, %dim0.13, %dim_size0) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:69:8
      %1238 : int = aten::dim(%out.3) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:19:21
      %size.4 : int[] = aten::mul(%1619, %1238) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:19:15
      %1240 : int[] = aten::_set_item(%size.4, %dim0.13, %38) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:20:8
      %1241 : Tensor = aten::view(%edge_index_i.1, %size.4) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:21:15
      %index0.4 : Tensor = aten::expand_as(%1241, %out.3) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:21:15
      %1243 : Tensor = aten::new_zeros(%out.3, %size.12, %6, %6, %6, %6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:74:19
      %1244 : Tensor = aten::scatter_add_(%1243, %dim0.13, %index0.4, %out.3) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:74:19
      -> (%1244)
  %1245 : Tensor = aten::add(%1119, %out0.1, %42) # /var/folders/fv/2kfwzrfd34g51h7_rsfb3pb00000gn/T/dvlpr_pyg/tmpruzz5l8h.py:206:25
  %self.gnn_node.convs.4.mlp.0.weight_fused_bn : Float(600, 300, strides=[300, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.convs.4.mlp.0.bias_fused_bn : Float(600, strides=[1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %input0.1 : Tensor = aten::linear(%1245, %self.gnn_node.convs.4.mlp.0.weight_fused_bn, %self.gnn_node.convs.4.mlp.0.bias_fused_bn) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/linear.py:114:15
  %1252 : int = aten::dim(%input0.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:11
  %1253 : bool = aten::ne(%1252, %41) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:11
  %1254 : bool = prim::If(%1253) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:11
    block0():
      %1256 : bool = aten::ne(%1252, %40) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:32
      -> (%1256)
    block1():
      -> (%self.gnn_node.convs.0.mlp.1.training)
   = prim::If(%1254) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:8
    block0():
      %1258 : str = aten::format(%30, %1252) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:302:16
       = prim::RaiseException(%1258, %21) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:301:12
      -> ()
    block1():
      -> ()
  %input2.1 : Tensor = aten::relu(%input0.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/functional.py:1457:17
  %self.gnn_node.convs.4.mlp.3.weight_fused_bn : Float(300, 600, strides=[600, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %self.gnn_node.convs.4.mlp.3.bias_fused_bn : Float(300, strides=[1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %h23.1 : Tensor = aten::linear(%input2.1, %self.gnn_node.convs.4.mlp.3.weight_fused_bn, %self.gnn_node.convs.4.mlp.3.bias_fused_bn) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/linear.py:114:15
  %1284 : int = aten::dim(%h23.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:11
  %1285 : bool = aten::ne(%1284, %41) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:11
  %1286 : bool = prim::If(%1285) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:11
    block0():
      %1288 : bool = aten::ne(%1284, %40) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:32
      -> (%1288)
    block1():
      -> (%self.gnn_node.convs.0.mlp.1.training)
   = prim::If(%1286) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:300:8
    block0():
      %1290 : str = aten::format(%30, %1284) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:302:16
       = prim::RaiseException(%1290, %21) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/batchnorm.py:301:12
      -> ()
    block1():
      -> ()
  %1335 : Tensor[] = aten::append(%h_list.1, %h23.1) # /Users/dvlpr/PyCharmProjects/gnn-acceleration-master-thesis/gnn/gin.py:161:12
  %h_node.1 : Tensor = aten::__getitem__(%h_list.1, %38) # /Users/dvlpr/PyCharmProjects/gnn-acceleration-master-thesis/gnn/gin.py:172:30
  %1347 : int = aten::dim(%h_node.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/pool/glob.py:58:16
  %1348 : bool = aten::eq(%1347, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/pool/glob.py:58:16
  %dim : int = prim::If(%1348) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/pool/glob.py:58:10
    block0():
      -> (%38)
    block1():
      -> (%self.gnn_node.convs.0.node_dim)
  %1359 : Tensor = aten::max(%batch.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/pool/glob.py:62:15
  %1360 : Scalar = aten::item(%1359) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/pool/glob.py:62:15
  %1361 : Scalar = aten::add(%1360, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/pool/glob.py:62:15
  %size0.1 : int = aten::Int(%1361) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/nn/pool/glob.py:62:11
  %1364 : int = aten::dim(%batch.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:45:11
  %1365 : bool = aten::ne(%1364, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:45:11
   = prim::If(%1365) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:45:8
    block0():
      %1366 : int = aten::dim(%batch.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:47:37
      %1367 : str = aten::format(%15, %1366) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:46:29
       = prim::RaiseException(%1367, %21) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:46:12
      -> ()
    block1():
      -> ()
  %1368 : bool = aten::lt(%dim, %43) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:49:33
  %dim0.7 : int = prim::If(%1368) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:49:14
    block0():
      %1370 : int = aten::dim(%h_node.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:49:14
      %dim0.9 : int = aten::add(%1370, %dim) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:49:14
      -> (%dim0.9)
    block1():
      -> (%dim)
  %1372 : bool = aten::lt(%dim0.7, %43) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:51:11
  %1373 : bool = prim::If(%1372) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:51:11
    block0():
      -> (%27)
    block1():
      %1374 : int = aten::dim(%h_node.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:51:29
      %1375 : bool = aten::ge(%dim0.7, %1374) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:51:22
      -> (%1375)
   = prim::If(%1373) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:51:8
    block0():
      %1376 : int = aten::dim(%h_node.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:53:32
      %1377 : int = aten::sub(%1376, %42) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:53:32
      %1378 : str = aten::format(%16, %1377, %dim0.7) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:52:29
       = prim::RaiseException(%1378, %21) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:52:12
      -> ()
    block1():
      -> ()
  %1379 : int[] = aten::size(%h_node.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:68:20
  %size.10 : int[] = aten::list(%1379) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:68:15
  %1381 : int[] = aten::_set_item(%size.10, %dim0.7, %size0.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:69:8
  %1382 : int[] = prim::ListConstruct(%size0.1)
  %count.1 : Tensor = aten::new_zeros(%h_node.1, %1382, %6, %6, %6, %6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:77:20
  %1384 : int = aten::size(%h_node.1, %dim0.7) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:78:54
  %1385 : int[] = prim::ListConstruct(%1384)
  %1386 : Tensor = aten::new_ones(%h_node.1, %1385, %6, %6, %6, %6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:78:41
  %1387 : Tensor = aten::scatter_add_(%count.1, %43, %batch.1, %1386) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:78:12
  %count0.2 : Tensor = aten::clamp(%count.1, %42, %6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:79:20
  %1390 : int = aten::dim(%h_node.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:19:21
  %size.6 : int[] = aten::mul(%1619, %1390) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:19:15
  %1392 : int[] = aten::_set_item(%size.6, %dim0.7, %38) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:20:8
  %1393 : Tensor = aten::view(%batch.1, %size.6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:21:15
  %index1.1 : Tensor = aten::expand_as(%1393, %h_node.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:21:15
  %1395 : Tensor = aten::new_zeros(%h_node.1, %size.10, %6, %6, %6, %6) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:82:18
  %out.1 : Tensor = aten::scatter_add_(%1395, %dim0.7, %index1.1, %h_node.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:82:18
  %1398 : int = aten::dim(%out.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:19:21
  %size.8 : int[] = aten::mul(%1619, %1398) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:19:15
  %1400 : int[] = aten::_set_item(%size.8, %dim0.7, %38) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:20:8
  %1401 : Tensor = aten::view(%count0.2, %size.8) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:21:15
  %1402 : Tensor = aten::expand_as(%1401, %out.1) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:21:15
  %1403 : Tensor = aten::div(%out.1, %1402) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:84:19
  %1406 : Tensor = aten::linear(%1403, %self.graph_pred_linear.weight, %self.graph_pred_linear.bias) # /Users/dvlpr/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/nn/modules/linear.py:114:15
  return (%1406)
"""

## torch._C.parse_ir(graph)

x = torch.tensor(np.array( [[ 5,  0,  4,  5,  3,  0,  2,  0,  0],
                            [ 5,  0,  4,  5,  2,  0,  2,  0,  0],
                            [ 5,  0,  3,  5,  0,  0,  1,  0,  1],
                            [ 7,  0,  2,  6,  0,  0,  1,  0,  1],
                            [28,  0,  4,  2,  0,  0,  5,  0,  1],
                            [ 7,  0,  2,  6,  0,  0,  1,  0,  1],
                            [ 5,  0,  3,  5,  0,  0,  1,  0,  1],
                            [ 5,  0,  4,  5,  2,  0,  2,  0,  0],
                            [ 5,  0,  4,  5,  3,  0,  2,  0,  0],
                            [ 5,  0,  4,  5,  2,  0,  2,  0,  1],
                            [ 7,  0,  2,  6,  0,  0,  1,  0,  1],
                            [ 5,  0,  3,  5,  0,  0,  1,  0,  1],
                            [ 5,  0,  4,  5,  2,  0,  2,  0,  0],
                            [ 5,  0,  4,  5,  3,  0,  2,  0,  0],
                            [ 5,  0,  4,  5,  2,  0,  2,  0,  1],
                            [ 5,  0,  3,  5,  0,  0,  1,  0,  1],
                            [ 5,  0,  4,  5,  2,  0,  2,  0,  0],
                            [ 5,  0,  4,  5,  3,  0,  2,  0,  0],
                            [ 7,  0,  2,  6,  0,  0,  1,  0,  1]]))

edge_index = torch.tensor(np.array([[ 0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,  6,  9,
                                       4, 10, 10, 11, 11, 12, 12, 13, 11, 14, 14, 15, 15, 16, 16, 17, 15, 18,
                                       9,  2, 18,  4],
                                     [ 1,  0,  2,  1,  3,  2,  4,  3,  5,  4,  6,  5,  7,  6,  8,  7,  9,  6,
                                      10,  4, 11, 10, 12, 11, 13, 12, 14, 11, 15, 14, 16, 15, 17, 16, 18, 15,
                                       2,  9,  4, 18]]))

edge_attr = torch.tensor(np.array([[0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    [1, 0, 0],
                                    [1, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    [1, 0, 0],
                                    [1, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    [1, 0, 0],
                                    [1, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    [1, 0, 0],
                                    [1, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0]]))

batch = torch.tensor(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

print(x)
print(edge_index)
print(edge_attr)
print(batch)

## the jit load need the import of the torch geometric Data Loader OR
## the import of both torch scatter and torch sparse, otherwise an error arises
load = torch.jit.load("gin-script.pt")
load.eval()

with torch.no_grad():
    module = torch_mlir.compile(load, (x, edge_index, edge_attr, batch), output_type="linalg-on-tensors")
backend = refbackend.RefBackendLinalgOnTensorsBackend()
compiled = backend.compile(module)
jit_module = backend.load(compiled)