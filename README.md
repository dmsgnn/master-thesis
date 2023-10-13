# An FPGA Toolchain for Graph Neural Network Acceleration using High-Level Synthesis

This repository contains all the resources used for the design and evaluation of an MLIR-based FPGA toolchain for Graph Neural Network acceleration using High-Level Synthesis. 
This work represents the Thesis Research of my Master of Science completed with a final grade of 110 cum Laude / 110, during the academic year 2022/23 at the Polytechnic University of Milan.

## Abstract

Graph Neural Networks are a class of Machine Learning models that have emerged as an efficient approach to dealing with graph-structured data, encompassing domains ranging from social networks to molecular chemistry and more.
Particularly in the contemporary era of Big Data, dedicated hardware accelerators are often required to achieve optimal computational performance when processing large amounts of information.
FPGAs offer a promising solution due to their inherent parallelism, but the process of translating Graph Neural Network models into FPGA accelerators is complex and requires extensive knowledge and expertise.
This thesis addresses the challenge of accelerating Graph Neural Networks on FPGA, introducing a comprehensive toolchain that simplifies the process of transitioning from the PyTorch high-level framework to synthesized hardware accelerators leveraging High-Level Synthesis and the MLIR compiler infrastructure.
Torch-MLIR is employed to produce the MLIR representation of the GNN model, which serves as input for the synthesizer.
Here, fine-tuned optimizations can be applied before generating the ultimate GNN accelerator, ready to enhance inference performance on FPGA architectures.
Experimental results demonstrate the efficacy of the toolchain, confirming substantial improvements in both performance and resource utilization.
This accomplishment became possible through the identification of model bottlenecks and a study on optimizing matrix multiplication operations, which resulted to be a critical component of GNN computations.
In conclusion, this thesis represents a significant advancement in the domain of FPGA-accelerated GNN models.
By developing an accessible and versatile toolchain and exploring synthesis optimizations, the research sets the stage for more efficient and widely accessible FPGA-accelerated GNN implementations.

## Design of the Toolchain

<img src="https://github.com/dmsgnn/master-thesis/blob/main/docs/executive_summary/Images/toolchain_modified.svg" width="50%" height="50%">

Firstly, the GNN model must be implemented in PyTorch, one of the most popular and powerful frameworks for Neural Network implementations. Subse- quently, the model is passed as input to Torch-MLIR, a crucial middle step that enables the generation of the MLIR representation. This intermediate representation serves as input for the synthesizer, where, once the frontend optimization is complete, the refined version proceeds to the backend, where the actual GNN accelerator is effectively produced, ready to enhance inference performance on FPGA architectures.


## Experimental results

Different experiments have been performed about matrix multiplication acceleration and GNN acceleration; an extract of them is showed in the following.

All the CPU experiments have been conducted using an Intel Core i9, with 8 cores and a frequency of 2,3 GHz. 
On the other hand, the synthesis experiments targeted an AMD Virtex UltraScale+ (Alveo U280) FPGA.

### Graph Convolutional Network accelerator

The toolchain has been evaluated using a Graph Convolutional Network by applying fine-tuned optimizations to accelerate inference time. 
The figure below shows the result of a comparative analysis between the PyTorch CPU time and the optimized FPGA accelerator time to perform inference.

<img src="https://github.com/dmsgnn/master-thesis/blob/main/docs/presentation/images/GCN_inference_comparison.svg" width="70%" height="70%">

The optimized setting uses two memory channels with on-chip BRAM and one full unroll of the innermost loop.
The results of this final evaluation are incredibly encouraging, showing significant improvement obtained by the optimized accelerator with respect to the PyTorch implementation on CPU.
