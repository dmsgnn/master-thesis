import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect


def pytorch_matmul_bench():
    ## number of torch.mm execution
    num_executions = [2000000, 4000000, 6000000, 8000000, 10000000]
    avg_runs_time = [1.57818189e-06, 1.59592361e-06, 1.60637588e-06, 1.60990077e-06, 1.60849089e-06]
    min_runs_time = [1.567656495499989e-06, 1.5850839912500021e-06, 1.6016767878333364e-06, 1.6008430469999979e-06, 1.6032683794000036e-06]
    max_runs_time = [1.5947923964999973e-06, 1.602999230249992e-06, 1.614738101499995e-06, 1.6179144736250066e-06, 1.6148078792000091e-06]

    w, h = figaspect(1 / 2)
    fig, ax = plt.subplots(figsize=(w, h))
    plt.xlabel('Number of executions')
    plt.ylabel('Average execution time in seconds')
    plt.title("PyTorch matmul execution time")

    ax.bar(x=num_executions,
           height=avg_runs_time,
           width=600000,
           edgecolor='black',
           color='orange',
           yerr=np.array([np.subtract(avg_runs_time, min_runs_time), np.subtract(max_runs_time, avg_runs_time)]),
           ecolor='red',
           capsize=5)
    path = "./pytorch_matmul_benchmark.pdf"
    plt.savefig(path)
    print("Plot saved in {0}".format(path))


def matmul_comparison():
    python_times = [1.608e-06, 1.733e-06, 2.480e-06, 4.554e-06, 5.161e-06]
    bambu_times = [96.492e-09, 351.18e-09, 1466.353e-09, 3150.105e-09, 9074.677e-09]
    sizes = ["15x15\n15x16", "30x30\n30x16", "60x60\n60x16", "90x90\n90x16", "150x150\n150x16"]

    w, h = figaspect(1 / 2)
    fig, ax = plt.subplots(figsize=(w, h))
    plt.xlabel('Size of input matrices')
    plt.ylabel('Execution time in seconds')
    plt.title("PyTorch and baseline Accelerator matmul comparison")
    plt.subplots_adjust(bottom=0.15)
    ax.scatter(x=sizes,
               y=python_times,
               marker='^',
               c='green',
               s=36,
               label="PyTorch")
    ax.scatter(x=sizes,
               y=bambu_times,
               marker='o',
               c='orange',
               s=36,
               label="Accelerator")
    ax.legend(loc="lower right")
    path = "../docs/thesis/Images/matmul_comparison.pdf"
    plt.savefig(path)
    print("Plot saved in {0}".format(path))


def matmul_optimization():
    print("opt")


def gcn_optimization():
    print("gcn")


if __name__ == '__main__':
    print("Which plot do you want to save?")
    print("   1. pytorch matmul benchmark")
    print("   2. matmul comparison")
    print("   3. matmul optimization")
    print("   4. GCN optimization")

    choice = '0'
    choices = {pytorch_matmul_bench: '1', matmul_comparison: '2', matmul_optimization: '3', gcn_optimization: '4'}

    while choice not in choices.values():
        choice = input()

        if choice not in choices.values():
            print("Please, select one of the options available")

    function = list(choices.keys())[list(choices.values()).index(choice)]
    function()
