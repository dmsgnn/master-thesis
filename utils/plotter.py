import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
import matplotlib.colors


def pytorch_matmul_bench():
    ## number of torch.mm execution
    num_executions = [2000000, 4000000, 6000000, 8000000, 10000000]
    avg_runs_time = [1.57818189e-06, 1.59592361e-06, 1.60637588e-06, 1.60990077e-06, 1.60849089e-06]
    min_runs_time = [1.567656495499989e-06, 1.5850839912500021e-06, 1.6016767878333364e-06, 1.6008430469999979e-06,
                     1.6032683794000036e-06]
    max_runs_time = [1.5947923964999973e-06, 1.602999230249992e-06, 1.614738101499995e-06, 1.6179144736250066e-06,
                     1.6148078792000091e-06]

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
    python_times = [1.608e-06, 1.733e-06, 2.480e-06, 4.554e-06, 4.792e-06, 5.161e-06]
    bambu_times = [96.492e-09, 351.18e-09, 1466.353e-09, 3150.105e-09, 5624.714e-09, 9074.677e-09]
    sizes = ["15x15\n15x16", "30x30\n30x16", "60x60\n60x16", "90x90\n90x16", "120x120\n120x16", "150x150\n150x16"]

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
               s=40,
               label="PyTorch")
    ax.scatter(x=sizes,
               y=bambu_times,
               marker='o',
               c='orange',
               s=40,
               label="Accelerator")
    ax.legend(loc="lower right")
    path = "../docs/thesis/Images/matmul_comparison.pdf"
    plt.savefig(path)
    print("Plot saved in {0}".format(path))


def matmul_optimization15():
    cycles_2channels_NO_BRAM = [29297, 26687, 25457, 26537, 16097, 10097, 7157, 5687]
    cycles_2channels = [25697, 21152, 19457, 22637, 12377, 6257, 3317, 2297]
    cycles_16channels = [29297, 25007, 21857, 24767, 12767, 6527, 3527, 2027]
    cycles_32channels = [29297, 25007, 21857, 24767, 12767, 6407, 3287, 1787]
    unroll_factors = ["baseline", "2", "4", "8", "15", "15+2", "15+4", "15+8"]

    w, h = figaspect(1 / 2)
    fig, ax = plt.subplots(figsize=(w, h))
    plt.xlabel('Unrolling factor')
    plt.ylabel('Execution cycles')
    plt.title("Matmul optimization comparison")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(3, 3))
    ax.scatter(x=unroll_factors,
               y=cycles_2channels_NO_BRAM,
               marker='s',
               c='tab:blue',
               s=40,
               label="2 channels NO_BRAM")
    ax.scatter(x=unroll_factors,
               y=cycles_2channels,
               marker='^',
               c='green',
               s=40,
               label="2 channels ALL_BRAM")
    ax.scatter(x=unroll_factors,
               y=cycles_16channels,
               marker='o',
               c='orange',
               s=40,
               label="16 channels")
    ax.scatter(x=unroll_factors,
               y=cycles_32channels,
               marker='+',
               c='black',
               s=40,
               label="32 channels")
    ax.legend(loc="upper right")
    path_thesis = "../docs/thesis/Images/matmul_comparison15.pdf"
    path_executive = "../docs/executive_summary/Images/matmul_comparison15.pdf"
    plt.savefig(path_thesis)
    plt.savefig(path_executive)
    print("Plot saved in {0} and {1}".format(path_thesis, path_executive))


def matmul_optimization30():
    cycles_2channels_NO_BRAM = [116192, 61892, 38612, 27332, 21692]
    cycles_2channels = [101792, 46562, 23522, 12242, 8402]
    cycles_16channels = [116192, 47672, 24872, 13472, 7772]
    cycles_32channels = [116192, 47132, 23852, 12452, 6752]
    unroll_factors = ["baseline", "30", "30+2", "30+4", "30+8"]

    w, h = figaspect(1 / 2)
    fig, ax = plt.subplots(figsize=(w, h))
    plt.xlabel('Unrolling factor')
    plt.ylabel('Execution cycles')
    plt.title("Matmul optimization comparison")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(3, 3))
    ax.scatter(x=unroll_factors,
               y=cycles_2channels_NO_BRAM,
               marker='s',
               c='tab:blue',
               s=40,
               label="2 channels NO_BRAM")
    ax.scatter(x=unroll_factors,
               y=cycles_2channels,
               marker='^',
               c='green',
               s=40,
               label="2 channels ALL_BRAM")
    ax.scatter(x=unroll_factors,
               y=cycles_16channels,
               marker='o',
               c='orange',
               s=40,
               label="16 channels")
    ax.scatter(x=unroll_factors,
               y=cycles_32channels,
               marker='+',
               c='black',
               s=40,
               label="32 channels")
    ax.legend(loc="upper right")
    path_thesis = "../docs/thesis/Images/matmul_comparison30.pdf"
    path_executive = "../docs/executive_summary/Images/matmul_comparison30.pdf"
    plt.savefig(path_thesis)
    plt.savefig(path_executive)
    print("Plot saved in {0} and {1}".format(path_thesis, path_executive))


def gcn_optimization():
    pytorch_times = [59.25e-06, 66.42e-06, 66.75e-06, 88.88e-06, 98.32e-06, 115.03e-06]
    bambu_times_2ch_1funrll = [523.24e-09, 1.79e-06, 7.13e-06, 14.90e-06, 29.64e-06, 41.12e-06]
    sizes = ["Cora15", "Cora30", "Cora60", "Cora90", "Cora120", "Cora150"]

    w, h = figaspect(1 / 2)
    fig, ax = plt.subplots(figsize=(w, h))
    plt.xlabel('Dataset')
    plt.ylabel('Execution time (s)')
    plt.title("GCN inference time comparison")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(-6, -6))
    ax.scatter(x=sizes,
               y=pytorch_times,
               marker='^',
               c='orange',
               s=40,
               label="PyTorch")
    ax.scatter(x=sizes,
               y=bambu_times_2ch_1funrll,
               marker='o',
               c='tab:blue',
               s=40,
               label="Optimized accelerator")
    ax.legend(loc="upper left")
    path_thesis = "../docs/thesis/Images/gcn_forward_comparison.pdf"
    path_executive = "../docs/executive_summary/Images/gcn_forward_comparison.pdf"
    plt.savefig(path_thesis)
    plt.savefig(path_executive)
    print("Plot saved in {0} and {1}".format(path_thesis, path_executive))

def gcn_cycles_comparison():
    baseline_cycles = [115852, 385874, 1402860, 3051630, 5332200, 8244570]
    optimized_cycles = [93705, 301800, 1064580, 2298510, 3987840, 6136470]
    dataset = ["Cora15", "Cora30", "Cora60", "Cora90", "Cora120", "Cora150"]

    w, h = figaspect(1 / 2)
    fig, ax = plt.subplots(figsize=(w, h))
    plt.xlabel('Dataset')
    plt.ylabel('Number of cycles')
    plt.title("GCN inference cycles comparison")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(6, 6))
    ax.bar(x=dataset,
           height=baseline_cycles,
           width=0.4,
           color='indianred',
           edgecolor='black',
           label="Baseline accelerator")
    ax.bar(x=dataset,
           height=optimized_cycles,
           width=0.4,
           color='cornflowerblue',
           edgecolor='black',
           label="Optimized accelerator")
    ax.legend(loc="upper left")
    path_thesis = "../docs/thesis/Images/gcn_forward_cycles_comparison.pdf"
    path_executive = "../docs/executive_summary/Images/gcn_forward_cycles_comparison.pdf"
    plt.savefig(path_thesis)
    plt.savefig(path_executive)
    print("Plot saved in {0} and {1}".format(path_thesis, path_executive))


if __name__ == '__main__':
    print("Which plot do you want to save?")
    print("   1. pytorch matmul benchmark")
    print("   2. matmul comparison")
    print("   3. matmul optimization [15][15]X[15][16]")
    print("   4. matmul optimization [30][30]X[30][16]")
    print("   5. GCN optimization")
    print("   6. GCN cycles comparison")

    choice = '0'
    choices = {pytorch_matmul_bench: '1', matmul_comparison: '2', matmul_optimization15: '3', matmul_optimization30: '4', gcn_optimization: '5', gcn_cycles_comparison: '6'}

    while choice not in choices.values():
        choice = input()

        if choice not in choices.values():
            print("Please, select one of the options available")

    function = list(choices.keys())[list(choices.values()).index(choice)]
    function()
