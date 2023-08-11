from numpy.random import seed, randn
from numpy import float32, float64
from time import process_time, perf_counter
from sys import argv, maxsize
from platform import uname, python_version
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
import torch.utils.benchmark as benchmark
import timeit
import scipy.sparse as sparse
import scipy.stats as stats


def matmul_benchmark():
    seed(1)

    # Print general computer info
    u = uname()
    print('Operating system:', u.system, 'release', u.release + ',',
          '64-bit,' if (maxsize > 2 ** 31 - 1) else '32-bit;',
          'Python version', python_version(),
          '\nCPU:', u.processor)
    try:
        from psutil import cpu_count, virtual_memory, cpu_freq
        print('Cores:',
              cpu_count(logical=False), 'physical,',
              cpu_count(logical=True), 'logical;',
              'RAM: {0:.3f}GB total'.format(virtual_memory().total / 1e9))
        print('Current CPU frequency: {0:.3f}GHz'.format(cpu_freq().current / 1e3))
    except:
        print('(Install psutil to find more details!')
        print(' You may have to do \'sudo apt-get install python-dev\'')
        print(' or similarly before \'pip install psutil\'.)')

    # System specific information
    if u.system == 'Darwin':
        from platform import mac_ver
        # m = mac_ver()
        release, versioninfo, machine = mac_ver()
        print('Release {0},'.format(release),
              'version {0},'.format(versioninfo),
              'machine {0}'.format(machine))

    a_rows = 15
    a_cols = 15
    b_rows = 15
    b_cols = 16

    # Choose precision here (standard is float64, NOT float32!)
    a = randn(a_rows, a_cols).astype(float32)
    b = randn(b_rows, b_cols).astype(float32)

    print('Size of matrix A ({0} x {1}): {1:.0f}MB'.format(a_rows, a_cols, a.nbytes / 1e6))
    print('Size of matrix B ({0} x {1}): {1:.0f}MB'.format(b_rows, b_cols, b.nbytes / 1e6))

    print('Execution of torch.mm(A, B)...')

    # Convert to PyTorch Tensor
    a = torch.Tensor(a)
    b = torch.Tensor(b)

    ## number of torch.mm execution
    num_executions = [2000000, 4000000, 6000000, 8000000, 10000000]
    ## number of runs for each number of execution
    num_runs = 5
    ## num of avg times per run per execution
    tot_times = [[], [], [], [], [], [], [], [], []]
    avg_times = [[], [], [], [], [], [], [], [], []]

    avg_runs_time = []
    min_runs_time = []
    max_runs_time = []
    num_threads = torch.get_num_threads()

    for idx, n in enumerate(num_executions):
        print('>>> Number of executions = {0}'.format(n))
        for i in range(num_runs):
            tot_time = 0
            # for j in range(0, n):
            # Perform measurement, use process_time() since it does not count sleeps
            # t = process_time()
            # c = torch.mm(a, b)
            # t = process_time() - t
            t0 = benchmark.Timer(
                stmt='torch.mm(a, b)',
                setup='from torch import mm',
                globals={'a': a, 'b': b},
                num_threads=num_threads)

            t = t0.timeit(n).times[0]

            # tot_time += t

            # tot_times[idx].append(tot_time)
            # avg_times[idx].append(tot_time / num_executions[idx])
            avg_times[idx].append(t)

        avg_runs_time.append(np.average(avg_times[idx]))
        min_runs_time.append(min(avg_times[idx]))
        max_runs_time.append(max(avg_times[idx]))

        ## printing timing results
        print('> Total execution times in seconds = {0} '.format(tot_times[idx]))
        print('> Average execution times in seconds = {0} '.format(avg_times[idx]))

        print('> Number of runs = {0}'.format(num_runs))
        print('> Min execution runs time in seconds = {0} '.format(min_runs_time[idx]))
        print('> Max execution runs time in seconds = {0} '.format(max_runs_time[idx]))
        print('> Average execution runs time in seconds = {0:.9}'.format(avg_runs_time[idx]))

    w, h = figaspect(1 / 2)
    fig, ax = plt.subplots(figsize=(w, h))
    plt.xlabel('Number of executions')
    plt.ylabel('Average execution time')
    plt.title("PyTorch matmul execution time")

    ax.bar(x=num_executions,  # positions to put the bar to
           height=avg_runs_time,  # height of each bar
           width=1000000,  # width of the bar
           edgecolor='black',  # edgecolor of the bar
           color='orange',  # fill color of the bar
           yerr=np.array([np.subtract(avg_runs_time, min_runs_time), np.subtract(max_runs_time, avg_runs_time)]),  #
           ecolor='red',
           capsize=5)
    # plt.show()
    plt.savefig('torch-mm.pdf')


def matmul_comparison():
    # Torch mul comparison
    num_threads = torch.get_num_threads()
    a_rows = 20
    a_cols = 20
    b_rows = 20
    b_cols = 20

    m = sparse.random(a_rows, a_cols, density=0.1).toarray().astype(float32)
    m = torch.Tensor(m)
    p = sparse.random(b_rows, b_cols, density=0.1).toarray().astype(float32)
    p = torch.Tensor(p)

    m_coo = m.to_sparse_coo()
    p_coo = p.to_sparse_coo()

    print(">> torch.mm")
    t0 = benchmark.Timer(
        stmt='torch.mm(m, p)',
        setup='from torch import mm',
        globals={'m': m, 'p': p},
        num_threads=num_threads)

    print("A, B: dense = {0}".format(t0.timeit(10000000).times[0]))

    t0 = benchmark.Timer(
        stmt='torch.mm(m_coo, p)',
        setup='from torch import mm',
        globals={'m_coo': m_coo, 'p': p},
        num_threads=num_threads)

    print("A: COO, B: dense = {0}".format(t0.timeit(10000000).times[0]))

    t0 = benchmark.Timer(
        stmt='torch.mm(m_coo, p_coo)',
        setup='from torch import mm',
        globals={'m_coo': m_coo, 'p_coo': p_coo},
        num_threads=num_threads)

    print("A, B: COO = {0}".format(t0.timeit(10000000).times[0]))

    print(">> torch.spmm")

    t0 = benchmark.Timer(
        stmt='torch.spmm(m, p)',
        setup='from torch import spmm',
        globals={'m': m, 'p': p},
        num_threads=num_threads)

    print("A, B: dense = {0}".format(t0.timeit(10000000).times[0]))

    t0 = benchmark.Timer(
        stmt='torch.spmm(m_coo, p)',
        setup='from torch import spmm',
        globals={'m_coo': m_coo, 'p': p},
        num_threads=num_threads)

    print("A: COO, B: dense = {0}".format(t0.timeit(10000000).times[0]))

    t0 = benchmark.Timer(
        stmt='torch.spmm(m_coo, p_coo)',
        setup='from torch import spmm',
        globals={'m_coo': m_coo, 'p_coo': p_coo},
        num_threads=num_threads)

    print("A, B: COO = {0}".format(t0.timeit(10000000).times[0]))


    # w, h = figaspect(1 / 2)
    # fig, ax = plt.subplots(figsize=(w, h))
    # plt.xlabel('Function and matrix representation')
    # plt.ylabel('Average execution time')
    # plt.title("PyTorch matmul functions comparison")
    #
    # ax.bar(x=['dense mm', 'COO mm', 'dense spmm', 'COO spmm'],  # positions to put the bar to
    #        height=[torch_mm_dense[0], torch_mm_coo[0], torch_spmm_dense[0], torch_spmm_coo[0]],  # height of each bar
    #        width=0.25,  # width of the bar
    #        edgecolor='black',  # edgecolor of the bar
    #        color='orange',  # fill color of the bar
    #        # yerr=np.array([np.subtract(avg_runs_time, min_runs_time), np.subtract(max_runs_time, avg_runs_time)]),
    #        ecolor='red',
    #        capsize=5)
    # plt.savefig('torch-mm-comparison.pdf')


# For calling from command line
if __name__ == '__main__':
    matmul_benchmark()
    #matmul_comparison()
