from numpy.random import seed, randn
from numpy import float32, float64
from time import process_time, perf_counter
from sys import argv, maxsize
from platform import uname, python_version
import torch


def matmul_benchmark():
    seed(1)

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

    # Convert to PyTorch Tensor and copy to GPU
    a = torch.Tensor(a)
    b = torch.Tensor(b)

    total_time = 0
    iterations = 0
    t0 = 0
    t1 = 0

    # 3 seconds of total time
    while total_time < 3:
        # Perform measurement
        t0 = process_time()
        t1 = perf_counter()
        c = torch.mm(a, b)  # Calls is async so this doesn't really measure much.
        t0 = process_time() - t0
        t1 = perf_counter() - t1
        total_time += t0
        iterations += 1

    # Print timing results
    # The time is cubic in n unless the system implements Strassen-type
    # matrix multiplication (but BLAS implementations usually don't).
    print('Total amount of time: {0:.6f}s'.format(total_time))
    print('Number of iterations: {0}'.format(iterations))
    print('Average execution time: {0:.9f}s = {1:.3f}ns'.format(total_time/iterations, (total_time*10**9)/iterations))

    print('{0:.3f}s process time (total for all cores)'.format(t0))  # process time
    print('{0:.3f}s perf time (elapsed time)'.format(t1))

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

# For calling from command line
if __name__ == '__main__':
    matmul_benchmark()
