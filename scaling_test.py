import os
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser
import subprocess
from tqdm import trange
import time


def get_factors(n):
    factors = []
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            factors.append((i, n // i))
    return factors[-1]  # Return the last pair of factors


def main():
    """
    Run a scaling test for the Sliding Lid simulation using different parameters.
    This script takes various parameters and runs the Sliding Lid simulation with those parameters,
    measuring the MLUPS (Million Lattice Updates Per Second) for each combination of parameters.
    """
    parser = ArgumentParser()
    parser.add_argument('-T', '--total_time_steps', type=int, required=True, help='Total number of time steps.')
    parser.add_argument('-P', '--different_num_processes', type=str, required=True, help='The different number of '
                                                                                         'processes.')
    parser.add_argument('-X', type=str, required=True, help='The different lattice size in the x direction.')
    parser.add_argument('-Y', type=str, required=True, help='The different lattice size in the y direction.')

    args = parser.parse_args()
    x = np.array([int(num) for num in args.X.split(',')])
    y = np.array([int(num) for num in args.Y.split(',')])
    processes = np.array([int(num) for num in args.different_num_processes.split(',')])
    time_steps = args.total_time_steps

    mlups = []

    if x.shape != y.shape:
        raise ValueError('Number of lattice sizes in x direction is not equal to number of lattice sizes in y '
                         'direction. Please provide the correct values.')

    for i in processes:
        # Calculate two factors for the given total number of processes to determine the number of processes in the x
        # and y directions.
        if i == 0:
            raise ValueError("Number of processes cannot be zero.")
        num_procs_x, num_procs_y = get_factors(i)
        for j in trange(x.shape[0]):
            start_time = time.time()
            # Command to run each simulation parallely
            command = f"mpirun -np {i} python src/slidingLid_scaling.py -T {time_steps} -X {x[j]} -Y {y[j]} " \
                      f"-NP {i} -NPX {num_procs_x} -NPY {num_procs_y}"
            subprocess.run(command, shell=True)
            end_time = time.time()
            time_required = end_time - start_time

            values = {'Number of Processes': i, 'Lattice Size': f'{x[j]}x{y[j]}',
                      'Value': x[j] * y[j] * time_steps / time_required / 1e6}
            mlups.append(values)

    output_path = "figures/scaling/slidingLid"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plt.figure(figsize=(10, 8))
    # Separate data for different lattice sizes
    lattice_sizes = set(item['Lattice Size'] for item in mlups)
    for size in lattice_sizes:
        subset = [item for item in mlups if item['Lattice Size'] == size]
        processes = [item['Number of Processes'] for item in subset]
        values = [item['Value'] for item in subset]
        plt.plot(processes, values, marker='o', label=size)

    plt.xlabel('Number of Processes')
    plt.ylabel('MLUPS')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(which='both', linestyle='--', linewidth=0.7)
    plt.legend().set_title('Lattice Size')
    plt.savefig(os.path.join(output_path, 'sliding_lid.jpg'), dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
