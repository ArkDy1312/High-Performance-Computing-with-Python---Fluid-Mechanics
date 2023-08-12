import numpy as np
import os
import sys


def main(vs, vp):
    """
    Compare and analyze the results of serial and parallel runs.

    Parameters:
    vs (numpy.ndarray): Array containing velocity field from the serial run.
    vp (numpy.ndarray): Array containing velocity field from the parallel run.
    """
    # Calculate the absolute differences between serial and parallel results
    dif = np.abs(vp - vs)

    # Print results for the serial run
    print('Results of the Serial Run:')
    print(f'Maximum: {vs.max():.5f}, Minimum: {vs.min():.5f}, Absolute Sum: {np.abs(vs).sum():.5f}')

    # Print results for the parallel run
    print('Results of the Parallel Run:')
    print(f'Maximum: {vp.max():.5f}, Minimum: {vp.min():.5f}, Absolute Sum: {np.abs(vp).sum():.5f}')

    # Print results of the absolute differences
    print('Results of Absolute Differences:')
    print(f'Maximum: {dif.max():.5f}, Minimum: {dif.min():.5f}, Absolute Sum: {dif.sum():.5f}')

    # Check if the absolute sum of differences is within a small tolerance
    if dif.sum() < 1e-5:
        print('[SUCCESS] Serial run and parallel run produced nearly identical results.')
    else:
        print('[FAIL] Serial run and parallel run produced different results.')


if __name__ == "__main__":
    # Get the parent directory's path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Add the parent directory to sys.path
    sys.path.insert(0, parent_dir)

    # Load the results from serial and parallel runs
    velocity_field_serial_output_path = "log/slidingLid_serial"
    if not os.path.exists(velocity_field_serial_output_path):
        os.makedirs(velocity_field_serial_output_path)
    velocity_serial = np.load(os.path.join(velocity_field_serial_output_path, 'vs.npy'))  # Serial results

    velocity_field_parallel_output_path = "log/slidingLid_parallel"
    if not os.path.exists(velocity_field_parallel_output_path):
        os.makedirs(velocity_field_parallel_output_path)
    velocity_parallel = np.load(os.path.join(velocity_field_parallel_output_path, 'vp.npy'))  # Parallel results

    main(vs=velocity_serial, vp=velocity_parallel)
