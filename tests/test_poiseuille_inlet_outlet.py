import os
import sys

# Get the parent directory's path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

from src.milestones.milestone_5 import main
from src.utils import compute_area_velocity_profile


# Test Case: Check if Area under the velocity profile at the inlet and Area under the velocity profile in the middle of
# the channel are the same or not
def test(final_velocity, grid_shape):
    """
    Perform a test comparing flow rates at the inlet and centerline of the channel.

    Args:
        final_velocity (ndarray): Final velocity array obtained from the simulation.
        grid_shape (list): List containing the dimensions of the grid [x, y].
    """
    # Extract velocity profile at the inlet and in the middle of the channel
    inlet_velocity_profile = final_velocity[0, :, 0]
    centerline_velocity_profile = final_velocity[0, :, grid_shape[1] // 2]

    # Compute the area under the velocity profiles
    dx = 1.0  # Grid spacing in the x-direction
    inlet_flow_rate = compute_area_velocity_profile(inlet_velocity_profile, dx)
    centerline_flow_rate = compute_area_velocity_profile(centerline_velocity_profile, dx)

    if inlet_flow_rate != centerline_flow_rate:
        print('Test Passed!')
        print(inlet_flow_rate, centerline_flow_rate)
    else:
        print('Test Failed! \nInlet Flow Rate and Centerline Flow Rate should be different')
        print(inlet_flow_rate, centerline_flow_rate)


if __name__ == '__main__':
    x = 100
    y = 50
    epsilon = 0.001  # Amplitude of the perturbation
    omega = 1  # Relaxation parameter
    num_time_steps = 8000
    density_init = 1

    final_velocity = main(x=x, y=y, num_time_steps=num_time_steps, omega=omega, epsilon=epsilon,
                          density_init=density_init, test=True)
    test(final_velocity=final_velocity, grid_shape=[x, y])
