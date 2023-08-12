import numpy as np
import os
import sys

# Get the parent directory's path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

from src.utils import get_velocities_sliding_lid
from src.visualisations import animate_sliding_lid


def perform_checks(viscosity, omega, reynolds_num, wall_velocity, x):
    """
    Perform checks whether the values of omega, viscosity and reynolds number are matching.

    Args:
        viscosity (float): Viscosity.
        omega (float): Relaxation Parameter.
        reynolds_num (int): Reynolds Number.
        wall_velocity (float): Velocity of the moving wall.
        x (int): Grid shape in x direction.

    Returns:
        omega (float): Relaxation Parameter.
    """

    if reynolds_num is not None and viscosity is not None and omega is not None:
        if reynolds_num != x * wall_velocity / viscosity or omega != 1 / (3 * viscosity + 0.5):
            raise ValueError(
                "Reynolds Number, Viscosity and Omega does not match. Please either give Reynolds Number or "
                "Omega or Viscosity")
    elif reynolds_num is not None and viscosity is not None and omega is None:
        if reynolds_num != x * wall_velocity / viscosity:
            raise ValueError(
                "Reynolds Number and Viscosity does not match. Please either give Reynolds Number or "
                "Omega or Viscosity")
        else:
            omega = 1 / (3 * viscosity + 0.5)  # Relaxation parameter
    elif reynolds_num is not None and viscosity is None and omega is not None:
        viscosity_from_re = x * wall_velocity / reynolds_num
        viscosity_from_omega = 1 / 3 * (1 / omega - 0.5)
        if viscosity_from_re != viscosity_from_omega:
            raise ValueError(
                "Reynolds Number and Omega does not match. Please either give Reynolds Number or "
                "Omega or Viscosity")
    elif reynolds_num is not None and viscosity is None and omega is None:
        viscosity = x * wall_velocity / reynolds_num
        omega = 1 / (3 * viscosity + 0.5)
    elif reynolds_num is None and viscosity is not None and omega is not None:
        if omega != 1 / (3 * viscosity + 0.5):
            raise ValueError(
                "Viscosity and Omega does not match. Please either give Reynolds Number or "
                "Omega or Viscosity")
    elif reynolds_num is None and viscosity is not None and omega is None:
        omega = 1 / (3 * viscosity + 0.5)
    elif reynolds_num is None and viscosity is None and omega is None:
        raise ValueError(
            "Please either give Reynolds Number or Omega or Viscosity")
    if omega > 1.7:
        raise ValueError(f'Omega: {omega} greater than 1.7. Give a lower Omega/Viscosity/Reynolds number value.')

    return omega


def main(total_time_steps, x, y, wall_velocity, viscosity=None, omega=None, reynolds_num=None):
    """
    Simulation function for Sliding Lid Parallel Implementation.

    Args:
        total_time_steps (int): Total time steps for simulation.
        x (int): Grid shape in x direction.
        y (int): Grid shape in y direction.
        wall_velocity (float): Velocity of the moving wall.
        viscosity (float, optional): Viscosity.
        omega (float, optional): Relaxation Parameter.
        reynolds_num (int, optional): Reynolds Number.

    Returns:
        None (if visualise=True) or final_velocity (if visualise=False).
    """
    # Initialization
    grid_shape = [x, y]

    # Perform checks and get omega (Relaxation parameter)
    omega = perform_checks(viscosity=viscosity, omega=omega, reynolds_num=reynolds_num, wall_velocity=wall_velocity,
                           x=x)

    density = np.ones((grid_shape[0], grid_shape[1]))
    velocity = np.zeros((2, grid_shape[0], grid_shape[1]))

    # Calculate final_velocity and velocity_fields
    velocity_fields = get_velocities_sliding_lid(grid_size_x=grid_shape[0], grid_size_y=grid_shape[1], omega=omega,
                                                 wall_velocity=wall_velocity, density=density, u=velocity,
                                                 time_steps=total_time_steps, anim=True)

    animate_sliding_lid(grid_shape=grid_shape, velocity_fields=velocity_fields, omega=omega,
                        wall_velocity=wall_velocity)


if __name__ == '__main__':
    time_steps = 10000
    x = 100
    y = 100
    wall_velocity = 0.1
    viscosity = 0.03
    # omega = 1.69
    # re_num = 1000
    main(total_time_steps=time_steps, x=x, y=y, wall_velocity=wall_velocity, viscosity=viscosity)
