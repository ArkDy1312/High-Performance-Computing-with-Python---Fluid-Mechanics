import numpy as np

from src.utils import get_velocities_couette_poiseuille
from src.visualisations import visualise_velocity_evolution_couette_poiseuille, \
    visualise_velocity_vectors_couette_poiseuille, plot_density_centerline


def main(x, y, omega, num_time_steps, epsilon, density_init=1, test=False):
    """
    Simulate a Poiseuille fluid flow using lattice Boltzmann method and visualize the results.

    Args:
        x (int): Number of grid points in the x-direction.
        y (int): Number of grid points in the y-direction.
        omega (float): Relaxation parameter.
        num_time_steps (int): Number of time steps for the simulation.
        epsilon (float): Amplitude of the perturbation.
        density_init (float, optional): Initial density value. Default is 1.
        test (bool, optional): Whether to run the test case. Default is False.

    Returns:
        None or ndarray: If test=True, returns the final velocity array.
    """
    # Initialization
    grid_shape = [x, y]

    # Parameters
    wall_velocity = 0
    density_in = density_init + epsilon
    density_out = density_init - epsilon

    density = np.ones((grid_shape[0], grid_shape[1]))
    velocity = np.zeros((2, grid_shape[0], grid_shape[1]))

    # Get velocities using the lattice Boltzmann method
    final_density, final_velocity, velocities = get_velocities_couette_poiseuille(grid_size_x=grid_shape[0],
                                                                                  grid_size_y=grid_shape[1],
                                                                                  epsilon=epsilon,
                                                                                  omega=omega, density=density,
                                                                                  u=velocity, time_steps=num_time_steps,
                                                                                  wall_velocity=wall_velocity,
                                                                                  poiseuille=True,
                                                                                  density_in=density_in,
                                                                                  density_out=density_out)

    if test:
        return final_velocity
    else:
        # Visualise Streamplot
        visualise_velocity_vectors_couette_poiseuille(velocity=final_velocity, time_step=num_time_steps,
                                                      wall_velocity=wall_velocity, poiseuille=True, omega=omega,
                                                      epsilon=epsilon)

        # Visualise Velocity Profile Evolution
        visualise_velocity_evolution_couette_poiseuille(grid_size_y=grid_shape[1], grid_size_x=grid_shape[0],
                                                        wall_velocity=wall_velocity, velocities=velocities,
                                                        epsilon=epsilon, omega=omega, poiseuille=True)

        # Plot the density along the centerline of the channel
        plot_density_centerline(final_density, grid_shape[0], grid_shape[1])


if __name__ == '__main__':
    x = 100
    y = 100
    epsilon = 0.001  # Amplitude of the perturbation
    omega = 1  # Relaxation parameter
    num_time_steps = 45000
    density_init = 1

    final_velocity = main(x=x, y=y, num_time_steps=num_time_steps, omega=omega, epsilon=epsilon,
                          density_init=density_init)
