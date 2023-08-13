import numpy as np
from src.utils import get_velocities_couette_poiseuille
from src.visualisations import visualise_velocity_evolution_couette_poiseuille, \
    visualise_velocity_vectors_couette_poiseuille


def main(x, y, num_time_steps, wall_velocity, omega, epsilon=0.01):
    """
    Main function to simulate and visualize Couette fluid flow using the lattice Boltzmann method.

    Parameters:
        x (int): Number of cells in the x-direction of the grid.
        y (int): Number of cells in the y-direction of the grid.
        num_time_steps (int): Number of time steps to simulate.
        wall_velocity (float): Velocity of the wall.
        omega (float): Relaxation parameter.
        epsilon (float, optional): Amplitude of the perturbation. Default is 0.01.
    """
    # Initialization
    grid_shape = [x, y]
    density = np.ones((grid_shape[0], grid_shape[1]))
    velocity = np.zeros((2, grid_shape[0], grid_shape[1]))

    # Get final velocity and velocity history using the Lattice Boltzmann method
    final_velocity, velocities = get_velocities_couette_poiseuille(grid_size_x=grid_shape[0],
                                                                   grid_size_y=grid_shape[1], epsilon=epsilon,
                                                                   omega=omega, density=density, u=velocity,
                                                                   time_steps=num_time_steps,
                                                                   wall_velocity=wall_velocity)

    # Visualize the final velocity vectors
    visualise_velocity_vectors_couette_poiseuille(velocity=final_velocity, time_step=num_time_steps,
                                                  wall_velocity=wall_velocity)

    # Visualize the evolution of velocity profiles
    visualise_velocity_evolution_couette_poiseuille(grid_size_y=grid_shape[1], grid_size_x=grid_shape[0],
                                                    wall_velocity=wall_velocity, velocities=velocities,
                                                    epsilon=epsilon, omega=omega)


if __name__ == "__main__":
    x = 100
    y = 100
    epsilon = 0.01  # Amplitude of the perturbation
    omega = 1  # Relaxation parameter
    wall_velocity = 0.1
    num_time_steps = 30000

    main(x=x, y=y, num_time_steps=num_time_steps, omega=omega, wall_velocity=wall_velocity)
