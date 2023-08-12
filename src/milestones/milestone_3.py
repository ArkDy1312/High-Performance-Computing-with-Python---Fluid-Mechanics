import numpy as np

from src.utils import get_densities_decay, get_velocities_decay
from src.visualisations import visualise_sinus_density_decay, visualise_decay_different_omegas, \
    visualise_theoretical_vs_measured_viscosity, visualise_evolution_velocity


def test_case_1(grid_size_x, grid_size_y, x_mesh, omega, len_domain_x, num_time_steps=10000, density_init=1,
                eps=0.01):
    """
    Run Test Case 1: Sinus Density Shear Wave Decay.

    Args:
        grid_size_x (int): Size of the grid in the x-direction.
        grid_size_y (int): Size of the grid in the y-direction.
        x_mesh (ndarray): Meshgrid of x-coordinates.
        eps (float): Amplitude of the perturbation.
        omega (float): Relaxation parameter.
        len_domain_x (int): Length of the domain in the x-direction.
        num_time_steps (int): Number of time steps.
        density_init (float): Initial density.

    """

    # Test Case 1: Choose an initial distribution of  𝜌(𝐫,𝑡)  and  𝐮(𝐫,𝑡)  at  𝑡=0  such as  𝜌(𝐫,0)=𝜌0+𝜀sin(2𝜋𝑥/𝐿𝑥)
    # and  𝐮(𝐫,0)=0 . Where  𝐿𝑥  is the length of the domain in the  𝑥  direction.

    density_init = density_init  # Initial density
    density = density_init + eps * np.sin(2.0 * np.pi * x_mesh / len_domain_x)
    max_position = np.argmax(density[0][:])

    # Sinus Density Shear Wave Decay
    visualise_sinus_density_decay(max_position=max_position, density_init=density_init, grid_size_x=grid_size_x,
                                  densities=get_densities_decay(grid_size_x=grid_size_x, grid_size_y=grid_size_y,
                                                                omega=omega, epsilon=eps, time_steps=num_time_steps,
                                                                density=density),
                                  omega=omega, time_steps=num_time_steps, epsilon=eps)

    different_omegas = np.arange(0.2, 2.0, 0.4)
    multiple_densities = np.zeros((different_omegas.shape[0], num_time_steps, grid_size_x))

    for idx, omega in enumerate(different_omegas):
        multiple_densities[idx, ...] = get_densities_decay(grid_size_x=grid_size_x, grid_size_y=grid_size_x,
                                                           omega=omega, epsilon=eps, time_steps=num_time_steps,
                                                           density=density)

    # Decay for different omegas
    visualise_decay_different_omegas(time_steps=num_time_steps, density_init=density_init, epsilon=eps,
                                     grid_size_x=grid_size_x, different_omegas=different_omegas,
                                     multiple_densities=multiple_densities, max_position=max_position)

    # Theoretical vs Measured viscosity for different omegas
    visualise_theoretical_vs_measured_viscosity(different_omegas=different_omegas, epsilon=eps,
                                                grid_size_x=grid_size_x, max_position=max_position,
                                                density_init=density_init, multiple_densities=multiple_densities)


def test_case_2(grid_size_x, grid_size_y, eps, y_mesh, len_domain_y, omega, num_time_steps=6000):
    """
    Run Test Case 2: Sinusoidal variation of velocities.

    Args:
        grid_size_x (int): Size of the grid in the x-direction.
        grid_size_y (int): Size of the grid in the y-direction.
        eps (float): Amplitude of the perturbation.
        y_mesh (ndarray): Meshgrid of y-coordinates.
        len_domain_y (int): Length of the domain in the y-direction.
        omega (float): Relaxation parameter.
        num_time_steps (int): Number of time steps.

    """

    # Test Case 2: Choose an initial distribution of  𝜌(𝐫,0)=1  and  𝑢𝑥(𝐫,0)=𝜀sin(2𝜋𝑦𝐿𝑦) , i.e. a sinusoidal variation
    # of the velocities  𝑢𝑥  with the position  𝑦 . Observe in both cases what happens dynamically and in the long time
    # limit 𝑡→∞ .

    density = np.ones((grid_size_x, grid_size_y))
    u = np.zeros((2, grid_size_x, grid_size_y))
    u[1, :, :] = eps * np.sin(2.0 * np.pi * y_mesh / len_domain_y)

    visualise_evolution_velocity(velocities=get_velocities_decay(epsilon=eps, omega=omega,
                                                                 density=density, time_steps=num_time_steps,
                                                                 grid_size_x=grid_size_x, u=u, grid_size_y=grid_size_y),
                                 time_steps=num_time_steps, grid_size_x=grid_size_x, grid_size_y=grid_size_y,
                                 omega=omega, epsilon=eps)


def main(x, y, lx, ly, omega_init, density_init, epsilon=0.01):
    """
    Main function to run the Shear Wave Decay simulation.

    Args:
        x (int): Size of the grid in the x-direction.
        y (int): Size of the grid in the y-direction.
        lx (int): Length of the domain in the x-direction.
        ly (int): Length of the domain in the y-direction.
        omega_init (float): Initial relaxation parameter.
        density_init (float): Initial density.
        epsilon (float, optional): Amplitude of the perturbation. Default is 0.01.
    """
    # Initialization
    grid_shape = [x, y]
    x_mesh, y_mesh = np.meshgrid(np.arange(grid_shape[0]), np.arange(grid_shape[1]))

    # Test Cases
    test_case_1(grid_size_x=grid_shape[0], grid_size_y=grid_shape[1], x_mesh=x_mesh, len_domain_x=lx,
                eps=epsilon, omega=omega_init, density_init=density_init)
    test_case_2(grid_size_x=grid_shape[0], grid_size_y=grid_shape[1], y_mesh=y_mesh, len_domain_y=ly,
                eps=epsilon, omega=omega_init)


if __name__ == "__main__":
    x = 100
    y = 100
    len_domain = [100, 100]  # Length of domain in both x and y directions
    epsilon = 0.01  # Amplitude of the perturbation
    initial_omega = 1  # Relaxation parameter
    density_init = 0.5

    main(x=x, y=y, lx=len_domain[0], ly=len_domain[1], omega_init=initial_omega, epsilon=epsilon,
         density_init=density_init)
