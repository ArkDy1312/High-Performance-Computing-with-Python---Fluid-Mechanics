import numpy as np

from src.Lattice_Boltzmann import LatticeBoltzmann
from src.visualisations import visualise_density_velocity as visualise


def main(x, y, omega, num_time_steps):
    """
    Main function to run Lattice Boltzmann simulations.

    Parameters:
        x (int): Width of the simulation domain.
        y (int): Height of the simulation domain.
        omega (float): Relaxation parameter for collision step.
        num_time_steps (int): Number of simulation time steps to run.
    """
    # Test Case 1: Set the density to a slightly higher value at the center, Uniform everywhere else
    lbm = LatticeBoltzmann(x, y, omega=omega)
    # Initialize the PDF with a uniform density
    lbm.density = np.ones((x, y))
    # Set the density to a slightly higher value at the center
    lbm.density[y//2, x//2] = 1.01 * lbm.density[y//2, x//2]

    for step in range(num_time_steps):
        lbm.run_simulation(step=step, collision=True)
        if step % 10 == 0:
            visualise(time_step=step, lbm=lbm, title='Test Case 1')

    # Test 2: Choose an initial distribution of density and velocity at t=0 and run simulation for a long time
    lbm = LatticeBoltzmann(x, y, omega=omega)

    # Choosing uniform densities
    initial_density = np.random.uniform(0, 1, (x, y))
    initial_velocity = np.random.uniform(-0.1, 0.1, (2, x, y))

    lbm.density = initial_density
    lbm.velocity = initial_velocity

    # Running simulation for a long time
    long_time_steps = 1000

    for step in range(long_time_steps):
        lbm.run_simulation(step=step, collision=True)
        if step % 100 == 0:
            visualise(time_step=step, lbm=lbm, title='Test Case 2')


if __name__ == "__main__":
    x = 10
    y = 10
    omega = 0.8
    num_time_steps = 100

    main(x=x, y=y, omega=omega, num_time_steps=num_time_steps)
