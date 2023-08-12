import numpy as np

from src.Lattice_Boltzmann import LatticeBoltzmann
from src.visualisations import visualise_density_velocity as visualise


# Test
def test_mass_conservation(lb_method):
    """
    Test the mass conservation property of the LBM simulation.

    Args:
        lb_method (LatticeBoltzmann): The LBM simulation object.

    Returns:
        bool: True if mass is conserved, False otherwise.

    """
    mass_before = np.sum(lb_method.compute_density())
    lb_method.streaming()
    mass_after = np.sum(lb_method.compute_density())
    if np.isclose(mass_before, mass_after):
        print('Test Passed!')
    else:
        print('Test Failed! Mass not conserved.')


def main(x, y, num_time_steps, test=False):
    """
    Run the LBM simulation.

    Args:
        x (int): Width of the simulation domain.
        y (int): Height of the simulation domain.
        num_time_steps (int): Number of simulation time steps.
        test (bool, optional): Whether to perform testing. Defaults to False.

    Returns:
        LatticeBoltzmann or None: The LBM simulation object if `test` is True, else None.

    """
    # Initialisation
    pdf = np.zeros((9, y, x))
    pdf[7, :y // 2, :x // 2] = np.ones((y // 2, x // 2))

    # Simulation
    lbm = LatticeBoltzmann(x, y)
    lbm.pdf = pdf

    for step in range(num_time_steps):
        lbm.run_simulation(step=None)
        visualise(time_step=step, lbm=lbm, title='Simple Simulation')

    if test:
        return lbm


if __name__ == "__main__":
    x = 50
    y = 50
    num_time_steps = 10

    lbm = main(x=x, y=y, num_time_steps=num_time_steps, test=True)
    test_mass_conservation(lb_method=lbm)
