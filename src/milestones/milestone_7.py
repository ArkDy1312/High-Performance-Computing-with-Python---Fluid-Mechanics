"""

Sliding Lid Parallel

"""
import numpy as np
from mpi4py import MPI
import os

from src.visualisations import visualise_sliding_lid, visualise_sliding_lid_at_different_time_steps
from src.utils import calculate_pdf_mpi, update_simulation_parameters


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


def main(total_time_steps, x, y, wall_velocity, num_procs_x, num_procs_y, viscosity=None, omega=None, reynolds_num=None,
         num_procs=None, visualise=True, scaling=False):
    """
    Simulation function for Sliding Lid Parallel Implementation.

    Args:
        total_time_steps (int): Total time steps for simulation.
        x (int): Grid shape in x direction.
        y (int): Grid shape in y direction.
        wall_velocity (float): Velocity of the moving wall.
        num_procs_x (int): Number of processes in x direction.
        num_procs_y (int): Number of processes in y direction.
        viscosity (float, optional): Viscosity.
        omega (float, optional): Relaxation Parameter.
        reynolds_num (int, optional): Reynolds Number.
        num_procs (int): Total number of processes.
        visualise (bool, optional): Flag for visualization of Sliding Lid at last time step and for different
        time steps.
        scaling (bool): Check for whether scaling test is being run.

    Returns:
        None (if visualise=True) or final_velocity (if visualise=False).
    """

    grid_shape = [x, y]

    # Checks
    omega = perform_checks(viscosity=viscosity, omega=omega, reynolds_num=reynolds_num, wall_velocity=wall_velocity,
                           x=x)  # Relaxation parameter

    # Initialization
    comm = MPI.COMM_WORLD
    num_procs = comm.Get_size()
    if num_procs_x * num_procs_y != num_procs:
        raise ValueError(f'Number of processes in x direction x Number of processes in y direction != Total number of '
                         f'processes: {num_procs_x} x {num_procs_y} != {num_procs}')
    process_info = update_simulation_parameters(rank=comm.Get_rank(), size=num_procs, num_procs_x=num_procs_x,
                                                num_procs_y=num_procs_y, grid_shape=grid_shape, omega=omega,
                                                steps=total_time_steps, wall_velocity=wall_velocity)

    density = np.ones((process_info.x, process_info.y))
    velocity = np.zeros((2, process_info.x, process_info.y))

    # Time steps for which to show the velocity field plots
    plot_range = np.linspace(total_time_steps / 100, total_time_steps, 6, dtype=int)

    # Calculate final_velocity and velocity_fields
    final_velocity, velocity_fields = calculate_pdf_mpi(process_info=process_info, communicator=comm, density=density,
                                                        u=velocity, time_steps=total_time_steps, plot_range=plot_range)

    # Visualization
    if process_info.rank == 0:
        if not scaling:
            if visualise:
                visualise_sliding_lid(grid_shape=process_info.base_grid, num_steps=total_time_steps,
                                      omega=process_info.relaxation, velocity=final_velocity,
                                      wall_velocity=wall_velocity, mpi=True)
                visualise_sliding_lid_at_different_time_steps(velocity_fields=velocity_fields,
                                                              grid_shape=process_info.base_grid, plot_range=plot_range,
                                                              mpi=True)
                # Saving the final velocity
                velocity_field_output_path = "log/slidingLid_parallel"
                if not os.path.exists(velocity_field_output_path):
                    os.makedirs(velocity_field_output_path)
                np.save(os.path.join(velocity_field_output_path, f"vp.npy"), final_velocity)

            else:
                return final_velocity


if __name__ == '__main__':
    time_steps = 2000
    x = 99
    y = 100
    wall_velocity = 0.1
    # viscosity = 0.03
    # omega = 1.69
    re_num = 200
    num_procs_x = 5
    num_procs_y = 2
    main(total_time_steps=time_steps, x=x, y=y, wall_velocity=wall_velocity, reynolds_num=re_num,
         num_procs_x=num_procs_x, num_procs_y=num_procs_y)
