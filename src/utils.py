import numpy as np
from tqdm import trange
from dataclasses import dataclass, field

from src.Lattice_Boltzmann import LatticeBoltzmann


def analytical_sinus_decay(time_steps, positions, grid_size, omega, epsilon):
    """
    Calculates the analytical sinusoidal decay for a given set of time steps and positions.

    Args:
        time_steps (np.ndarray): Array of time steps.
        positions (int or ndarray): Array of positions.
        grid_size (int): The size of the grid.
        omega: Relaxation time parameter.
        epsilon: Small parameter used for normalization.

    Returns:
        float: The analytical sinusoidal decay value.
    """
    viscosity = 1 / 3 * (1 / omega - 0.5)
    domain_size = 2.0 * np.pi / grid_size
    return epsilon * np.exp(- viscosity * domain_size ** 2 * time_steps) * np.sin(domain_size * positions)


def get_densities_decay(grid_size_x, grid_size_y, omega, epsilon, time_steps, density):
    """
    Runs the LBM simulation to obtain density values at different time steps for a sinusoidal wave decay.

    Args:
        grid_size_x (int): The size of the grid in the x-dimension.
        grid_size_y (int): The size of the grid in the y-dimension.
        omega: Relaxation time parameter.
        epsilon: Small parameter used for normalization.
        time_steps (int): The total number of time steps.
        density: Initial density value.

    Returns:
        ndarray: Array of density values at different time steps.
    """
    lbm = LatticeBoltzmann(grid_size_x, grid_size_y, omega=omega, epsilon=epsilon)
    densities = np.zeros((time_steps, grid_size_y))

    lbm.density = density

    for step in trange(time_steps):
        lbm.run_simulation(step=step, collision=True)
        densities[step] = lbm.density[0]

    return densities


def get_velocities_decay(grid_size_x, grid_size_y, omega, epsilon, time_steps, density, u):
    """
    Runs the LBM simulation to obtain velocity profiles at different time steps for a sinusoidal wave decay.

    Args:
        grid_size_x (int): The size of the grid in the x-dimension.
        grid_size_y (int): The size of the grid in the y-dimension.
        omega: Relaxation time parameter.
        epsilon: Small parameter used for normalization.
        time_steps (int): The total number of time steps.
        density: Initial density value.
        u: Initial velocity value.

    Returns:
        ndarray: Array of velocity profiles at different time steps.
    """
    lbm = LatticeBoltzmann(grid_size_x, grid_size_y, omega=omega, epsilon=epsilon)
    velocities = np.zeros((time_steps, grid_size_y))

    lbm.density = density
    lbm.velocity = u

    for step in trange(time_steps):
        lbm.run_simulation(step=step, collision=True)
        velocities[step] = lbm.velocity[1, :, grid_size_y//2]

    return velocities


def get_velocities_couette_poiseuille(grid_size_x, grid_size_y, omega, epsilon, time_steps, density, u, wall_velocity,
                                      poiseuille=False, density_in=None, density_out=None):
    """
    Runs the LBM simulation to obtain velocity profiles for Couette or Poiseuille flow.

    Args:
        grid_size_x (int): The size of the grid in the x-dimension.
        grid_size_y (int): The size of the grid in the y-dimension.
        omega: Relaxation time parameter.
        epsilon: Small parameter used for normalization.
        time_steps (int): The total number of time steps.
        density: Initial density value.
        u: Initial velocity value.
        wall_velocity: The velocity of the wall in Couette flow.
        poiseuille (bool): Flag indicating whether to simulate Couette or Poiseuille flow.
        density_in: Density at the inlet for Poiseuille flow.
        density_out: Density at the outlet for Poiseuille flow.

    Returns:
        tuple: Array of velocity profiles at different time intervals and corresponding time intervals.
    """
    lbm = LatticeBoltzmann(grid_size_x, grid_size_y, omega=omega, epsilon=epsilon)
    velocities = []
    periodic_boundary = poiseuille

    lbm.density = density
    lbm.velocity = u
    lbm.wall_velocity = wall_velocity
    if poiseuille:
        lbm.density_in = density_in
        lbm.density_out = density_out

    for step in trange(time_steps):
        lbm.run_simulation(step=step, collision=True, boundary_conditions=True, periodic_boundary=periodic_boundary)
        velocities.append(lbm.velocity[0, grid_size_x // 2, :])

    final_velocity = lbm.velocity
    final_density = lbm.density

    if poiseuille:
        return final_density, final_velocity, velocities
    else:
        return final_velocity, velocities


def compute_area_velocity_profile(velocity_profile, dx):
    """
    Compute the area under the velocity profile using numerical integration.

    Args:
        velocity_profile (ndarray): Velocity profile along the centerline of the channel.
        dx (float): Grid spacing in the x-direction.

    Returns:
        float: Area under the velocity profile.
    """
    area = np.trapz(velocity_profile, dx=dx)
    return area


def get_velocities_sliding_lid(grid_size_x, grid_size_y, omega, wall_velocity, density, u, time_steps, plot_range=None,
                               anim=False):
    """
    Runs the LBM simulation to obtain velocity profiles for a sliding lid simulation.

    Args:
        grid_size_x (int): The size of the grid in the x-dimension.
        grid_size_y (int): The size of the grid in the y-dimension.
        omega: Relaxation time parameter.
        wall_velocity: The velocity of the wall in sliding lid simulation.
        density (ndarray): Initial density value.
        u (ndarray): Initial velocity value.
        time_steps (int): The total number of time steps.
        plot_range (ndarray): Time steps for which to show the velocity field plots.
        anim (bool): Create animation of the flow field.

    Returns:
        ndarray: Array of velocity values at different time steps.
    """
    lbm = LatticeBoltzmann(grid_size_x, grid_size_y, omega=omega, s_lid=True)

    lbm.density = density
    lbm.velocity = u
    lbm.wall_velocity = wall_velocity
    velocity_fields = []
    if anim:
        all_velocity_fields = []

    for step in trange(time_steps):
        lbm.run_simulation(step=step, collision=True, boundary_conditions=True)

        if plot_range is not None:
            if step in plot_range-1:
                velocity_fields.append(lbm.velocity)
        if anim:
            all_velocity_fields.append(lbm.velocity)

    if anim:
        return all_velocity_fields
    else:
        return lbm.velocity[:, :, :], velocity_fields


'''

For MPI
    
'''


@dataclass
class BoundaryFlags:
    """
    Flags indicating the application of boundaries.

    Attributes:
        apply_left: Flag indicating whether the left boundary is applied.
        apply_right: Flag indicating whether the right boundary is applied.
        apply_top: Flag indicating whether the top boundary is applied.
        apply_bottom: Flag indicating whether the bottom boundary is applied.
    """
    apply_left: bool = False
    apply_right: bool = False
    apply_top: bool = False
    apply_bottom: bool = False


@dataclass
class NeighborIndices:
    """
    Indices of neighboring cells.

    Attributes:
        left: Index of the left neighbor.
        right: Index of the right neighbor.
        top: Index of the top neighbor.
        bottom: Index of the bottom neighbor.
    """
    left: int = -1
    right: int = -1
    top: int = -1
    bottom: int = -1


@dataclass
class MpiSimulationParameters:
    """
    Contains information related to MPI communication and grid properties for the simulation.

    Attributes:
        boundary_flags: Object containing information about applied boundaries.
        neighbors: Object containing indices of neighboring cells.
        x: Size of the grid in the x-direction.
        y: Size of the grid in the y-direction.
        pos_x: X-coordinate position of the process in the grid.
        pos_y: Y-coordinate position of the process in the grid.
        relaxation: Relaxation parameter for the simulation (omega).
        base_grid: Size of the base grid.
        steps: Number of simulation steps.
        wall_velocity: Parameter for sliding lid simulation.
        rank: Rank of the process.
        size: Total number of processes.
        num_procs_x: Number of processes in x direction.
        num_procs_y: Number of processes in y direction.
    """
    boundary_flags: 'BoundaryFlags' = field(default_factory=lambda: BoundaryFlags())
    neighbors: 'NeighborIndices' = field(default_factory=lambda: NeighborIndices())
    x: int = -1
    y: int = -1
    pos_x: int = -1
    pos_y: int = -1
    relaxation: int = -1
    base_grid: list = -1
    steps: int = -1
    wall_velocity: int = -1
    rank: int = -1
    size: int = -1
    num_procs_x: int = None
    num_procs_y: int = None


def calculate_coordinates(rank, num_procs_x):
    """
    Calculate the x and y coordinates of a process (rank) in a quadratic grid.

    Args:
        rank (int): The rank of the process.
        num_procs_x (int): Number of processes in x direction.

    Returns:
        tuple: The x and y coordinates of the process.
    """
    pos_x = rank % num_procs_x
    pos_y = rank // num_procs_x

    return pos_x, pos_y


def apply_boundary_info(pos_x, pos_y, num_procs_x, num_procs_y):
    """
    Set boundary information based on process position.

    Args:
        pos_x (int): The x-coordinate of the process.
        pos_y (int): The y-coordinate of the process.
        num_procs_x (int): Number of processes in x direction.
        num_procs_y (int): Number of processes in y direction.

    Returns:
        BoundaryFlags: An object containing the boundary information.
    """
    boundaries_info = BoundaryFlags()
    if pos_x == 0:
        boundaries_info.apply_left = True
    if pos_y == 0:
        boundaries_info.apply_bottom = True
    if pos_x == num_procs_x:
        boundaries_info.apply_right = True
    if pos_y == num_procs_y:
        boundaries_info.apply_top = True
    return boundaries_info


def determine_neighbors(rank, pos_x, pos_y, num_procs_x, num_procs_y):
    """
    Determine the indices of neighboring cells for a given rank.

    Args:
        rank (int): Rank of the process.
        pos_x (int): x position of process(rank) in the rank grid.
        pos_y (int): y position of process(rank) in the rank grid.
        num_procs_x (int): Number of processes in x direction.
        num_procs_y (int): Number of processes in y direction.

    Returns:
        CellNeighbors: Object containing the indices of neighboring cells.
    """
    neighbors = NeighborIndices()
    neighbors.top = rank + num_procs_x
    neighbors.bottom = rank - num_procs_x
    neighbors.right = rank + 1
    neighbors.left = rank - 1

    # If there is an actual boundary in that direction then neghbor = -1
    if pos_x == 0:
        neighbors.left = -1
    if pos_x == num_procs_x-1:
        neighbors.right = -1
    if pos_y == 0:
        neighbors.bottom = -1
    if pos_y == num_procs_y-1:
        neighbors.top = -1

    return neighbors


def calculate_local_shape(neighbors, initial_x, initial_y, pos_x, pos_y, num_procs_x, num_procs_y, grid_shape):
    """
    Calculate the local shape of the grid for the current MPI process, accounting for ghost cells.

    Args:
        neighbors (NeighborIndices): Object containing indices of neighboring cells.
        initial_x (int): Initial number of rows for the current process.
        initial_y (int): Initial number of columns for the current process.
        pos_x (int): x-coordinate position of the process in the grid.
        pos_y (int): y-coordinate position of the process in the grid.
        num_procs_x (int): Number of processes in x direction.
        num_procs_y (int): Number of processes in y direction.
        grid_shape (tuple): Base grid shape.

    Returns:
        tuple: Tuple containing updated local grid shape (x, y) and initial shape of the grid before adding the ghost
        cells.
    """
    # If rows/columns are not divisible by num of processes defined by user
    if grid_shape[0] % num_procs_x != 0 and pos_x == num_procs_x - 1:
        initial_x += grid_shape[0] % num_procs_x
    if grid_shape[1] % num_procs_y != 0 and pos_y == num_procs_y - 1:
        initial_y += grid_shape[1] % num_procs_y

    # Adding extra rows/columns only for ghost cells and not for actual boundaries
    x = initial_x
    y = initial_y
    if neighbors.left != -1 or neighbors.right != -1:
        x += 1
    if neighbors.left != -1 and neighbors.right != -1:
        x += 1
    if neighbors.top != -1 or neighbors.bottom != -1:
        y += 1
    if neighbors.top != -1 and neighbors.bottom != -1:
        y += 1

    return x, y, initial_x, initial_y


def update_simulation_parameters(rank, size, num_procs_x, num_procs_y, grid_shape, omega, steps, wall_velocity):
    """
    Update the MPI simulation parameters with relevant information.

    Args:
        rank: Rank of the process.
        size: Total number of processes.
        num_procs_x: Number of processes in x direction.
        num_procs_y: Number of processes in y direction.
        grid_shape: Base grid shape.
        omega: Relaxation factor (omega).
        steps: Number of simulation steps.
        wall_velocity: Wall velocity value.

    Returns:
        MpiSimulationParameters: Updated MPI simulation parameters.
    """
    # Create an instance of BoundaryFlags
    info = BoundaryFlags()

    # Fill the attributes of MpiPackageStructure
    info.rank = rank
    info.size = size
    info.pos_x, info.pos_y = calculate_coordinates(rank=rank, num_procs_x=num_procs_x)
    info.boundaries_info = apply_boundary_info(pos_x=info.pos_x, pos_y=info.pos_y, num_procs_x=num_procs_x - 1,
                                               num_procs_y=num_procs_y - 1)
    info.initial_x = grid_shape[0] // num_procs_x
    info.initial_y = grid_shape[1] // num_procs_y
    info.neighbors = determine_neighbors(rank=rank, pos_x=info.pos_x, pos_y=info.pos_y,
                                         num_procs_x=num_procs_x, num_procs_y=num_procs_y)
    info.relaxation = omega
    info.base_grid = grid_shape
    info.steps = steps
    info.wall_velocity = wall_velocity
    info.x, info.y, info.initial_x, info.initial_y = \
        calculate_local_shape(neighbors=info.neighbors, initial_x=info.initial_x, initial_y=info.initial_y,
                              pos_x=info.pos_x, pos_y=info.pos_y, num_procs_x=num_procs_x, num_procs_y=num_procs_y,
                              grid_shape=grid_shape)
    info.num_procs_x = num_procs_x
    info.num_procs_y = num_procs_y

    return info


def get_required_indices(rank, process_info):
    """
    Calculate the required indices for slicing the local grid based on the rank and process information.

    Args:
        rank (int): Rank of the process.
        process_info (MpiSimulationParameters): Information about the MPI process and simulation.

    Returns:
        tuple: Initial x and y size for the current rank, start and end indices for slicing the full grid.
    """
    # Calculate the domain size for rank
    pos_x, pos_y = calculate_coordinates(rank=rank, num_procs_x=process_info.num_procs_x)

    # Calculate the initial size for the current rank based on the base grid size and number of processes
    initial_x_curr_rank = process_info.base_grid[0] // process_info.num_procs_x
    initial_y_curr_rank = process_info.base_grid[1] // process_info.num_procs_y

    # Adjust initial size if base grid size is not evenly divisible by the number of processes
    if process_info.base_grid[0] % process_info.num_procs_x != 0 and pos_x == process_info.num_procs_x - 1:
        initial_x_curr_rank += process_info.base_grid[0] % process_info.num_procs_x
    if process_info.base_grid[1] % process_info.num_procs_y != 0 and pos_y == process_info.num_procs_y - 1:
        initial_y_curr_rank += process_info.base_grid[1] % process_info.num_procs_y

    # Calculate Start and end indices for the full grid
    start_x = 0 + process_info.initial_x * pos_x
    end_x = process_info.initial_x + initial_x_curr_rank * pos_x
    start_y = 0 + process_info.initial_y * pos_y
    end_y = process_info.initial_y + initial_y_curr_rank * pos_y

    return initial_x_curr_rank, initial_y_curr_rank, start_x, end_x, start_y, end_y


def calculate_local_shape_without_padding(neighbors, partial_grid):
    """
    Calculate the local shape of the grid without including padding due to ghost cells.

    Args:
        neighbors (NeighborIndices): Indices of neighboring cells.
        partial_grid (ndarray): Partial grid with potential padding.

    Returns:
        ndarray: Local shape of the grid without padding from ghost cells.
    """
    if neighbors.left != -1:
        partial_grid = partial_grid[:, 1:, :]
    if neighbors.right != -1:
        partial_grid = partial_grid[:, :-1, :]
    if neighbors.top != -1:
        partial_grid = partial_grid[:, :, :-1]
    if neighbors.bottom != -1:
        partial_grid = partial_grid[:, :, 1:]

    return partial_grid


def perform_communication(grid, info, comm):
    """
    Perform communication between neighboring cells to exchange boundary data.

    Args:
        grid: The grid containing distribution functions.
        info: Information about the MPI process.
        comm: MPI communicator.

    Returns:
        ndarray: The updated grid after exchanging boundary data.
    """

    # Communication for the right boundary
    if not info.boundaries_info.apply_right:
        # Create a receive buffer for the right boundary
        recv_buf = grid[:, -1, :].copy()
        # Send the rightmost column to the neighboring process on the right
        # and receive the column from the neighboring process
        comm.Sendrecv(grid[:, -2, :].copy(), info.neighbors.right, recvbuf=recv_buf, sendtag=20, recvtag=21)
        # Update the rightmost column with the received values
        grid[:, -1, :] = recv_buf

    # Communication for the left boundary
    if not info.boundaries_info.apply_left:
        # Create a receive buffer for the left boundary
        recv_buf = grid[:, 0, :].copy()
        # Send the leftmost column to the neighboring process on the left
        # and receive the column from the neighboring process
        comm.Sendrecv(grid[:, 1, :].copy(), info.neighbors.left, recvbuf=recv_buf, sendtag=21, recvtag=20)
        # Update the leftmost column with the received values
        grid[:, 0, :] = recv_buf

    # Communication for the bottom boundary
    if not info.boundaries_info.apply_bottom:
        # Create a receive buffer for the bottom boundary
        recv_buf = grid[:, :, 0].copy()
        # Send the bottommost layer to the neighboring process at the bottom
        # and receive the layer from the neighboring process
        comm.Sendrecv(grid[:, :, 1].copy(), info.neighbors.bottom, recvbuf=recv_buf, sendtag=51, recvtag=50)
        # Update the bottommost layer with the received values
        grid[:, :, 0] = recv_buf

    # Communication for the top boundary
    if not info.boundaries_info.apply_top:
        # Create a receive buffer for the top boundary
        recv_buf = grid[:, :, -1].copy()
        # Send the topmost layer to the neighboring process at the top
        # and receive the layer from the neighboring process
        comm.Sendrecv(grid[:, :, -2].copy(), info.neighbors.top, recvbuf=recv_buf, sendtag=50, recvtag=51)
        # Update the topmost layer with the received values
        grid[:, :, -1] = recv_buf

    return grid


def combine_data(process_info, grid, comm):
    """
    Collect data from all processes and combine them into a full grid.

    Args:
        process_info: Information about the MPI process.
        grid: The grid containing density functions.
        comm: MPI communicator.

    Returns:
        numpy.ndarray: The full grid containing density functions.
    """
    full_grid = np.zeros(9)

    if process_info.rank == 0:
        # Create a full grid with the correct size
        full_grid = np.ones((9, process_info.base_grid[0], process_info.base_grid[1]))
        # Copy the local grid data to the corresponding region in the full grid
        full_grid[:, 0:process_info.initial_x, 0:process_info.initial_y] = \
            calculate_local_shape_without_padding(neighbors=process_info.neighbors, partial_grid=grid)
        # Receive data from other processes and copy them to the appropriate region in the full grid
        for i in range(1, process_info.size):
            initial_x_curr_rank, initial_y_curr_rank, start_x, end_x, start_y, end_y = \
                get_required_indices(rank=i, process_info=process_info)
            tmp_grid = np.zeros((9, initial_x_curr_rank, initial_y_curr_rank))
            comm.Recv(tmp_grid, source=i, tag=i)
            full_grid[:, start_x:end_x, start_y:end_y] = tmp_grid
    else:
        # Send local grid data to the root process (rank 0)
        partial_grid = calculate_local_shape_without_padding(neighbors=process_info.neighbors, partial_grid=grid)
        comm.Send(np.ascontiguousarray(partial_grid), dest=0, tag=process_info.rank)

    return full_grid


def calculate_pdf_mpi(process_info, communicator, density, u, time_steps, plot_range=None):
    """
    Perform the sliding lid simulation using MPI.

    Args:
        process_info: MpiPackageStructure instance containing information about the process.
        communicator: MPI communicator.
        density (ndarray): Initial density value.
        u (ndarray): Initial velocity value.
        time_steps (int): The total number of time steps.
        plot_range (ndarray): Time steps for which to show the velocity field plots.

    Returns:
        ndarray: The final velocity profile obtained from the sliding lid simulation.
    """

    def calculate_final_velocity():
        c = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1], [0, 0, 1, 0, -1, 1, 1, -1, -1]])  # velocity set
        final_pdf = combine_data(process_info=process_info, grid=lbm.pdf, comm=communicator)

        if process_info.rank == 0:
            final_density = np.einsum('ijk->jk', final_pdf)
            final_velocity = np.einsum('ai,ixy->axy', c, final_pdf) / final_density

            return final_velocity

    lbm = LatticeBoltzmann(x=process_info.x, y=process_info.y, process_info=process_info, communicator=communicator)

    lbm.density = density
    lbm.velocity = u
    lbm.update_params_mpi()
    velocity_fields = []

    for step in trange(time_steps):
        lbm.run_simulation(step=step, collision=True, boundary_conditions=True, mpi=True)
        lbm.pdf = perform_communication(grid=lbm.pdf, info=process_info, comm=communicator)

        if step in plot_range-1:
            velocity_fields.append(calculate_final_velocity())

    return calculate_final_velocity(), velocity_fields
