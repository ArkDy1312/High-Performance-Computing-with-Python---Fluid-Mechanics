import numpy as np
import warnings

# Suppress the warning
warnings.filterwarnings('ignore', category=RuntimeWarning)


class LatticeBoltzmann:
    def __init__(self, x, y, omega=None, epsilon=None, wall_velocity=None, density_in=None, density_out=None,
                 process_info=None, communicator=None, s_lid=False):
        """
        Initialize the LatticeBoltzmann object.

        Args:
            x (int): Size of the grid in the x-dimension.
            y (int): Size of the grid in the y-dimension.
            omega (float): Relaxation factor.
            epsilon (float): Small parameter used for normalization.
            wall_velocity (float): The velocity of the wall.
            density_in (float): Density at the inlet.
            density_out (float): Density at the outlet.
            process_info: Information about the MPI process.
            communicator: MPI communicator.
            s_lid (bool): Whether to perform Sliding Lid simulation.
        """
        self.x = x
        self.y = y
        self.density = np.zeros((x, y))
        self.velocity = np.zeros((2, x, y))
        self.pdf = np.zeros((9, x, y))  # Grid / Probability Density Function
        self.c = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1], [0, 0, 1, 0, -1, 1, 1, -1, -1]])  # Velocity Set
        self.omega = omega  # Relaxation Parameter
        self.weights = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
        self.epsilon = epsilon  # Amplitude of the perturbation
        self.wall_velocity = wall_velocity  # Moving Wall Velocity
        self.density_in = density_in  # Inlet Density
        self.density_out = density_out  # Outlet Density
        self.process_info = process_info
        self.communicator = communicator
        self.s_lid = s_lid
        self.old_pdf = np.zeros((9, x, y))
        self.opp_idxs = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

    def compute_density(self):
        """
        Compute the density from the distribution functions.

        Returns:
            ndarray: Density values.
        """

        return np.einsum('ijk->jk', self.pdf)

    def compute_velocity(self):
        """
        Compute the velocity from the distribution functions.

        Returns:
            ndarray: Velocity values.
        """

        density = self.compute_density()
        velocity = np.einsum('ai,ixy->axy', self.c, self.pdf) / np.maximum(density, 1e-12)
        return velocity

    def streaming(self):
        """
        Perform the streaming step of the LBM simulation.
        """

        for i in range(0, 9):
            self.pdf[i, :, :] = np.roll(self.pdf[i, :, :], self.c.T[i], axis=[0, 1])

    def equilibrium_distribution(self):
        """
        Compute the equilibrium distribution based on density and velocity.

        Returns:
            ndarray: Equilibrium distribution.
        """

        density = self.density
        velocity = self.velocity
        velocity_squared = np.sum(velocity ** 2, axis=0)
        equilibrium = np.einsum('i,xy->ixy', self.weights, density) * (
                1 + 3 * np.einsum('ai,ixy->axy', self.c.T, velocity)
                + (9 / 2) * np.einsum('ai,ixy->axy', self.c.T, velocity) ** 2
                - (3 / 2) * velocity_squared)
        return equilibrium

    def collision(self):
        """
        Perform the collision step of the LBM simulation.
        """

        equilibrium = self.equilibrium_distribution()
        self.pdf += self.omega * (equilibrium - self.pdf)

    def periodic_boundary_with_pressure_variations(self, density_in, density_out):
        """
        Apply periodic boundary conditions with pressure variations.

        Args:
            density_in: Density at the inlet.
            density_out: Density at the outlet.
        """

        def eq_dist(density, velocity):
            velocity_squared = np.sum(velocity ** 2, axis=0)
            eq = (self.weights * density) * (1 + 3 * np.einsum('ai,ix->ax', self.c.T, velocity)
                                             + (9 / 2) * np.einsum('ai,ix->ax', self.c.T, velocity) ** 2
                                             - (3 / 2) * velocity_squared).T
            return eq.T

        # get all the values
        self.density = self.compute_density()
        self.velocity = self.compute_velocity()
        equilibrium = self.equilibrium_distribution()

        # Inlet
        equilibrium_in = eq_dist(density_in, self.velocity[:, -2, :])
        in_idxs = [1, 5, 8]
        self.pdf[in_idxs, 0, :] = equilibrium_in[in_idxs, :] + (self.pdf[in_idxs, -2, :] - equilibrium[in_idxs, -2, :])

        # Outlet
        out_idxs = [3, 6, 7]
        equilibrium_out = eq_dist(density_out, self.velocity[:, 1, :])
        self.pdf[out_idxs, -1, :] = equilibrium_out[out_idxs, :] + (self.pdf[out_idxs, 1, :] -
                                                                    equilibrium[out_idxs, 1, :])

    def boundary_conditions(self):
        """
        Apply boundary conditions.
        """

        if self.s_lid:
            # Left Rigid Wall
            idxs = [6, 3, 7]
            self.pdf[self.opp_idxs[idxs], 0, :] = self.old_pdf[idxs, 0, :]

            # Right Rigid Wall
            idxs = [5, 1, 8]
            self.pdf[self.opp_idxs[idxs], -1, :] = self.old_pdf[idxs, -1, :]

        # Bottom Rigid Wall
        idxs = [4, 7, 8]
        self.pdf[self.opp_idxs[idxs], :, 0] = self.old_pdf[idxs, :, 0]

        # Top Moving Wall
        density_wall = 2.0 * (self.old_pdf[2, :, -1] + self.old_pdf[5, :, -1] + self.old_pdf[6, :, -1]) + \
                       self.old_pdf[0, :, -1] + self.old_pdf[1, :, -1] + self.old_pdf[3, :, -1]

        self.pdf[4, :, -1] = self.old_pdf[2, :, -1]
        self.pdf[7, :, -1] = self.old_pdf[5, :, -1] - 1 / 6 * self.wall_velocity * density_wall
        self.pdf[8, :, -1] = self.old_pdf[6, :, -1] + 1 / 6 * self.wall_velocity * density_wall

    def boundary_conditions_mpi(self):
        """
        Apply boundary conditions for the sliding lid simulation with MPI.
        """

        if self.process_info.boundaries_info.apply_right:
            idxs = [5, 1, 8]
            self.pdf[self.opp_idxs[idxs], -1, :] = self.old_pdf[idxs, -1, :]
        if self.process_info.boundaries_info.apply_left:
            idxs = [6, 3, 7]
            self.pdf[self.opp_idxs[idxs], 0, :] = self.old_pdf[idxs, 0, :]
        if self.process_info.boundaries_info.apply_bottom:
            idxs = [4, 7, 8]
            self.pdf[self.opp_idxs[idxs], :, 0] = self.old_pdf[idxs, :, 0]
        if self.process_info.boundaries_info.apply_top:
            density_wall = 2.0 * (self.old_pdf[2, :, -1] + self.old_pdf[5, :, -1] + self.old_pdf[6, :, -1]) + \
                           self.old_pdf[0, :, -1] + self.old_pdf[1, :, -1] + self.old_pdf[3, :, -1]

            self.pdf[4, :, -1] = self.old_pdf[2, :, -1]
            self.pdf[7, :, -1] = self.old_pdf[5, :, -1] - 1 / 6 * self.wall_velocity * density_wall
            self.pdf[8, :, -1] = self.old_pdf[6, :, -1] + 1 / 6 * self.wall_velocity * density_wall

    def update_params_mpi(self):
        """
        Update the simulation parameters for MPI.
        """

        self.wall_velocity = self.process_info.wall_velocity
        self.omega = self.process_info.relaxation

    def run_simulation(self, step, collision=False, boundary_conditions=False, periodic_boundary=False, mpi=False):
        """
        Run the LBM simulation for a given step.

        Args:
            step (int): The current step.
            collision (bool): Flag indicating whether to perform the collision step.
            boundary_conditions (bool): Flag indicating whether to apply boundary conditions.
            periodic_boundary (bool): Flag indicating whether to apply periodic boundary conditions.
            boundary_conditions (bool): Flag indicating whether to apply boundary conditions.
            mpi (bool): Flag indicating whether to use MPI for parallel simulation.
        """

        if step == 0:
            self.pdf = self.equilibrium_distribution()

        self.old_pdf = self.pdf.copy()

        if periodic_boundary:
            self.periodic_boundary_with_pressure_variations(density_in=self.density_in, density_out=self.density_out)

        self.streaming()

        if boundary_conditions and not mpi:
            self.boundary_conditions()
        if boundary_conditions and mpi:
            self.boundary_conditions_mpi()

        self.density = self.compute_density()
        self.velocity = self.compute_velocity()

        if collision:
            self.collision()
