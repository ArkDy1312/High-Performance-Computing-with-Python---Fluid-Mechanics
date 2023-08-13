import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib.cm import ScalarMappable
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
import os

from src.utils import analytical_sinus_decay


def visualise_density_velocity(time_step, lbm, title):
    """
    Visualizes the density and velocity fields at a given time step using heatmaps and arrow plots.

    Args:
        time_step (int): The time step at which to visualize the fields.
        lbm (LatticeBoltzmann object): An object containing the density and velocity fields.
        title (str): The title for the plot.

    Returns:
        None
    """
    density = lbm.density
    velocity = lbm.velocity

    # Visualize density as a heatmap
    plt.subplot(1, 2, 1)
    plt.imshow(density, cmap='hot', origin='lower')
    plt.colorbar()
    plt.title(f'Density at time step {time_step}')

    # Visualize velocity components as arrows
    plt.subplot(1, 2, 2)
    plt.quiver(velocity[0, :, :], velocity[1, :, :])
    plt.title(f'Velocity Field at time step {time_step}')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def visualise_sinus_density_decay(max_position, density_init, densities, time_steps, grid_size_x, omega, epsilon):
    """
   Visualizes the decay of density over time for a sinusoidal wave, comparing the numerical and analytical results.

   Args:
       max_position (ndarray): The position at which to compare the densities.
       density_init (float): The initial density value.
       densities (ndarray): Array of density values at different time steps.
       time_steps (int): The total number of time steps.
       grid_size_x (int): The size of the grid in the x-dimension.
       omega (float): The relaxation factor for the simulation.
       epsilon (float): Small parameter used for normalization.

   Returns:
       None
   """
    output_path = "figures/shearWaveDecay"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    t = np.arange(time_steps)

    plt.figure(figsize=(10, 6))
    plt.plot(t, densities[:, max_position], label='Numerical Density')
    plt.plot(t, density_init + analytical_sinus_decay(time_steps=t, positions=max_position, grid_size=grid_size_x,
                                                      omega=omega, epsilon=epsilon),
             label="Analytical decay", linestyle="dashed", color="red")
    plt.plot(t, density_init - analytical_sinus_decay(time_steps=t, positions=max_position, grid_size=grid_size_x,
                                                      omega=omega, epsilon=epsilon),
             linestyle="dashed", lw=1.5, color="red")
    plt.xlabel("time_steps")
    plt.ylabel("a(t)")
    # plt.title(f'Evolution of Density with time for omega={omega}, Position: y={max_position}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'sinusoidal_density_decay.jpg'), dpi=300)
    plt.show()


def visualise_decay_different_omegas(time_steps, density_init, epsilon, grid_size_x, different_omegas,
                                     max_position, multiple_densities):
    """
    Visualizes the decay of density with different relaxation times (omegas) for a sinusoidal wave.

    Args:
        time_steps (int): The total number of time steps.
        density_init (float): The initial density value.
        epsilon (float): Small parameter used for normalization.
        grid_size_x (int): The size of the grid in the x-dimension.
        different_omegas (ndarray): Array of different relaxation times (omegas).
        max_position (ndarray): The position at which to compare the densities.
        multiple_densities (ndarray): Array of density values at different relaxation times.

    Returns:
        None
    """
    output_path = "figures/shearWaveDecay"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    x_axis = np.arange(time_steps)
    plt.figure(figsize=(18, 10))

    for idx, omega in enumerate(different_omegas):
        # get absolut function for more function points
        absolut_function = np.abs(multiple_densities[idx][:, max_position] - density_init)
        # normalizing
        absolut_function = absolut_function / epsilon
        maxima_s = argrelextrema(absolut_function, np.greater, mode='wrap')
        plt.plot(x_axis, (analytical_sinus_decay(x_axis, max_position, grid_size_x, omega, epsilon) / epsilon),
                 label="theoretical" if idx == 0 else "", linestyle="dashed", lw=1.5, color="red")
        plt.plot(maxima_s[0][::2], absolut_function[maxima_s[0]][::2], 'x', label=round(omega, 2), markersize=8,
                 linewidth=4)

    # plt.title(f"Shear Wave decay with Different Omegas, Position: y={max_position}")
    plt.xlabel("Time Steps")
    plt.ylabel("a(t)/a(0)")
    plt.ylim(-0.1, 1.1)
    plt.grid()
    plt.legend(title="Omega", loc='upper right', fancybox=True, numpoints=1)
    plt.savefig(os.path.join(output_path, 'shear_wave_decay_for_multiple_omegas.jpg'), bbox_inches='tight', dpi=300)
    plt.show()


def visualise_theoretical_vs_measured_viscosity(different_omegas, multiple_densities, max_position, density_init,
                                                epsilon, grid_size_x):
    """
    Visualizes the theoretical and measured viscosity for different relaxation times (omegas).

    Args:
        different_omegas (ndarray): Array of different relaxation times (omegas).
        multiple_densities (ndarray): Array of density values at different relaxation times.
        max_position (ndarray): The position at which to measure the viscosity.
        density_init (float): The initial density value.
        epsilon (float): Small parameter used for normalization.
        grid_size_x (int): The size of the grid in the x-dimension.

    Returns:
        None
    """
    output_path = "figures/shearWaveDecay"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    theoretical_viscosity = np.zeros((different_omegas.shape[0]))
    measured_viscosity = np.zeros((different_omegas.shape[0]))

    for idx, omega in enumerate(different_omegas):
        # theoretical viscosity
        theoretical_viscosity[idx] = 1 / 3 * (1 / omega - 0.5)
        # measured viscosity
        quantities = multiple_densities[idx, ...][:, max_position]
        densities = np.array(np.abs(quantities - density_init))
        # get absolut function for more function points
        peaks = argrelextrema(densities, np.greater, mode='wrap')[0]
        densities = densities[peaks]
        measured_viscosity[idx] = curve_fit(lambda t, v: epsilon * np.exp(-v * t * (2 * np.pi / grid_size_x) ** 2),
                                            xdata=peaks, ydata=densities)[0][0]
    plt.figure(figsize=(12, 8))
    # plt.title("Theoretical vs Measured Viscosity for Different Omegas")
    plt.xlabel("Omega ω")
    plt.ylabel("Kinematic Viscosity")
    plt.plot(different_omegas, theoretical_viscosity, 'b', label="Theoretical")
    plt.plot(different_omegas, measured_viscosity, 'r', label="Measured")
    plt.legend(title="Viscosity")
    plt.savefig(os.path.join(output_path, 'theoretical_vs_measured_viscosity.jpg'), bbox_inches='tight', dpi=300)
    plt.show()


def visualise_evolution_velocity(velocities, time_steps, grid_size_x, grid_size_y, omega, epsilon):
    """
    Visualizes the evolution of velocity over time for a sinusoidal wave.

    Args:
        velocities (ndarray): Array of velocity values at different time steps.
        time_steps (int): The total number of time steps.
        grid_size_y (int): The size of the grid in the y-dimension.
        grid_size_x (int): The size of the grid in the x-dimension.
        omega (float): The relaxation time for the simulation.
        epsilon (float): Small parameter used for normalization.

    Returns:
        None
    """
    output_path = "figures/shearWaveDecay"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # First plot
    plt.figure(figsize=(10, 8))
    velocity_time_steps = np.arange(0, time_steps, 400)
    label = ''
    color_map = plt.cm.jet
    norm = plt.Normalize(vmin=min(velocity_time_steps), vmax=max(velocity_time_steps))
    i = 0
    for time in velocity_time_steps:
        plt.plot(velocities[time, :], range(grid_size_y), color=color_map(norm(velocity_time_steps[i])))
        i += 1
        if time == velocity_time_steps[-1]:
            label = "Analytical Decay"
        plt.plot(analytical_sinus_decay(time_steps=time, positions=np.arange(grid_size_y), grid_size=grid_size_y,
                                        omega=omega, epsilon=epsilon), np.arange(grid_size_y), label=label,
                 linestyle="dashed", color="black", alpha=0.3)
    plt.legend(title="Measured Decay at Time Step:")
    plt.xlabel("Velocity")
    plt.ylabel("Y dimension")
    plt.xlim(-epsilon * 1.1, epsilon * 1.1)
    sm = ScalarMappable(cmap=color_map, norm=norm)
    plt.colorbar(sm, label="Step")
    plt.legend()
    # plt.title("Evolution of Velocity with time")

    # Save the first plot
    plt.savefig(os.path.join(output_path, "evolution_of_velocity_over_time.jpg"), bbox_inches='tight', dpi=300)
    plt.show()

    # Second plot
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(time_steps), velocities[:, grid_size_x // 4], color="black", label="Measured Decay")
    plt.plot(np.arange(time_steps), analytical_sinus_decay(time_steps=np.arange(time_steps),
                                                           positions=grid_size_x // 4, grid_size=grid_size_x,
                                                           omega=omega, epsilon=epsilon), label="Analytical Decay",
             linestyle="dashed", color="red")
    plt.xlabel("Time Step")
    plt.ylabel("Velocity")
    plt.legend()
    # plt.title(f"Velocity Decay at Position x={grid_size_x // 4}")

    # Save the second plot
    plt.savefig(os.path.join(output_path, "velocity_decay_with_time_at_position.jpg"), bbox_inches='tight', dpi=300)
    plt.show()


def visualise_velocity_vectors_couette_poiseuille(velocity, time_step, wall_velocity, poiseuille=False,
                                                  epsilon=None, omega=None):
    """
   Visualizes the velocity streamplot for Couette or Poiseuille flow.

   Args:
       velocity (ndarray): Velocity at the final timestep.
       time_step (int): Time Step at which the velocity was recorded.
       wall_velocity (float): Wall Velocity.
       poiseuille (bool): Flag indicating whether to plot Couette or Poiseuille flow.
       epsilon (float): Small parameter used for normalization.
       omega (float): The relaxation time for the simulation.

   Returns:
       None
   """
    if poiseuille:
        output_path = "figures/poiseuilleFlow"
    else:
        output_path = "figures/couetteFlow"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    grid_size_y = velocity.shape[2]
    grid_size_x = velocity.shape[1]
    y_a = np.arange(0, grid_size_y)
    vel_x = velocity[0][int(grid_size_x / 2), :]
    plt.figure(figsize=(12, 8))
    if poiseuille:
        viscosity = 1 / 3 * (1 / omega - 0.5)
        y_analytical = np.linspace(0, grid_size_y, grid_size_y + 1) + 0.5
        analytical = (-0.5 * (-2.0 * epsilon / grid_size_x / viscosity / 3) * y_analytical *
                      (grid_size_y - y_analytical))[:-1]
    else:
        analytical = y_a/grid_size_y*wall_velocity

    # Plot only 1/3 of the vectors
    for i in range(0, len(vel_x), 3):
        val, y_coord = vel_x[i], y_a[i]
        origin = [0, y_coord]
        plt.quiver(*origin, val, 0.0, color='royalblue', scale_units='xy', scale=1, headwidth=2)

    plt.plot(vel_x, y_a, label='Simulated', c='blue', linestyle=':')
    plt.plot(analytical, y_a, label="Analytical", linestyle='--', color='black')
    plt.axhline(y=0, color='dimgray', label='Bottom Wall')  # Plotting the wall at y = 0
    if poiseuille:
        plt.axhline(y=grid_size_y - 1, color='r', label='Top Wall')  # Plotting the top wal
    else:
        plt.axhline(y=grid_size_y - 1, color='r', label='Moving Wall')  # Plotting the moving wall

    plt.ylabel('y')
    plt.xlabel('Velocity')
    # plt.title(f'Velocity Vectors at Time Step: {time_step}')
    plt.legend()

    if poiseuille:
        plt.savefig(os.path.join(output_path, 'poiseuille_flow_velocity_field.jpg'), bbox_inches='tight', dpi=300)
    else:
        plt.savefig(os.path.join(output_path, 'couette_flow_velocity_field.jpg'), bbox_inches='tight', dpi=300)
    plt.show()


def visualise_velocity_evolution_couette_poiseuille(grid_size_x, grid_size_y, wall_velocity, velocities, epsilon,
                                                    omega, poiseuille=False):
    """
   Visualizes the velocity profiles for Couette or Poiseuille flow.

   Args:
       grid_size_x (int): The size of the grid in the x-dimension.
       grid_size_y (int): The size of the grid in the y-dimension.
       wall_velocity (float): The velocity of the wall in Couette flow.
       velocities (ndarray): Array of velocity profiles at different time intervals.
       epsilon (float): Small parameter used for normalization.
       omega (float): The relaxation time for the simulation.
       poiseuille (bool): Flag indicating whether to plot Couette or Poiseuille flow.

   Returns:
       None
   """
    if poiseuille:
        output_path = "figures/poiseuilleFlow"
    else:
        output_path = "figures/couetteFlow"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if poiseuille:
        viscosity = 1 / 3 * (1 / omega - 0.5)
        y_analytical = np.linspace(0, grid_size_y, grid_size_y + 1) + 0.5
        analytical = (-0.5 * (-2.0 * epsilon / grid_size_x / viscosity / 3) * y_analytical *
                      (grid_size_y - y_analytical))[:-1]
    else:
        analytical = np.arange(0, grid_size_y) / grid_size_y * wall_velocity

    num_profiles = np.arange(len(velocities))
    x_vals = np.arange(grid_size_y)
    color_map = plt.cm.jet
    norm = plt.Normalize(vmin=min(num_profiles), vmax=max(num_profiles))  # Normalize for the color map
    plt.figure(figsize=(12, 8))
    for i in range(len(velocities)):
        if i % 200 == 0:
            plt.plot(velocities[i], x_vals, color=color_map(norm(num_profiles[i])))
    plt.plot(analytical, x_vals, label="Analytical", linestyle='--', linewidth=3, color='black')
    if not poiseuille:
        plt.axhline(y=0, color='dimgray', linestyle='--', label='Bottom Wall')  # Plotting the wall at y = 0
        plt.axhline(y=grid_size_y-1, color='r', linestyle='--', label='Moving Wall')  # Plotting the moving wall
    # plt.title("Evolution of Velocity Profile")
    plt.xlabel("Velocity")
    plt.ylabel("y")
    plt.legend()
    # Create a ScalarMappable to add the color bar
    sm = ScalarMappable(cmap=color_map, norm=norm)
    plt.colorbar(sm, label="Step")

    if poiseuille:
        plt.savefig(os.path.join(output_path, 'poiseuille_flow_velocity_field_evolution.jpg'), bbox_inches='tight',
                    dpi=300)
    else:
        plt.savefig(os.path.join(output_path, 'couette_flow_velocity_field_evolution.jpg'), bbox_inches='tight',
                    dpi=300)
    plt.show()


def plot_density_centerline(density, grid_size_x, grid_size_y):
    """
    Plot the density along the centerline of the channel 𝜌(𝑥, ℎ/2).

    Args:
        density (ndarray): Density field from the simulation.
        grid_size_x (int): Size of the grid in the x-direction.
        grid_size_y (int): Size of the grid in the y-direction.

    Returns:
        None
    """
    output_path = "figures/poiseuilleFlow"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    centerline_index = grid_size_y // 2
    centerline_density = density[:, centerline_index]
    x_coords = np.arange(grid_size_x)

    plt.figure(figsize=(8, 6))
    plt.plot(x_coords, centerline_density)
    plt.xlabel('x')
    plt.ylabel('Density')
    # plt.title('Density along the Centerline of the Channel')
    plt.grid(axis='y')
    plt.savefig(os.path.join(output_path, 'poiseuille_flow_centerline_density.jpg'), bbox_inches='tight', dpi=300)
    plt.show()


def visualise_sliding_lid(grid_shape, velocity, num_steps, omega, wall_velocity, mpi=False):
    """
    Visualizes the fluid flow in a sliding lid simulation.

    Args:
        grid_shape (list): List containing grid shape in x and y directions.
        velocity (ndarray): Array of velocity values at different time steps.
        num_steps (int): The total number of time steps.
        omega (float): The relaxation time for the simulation.
        mpi (bool): Flag indicating whether MPI (Message Passing Interface) is used.
        wall_velocity (float): Velocity of the moving wall.

    Returns:
        None
    """
    if mpi:
        output_path = "figures/slidingLidParallel"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    else:
        output_path = "figures/slidingLidSerial"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    x_a = np.arange(0, grid_shape[0])
    y_a = np.arange(0, grid_shape[1])
    x, y = np.meshgrid(x_a, y_a)
    norm = plt.Normalize(vmin=0, vmax=wall_velocity)  # Normalize for the color map
    cmap = plt.cm.jet
    speed = np.sqrt(velocity[0].T ** 2 + velocity[1].T ** 2)

    # plot
    plt.close('all')
    plt.figure(figsize=(8, 5))
    plt.streamplot(x, y, velocity[0].T, velocity[1].T, color=speed, cmap=cmap)
    ax = plt.gca()
    ax.set_xlim([0, grid_shape[0]])
    ax.set_ylim([0, grid_shape[1]+1])
    plt.title(f"Grid={grid_shape[0]}x{grid_shape[1]}, Omega={omega}, step={num_steps}")
    plt.xlabel("x-Position")
    plt.ylabel("y-Position")
    # Create a ScalarMappable to add the color bar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, label="Velocity", orientation="vertical")

    plt.tight_layout()
    if mpi:
        plt.savefig(os.path.join(output_path, 'sliding_lid.jpg'), dpi=300)
    else:
        plt.savefig(os.path.join(output_path, 'sliding_lid.jpg'), dpi=300)
    plt.show()


def animate_sliding_lid(grid_shape, velocity_fields, omega, wall_velocity):
    """
    Creates and saves an animation of the fluid flow in a sliding lid simulation.

    Args:
        grid_shape (list): List containing grid shape in x and y directions.
        velocity_fields (list): List of velocity field arrays at different time steps.
        omega (float): The relaxation time for the simulation.
        wall_velocity (float): Velocity of the moving wall.

    Returns:
        None
    """
    output_path = "animation/slidingLid"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    fig, ax = plt.subplots(figsize=(8, 5))

    x_a = np.arange(0, grid_shape[0])
    y_a = np.arange(0, grid_shape[1])
    x, y = np.meshgrid(x_a, y_a)
    norm = plt.Normalize(vmin=0, vmax=wall_velocity)  # Normalize for the color map
    cmap = plt.cm.jet

    ax.set_xlim([0, grid_shape[0]])
    ax.set_ylim([0, grid_shape[1]+1])
    ax.set_title(f"Grid={grid_shape[0]}x{grid_shape[1]}, Omega={omega}, step=1")
    ax.set_xlabel("x-Position")
    ax.set_ylabel("y-Position")
    # Create a ScalarMappable to add the color bar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, label="Velocity", orientation="vertical")

    def animate(frame):
        ax.clear()
        speed = np.sqrt(velocity_fields[frame][0].T ** 2 + velocity_fields[frame][1].T ** 2)
        ax.streamplot(x, y, velocity_fields[frame][0].T, velocity_fields[frame][1].T, color=speed, cmap=cmap)
        ax.set_xlim([0, grid_shape[0]])
        ax.set_ylim([0, grid_shape[1]+1])
        ax.set_title(f"Grid={grid_shape[0]}x{grid_shape[1]}, Omega={omega}, step={frame+1}")
        ax.set_xlabel("x-Position")
        ax.set_ylabel("y-Position")

    anim = FuncAnimation(fig, animate, frames=len(velocity_fields), blit=False)
    anim.save(os.path.join(output_path, "slidingLid.mp4"), writer='ffmpeg', fps=1000, dpi=50)


def visualise_sliding_lid_at_different_time_steps(velocity_fields, grid_shape, plot_range, mpi=False):
    """
   Visualizes the fluid flow at different time steps in a sliding lid simulation and saves the plots.

   Args:
       velocity_fields (list): List of velocity field arrays at different time steps.
       grid_shape (list): List containing grid shape in x and y directions.
       plot_range (ndarray): Time steps for which to show the velocity field plots.
       mpi (bool): Flag indicating whether MPI (Message Passing Interface) is used.

   Returns:
       None
   """
    if mpi:
        output_path = "figures/slidingLidParallel"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    else:
        output_path = "figures/slidingLidSerial"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    n_cols = 3
    n_rows = 2
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(20, 10))
    flat_axes = ax.flatten()
    x_a = np.arange(0, grid_shape[0])
    y_a = np.arange(0, grid_shape[1])
    x, y = np.meshgrid(x_a, y_a)

    for i, ax in enumerate(flat_axes):
        ax.set_xlim(0, grid_shape[0])
        ax.set_ylim(0, grid_shape[1]+1)
        ax.set_ylabel("y-Position")
        ax.set_xlabel("x-Position")
        ax.set_title(f"Time Step: {plot_range[i]}")

        v_x = velocity_fields[i][0]
        v_y = velocity_fields[i][1]
        speed = np.sqrt(v_x.T ** 2 + v_y.T ** 2)
        ax.streamplot(x, y, v_x.T, v_y.T, color=speed, cmap=plt.cm.jet)
    fig.tight_layout()
    if mpi:
        fig.savefig(os.path.join(output_path, 'sliding_lid_at_different_time_steps.jpg'), dpi=300)
    else:
        fig.savefig(os.path.join(output_path, 'sliding_lid_at_different_time_steps.jpg'), dpi=300)
    plt.show()


def visualise_sliding_lid_for_different_parameters(values, mpi=False):
    """
    Visualizes the fluid flow for different parameter values in a sliding lid simulation and saves the plots.

    Args:
        values (list): List of dictionaries containing simulation parameter values and velocity fields.
        mpi (bool): Flag indicating whether MPI (Message Passing Interface) is used.

    Returns:
        None
    """
    if mpi:
        output_path = "figures/slidingLidParallel"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    else:
        output_path = "figures/slidingLidSerial"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    n_cols = 4
    n_rows = 4
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(44, 24))
    flat_axes = ax.flatten()

    for i, ax in enumerate(flat_axes):
        x_a = np.arange(0, values[i]["Grid_shape"][0])
        y_a = np.arange(0, values[i]["Grid_shape"][1])
        x, y = np.meshgrid(x_a, y_a)
        ax.set_xlim(0, values[i]["Grid_shape"][0])
        ax.set_ylim(0, values[i]["Grid_shape"][1] + 1)
        ax.set_ylabel("y-Position", fontsize=20)
        ax.set_xlabel("x-Position", fontsize=20)
        ax.set_title(f"Grid={values[i]['Grid_shape'][0]}x{values[i]['Grid_shape'][1]}, "
                     f"Re_Number={values[i]['Reynolds_number']}, W_Velocity={values[i]['Wall_velocity']}",
                     fontsize=28)

        v_x = values[i]['Velocity_field'][0]
        v_y = values[i]['Velocity_field'][1]
        speed = np.sqrt(v_x.T ** 2 + v_y.T ** 2)
        ax.streamplot(x, y, v_x.T, v_y.T, color=speed, cmap=plt.cm.jet)
    fig.tight_layout()
    if mpi:
        plt.savefig(os.path.join(output_path, 'sliding_lid_for_different_parameters.jpg'), dpi=300)
    else:
        plt.savefig(os.path.join(output_path, 'sliding_lid_for_different_parameters.jpg'), dpi=300)
    # plt.show()
