from argparse import ArgumentParser
import subprocess

import src.milestones.milestone_1 as m1
import src.milestones.milestone_2 as m2
import src.milestones.milestone_3 as m3
import src.milestones.milestone_4 as m4
import src.milestones.milestone_5 as m5
import src.milestones.milestone_6 as m6
import src.milestones.milestone_7 as m7


def main():
    """
    Main function to run different simulation experiments based on the provided arguments.
    """
    # Names for the different simulations
    experiment_names = {'m1': m1, 'm2': m2, 'swd': m3, 'cf': m4, 'pf': m5, 'sls': m6, 'slp': m7}

    # Get the command arguments
    parser = ArgumentParser()
    parser.add_argument('-E', '--experiment_name', type=str, required=True, help='The simulation experiment to run.')
    parser.add_argument('-X', type=int, required=True, help='The different lattice size in the x direction.')
    parser.add_argument('-Y', type=int, required=True, help='The different lattice size in the y direction.')
    parser.add_argument('-T', '--num_time_steps', type=int, default=None, help='The total time steps for which to run '
                                                                               'the simulation.')
    parser.add_argument('-O', '--omega', type=float, default=None, help='Relaxation Parameter.')
    parser.add_argument('--lx', type=int, default=None, help='The length of the domain in the x direction.')
    parser.add_argument('--ly', type=int, default=None, help='The length of the domain in the y direction.')
    parser.add_argument('--eps', type=float, default=None, help='Amplitude of the perturbation')
    parser.add_argument('-WV', '--wall_velocity', type=float, default=None, help='Velocity of the moving wall.')
    parser.add_argument('-DIPF', '--density_initial_poiseuille', type=float, default=1, help='Initial density for '
                                                                                             'Poiseuille flow.')
    parser.add_argument('-DISWD', '--density_initial_sinus_density', type=float, default=1, help='Initial density for'
                                                                                                   ' Shear Wave Decay '
                                                                                                   'of density.')
    parser.add_argument('-V', '--viscosity', type=float, default=None, help='Viscosity of the fluid.')
    parser.add_argument('-RN', '--reynolds_number', type=int, default=None, help='Reynolds Number.')
    parser.add_argument('-NP', '--num_processes', type=int, default=None, help='Total number of processes.')
    parser.add_argument('-NPX', '--num_processes_x', type=int, default=None, help='Number of processes in x direction.')
    parser.add_argument('-NPY', '--num_processes_y', type=int, default=None, help='Number of processes in y direction.')

    args = parser.parse_args()

    # For Milestone 1
    if args.experiment_name == 'm1':
        if args.num_time_steps is None:
            raise ValueError('Please provide the total number of time steps using -T argument.')
        experiment_names[args.experiment_name].main(x=args.X, y=args.Y, num_time_steps=args.num_time_steps)

    # For Milestone 2
    if args.experiment_name == 'm2':
        if args.num_time_steps is None:
            raise ValueError('Please provide the total number of time steps using -T argument.')
        elif args.omega is None:
            raise ValueError('Please provide the omega value using -O argument.')
        experiment_names[args.experiment_name].main(x=args.X, y=args.Y, num_time_steps=args.num_time_steps,
                                                    omega=args.omega)

    # For Shear Wave Decay
    if args.experiment_name == 'swd':
        if args.X != args.Y:
            raise ValueError('The lattice size in x direction should be equal to the lattice size in y direction for '
                             'this simulation. Please provide the correct lattice sizes using the -X and -Y command.')
        if args.omega is None:
            raise ValueError('Please provide the omega value using -O argument.')
        elif args.eps is None:
            raise ValueError('Please provide the epsilon value using --eps argument.')
        elif args.lx is None:
            raise ValueError('The value length of the domain in x direction is either not given. PLease provide the '
                             'value using --lx argument.')
        elif args.lx != args.X:
            raise ValueError('The value length of the domain in x direction is not matching with the lattice size in x '
                             'direction. PLease provide the correct value using --lx argument.')
        elif args.ly is None:
            raise ValueError('The value length of the domain in y direction is either not given. PLease provide the '
                             'value using --ly argument.')
        elif args.ly != args.Y:
            raise ValueError('The value length of the domain in y direction is not matching with the lattice size in y '
                             'direction. PLease provide the correct value using --ly argument.')
        elif args.num_time_steps is not None:
            print('You do not need to provide the number of time steps here. The sinusoidal density decay is running '
                  'for 10000 steps. The sinusoidal density decay is running for 6000 steps.')
        experiment_names[args.experiment_name].main(x=args.X, y=args.Y, lx=args.lx, ly=args.ly, omega_init=args.omega,
                                                    density_init=args.density_initial_sinus_density, epsilon=args.eps)

    # For Couette Flow
    if args.experiment_name == 'cf':
        if args.num_time_steps is None:
            raise ValueError('Please provide the total number of time steps using -T argument.')
        elif args.omega is None:
            raise ValueError('Please provide the omega value using -O argument.')
        elif args.wall_velocity is None:
            raise ValueError('Please provide the moving wall velocity value using -WV argument.')
        experiment_names[args.experiment_name].main(x=args.X, y=args.Y, num_time_steps=args.num_time_steps,
                                                    omega=args.omega, epsilon=args.eps,
                                                    wall_velocity=args.wall_velocity)

    # For Poiseuille Flow
    if args.experiment_name == 'pf':
        if args.num_time_steps is None:
            raise ValueError('Please provide the total number of time steps using -T argument.')
        elif args.omega is None:
            raise ValueError('Please provide the omega value using -O argument.')
        elif args.eps is None:
            raise ValueError('Please provide the epsilon value using --eps argument.')
        elif args.density_initial_poiseuille is None:
            raise ValueError('Please provide the initial density value for poiseuille flow using -DIPF argument.')
        elif args.wall_velocity is not None:
            raise ValueError('Moving wall velocity should not be passed for this simulation. Please remove the -WV.'
                             'argument')
        experiment_names[args.experiment_name].main(x=args.X, y=args.Y, num_time_steps=args.num_time_steps,
                                                    omega=args.omega, epsilon=args.eps,
                                                    density_init=args.density_initial_poiseuille)

    # For Sliding Lid Serial
    if args.experiment_name == 'sls':
        if args.num_time_steps is None:
            raise ValueError('Please provide the total number of time steps using -T argument.')
        elif args.omega is None and args.viscosity is None and args.reynolds_number is None:
            raise ValueError('Please provide either omega or viscosity or reynolds number using -O/-V/-RN argument.')
        elif args.wall_velocity is None:
            raise ValueError('Please provide the moving wall velocity value using -WV argument.')
        experiment_names[args.experiment_name].main(x=args.X, y=args.Y, total_time_steps=args.num_time_steps,
                                                    wall_velocity=args.wall_velocity, omega=args.omega,
                                                    viscosity=args.viscosity, reynolds_num=args.reynolds_number)

    # For Sliding Lid Parallel
    if args.experiment_name == 'slp':
        if args.num_time_steps is None:
            raise ValueError('Please provide the total number of time steps using -T argument.')
        elif args.omega is None and args.viscosity is None and args.reynolds_number is None:
            raise ValueError(
                'Please provide either omega or viscosity or reynolds number using -O/-V/-RN argument.')
        elif args.wall_velocity is None:
            raise ValueError('Please provide the moving wall velocity value using -WV argument.')
        elif args.num_processes is None:
            raise ValueError('Please provide the total number of processes to use using -NP argument.')
        elif args.num_processes_x is None:
            raise ValueError('Please provide the total number of processes in x direction using -NPX argument.')
        elif args.num_processes_y is None:
            raise ValueError('Please provide the total number of processes in y direction using -NPY argument.')
        # Command to run
        command = f"mpirun -np {args.num_processes} python src/run_sLid_parallel.py -T {args.num_time_steps} " \
                  f"-X {args.X} -Y {args.Y} -WV {args.wall_velocity} -O {args.omega} -V {args.viscosity} " \
                  f"-RN {args.reynolds_number} -NP {args.num_processes} -NPX {args.num_processes_x} " \
                  f"-NPY {args.num_processes_y}"
        subprocess.run(command, shell=True)


if __name__ == "__main__":
    main()
