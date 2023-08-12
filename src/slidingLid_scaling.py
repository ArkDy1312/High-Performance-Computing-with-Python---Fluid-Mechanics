from argparse import ArgumentParser
import os
import sys

# Get the parent directory's path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

import src.milestones.milestone_7 as m7


def main():
    parser = ArgumentParser()
    parser.add_argument('-T', '--total_time_steps', type=int, required=True, help='The total time steps.')
    parser.add_argument('-X', type=int, required=True, help='The lattice size in the x direction.')
    parser.add_argument('-Y', type=int, required=True, help='The lattice size in the y direction.')
    parser.add_argument('-O', '--omega', type=str, default=None, help='Relaxation Parameter.')
    parser.add_argument('-WV', '--wall_velocity', type=float, default=0.1, help='Velocity of the moving wall.')
    parser.add_argument('-V', '--viscosity', type=float, default=0.03, help='Viscosity of the fluid.')
    parser.add_argument('-RN', '--reynolds_number', type=str, default=None, help='Reynolds Number.')
    parser.add_argument('-NP', '--num_processes', type=int, help='Total number of processes.')
    parser.add_argument('-NPX', '--num_processes_x', type=int, help='Number of processes in x direction.')
    parser.add_argument('-NPY', '--num_processes_y', type=int, help='Number of processes in y direction.')
    args = parser.parse_args()

    # Run simulation
    m7.main(x=args.X, y=args.Y, total_time_steps=args.total_time_steps, wall_velocity=args.wall_velocity,
            omega=args.omega, viscosity=args.viscosity, reynolds_num=args.reynolds_number,
            num_procs=args.num_processes, num_procs_x=args.num_processes_x, num_procs_y=args.num_processes_y,
            visualise=False, scaling=True)


if __name__ == '__main__':
    main()
