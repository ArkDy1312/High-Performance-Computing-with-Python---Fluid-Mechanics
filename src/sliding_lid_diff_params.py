"""

Plot the velocity fields for different Reynolds numbers.

"""
from argparse import ArgumentParser
from mpi4py import MPI
import os
import sys

# Get the parent directory's path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

from milestones import milestone_7, milestone_6
from visualisations import visualise_sliding_lid_for_different_parameters

parser = ArgumentParser()
parser.add_argument('-I', '--implementation_type', type=str, required=True, help='Whether parallel implementation True '
                                                                                 'or False. Provide either T or F')
args = parser.parse_args()

# Initialization
if args.implementation_type == 'T':
    mpi = True
elif args.implementation_type == 'F':
    mpi = False
else:
    raise ValueError('Argument "I"/"implementation_type" should be either T or F')
total_time_steps = 100000
x = [400, 300]
y = [400, 300]
wall_velocity = [0.1, 0.5]
reynolds_number = [250, 500, 750, 1000]
values = []
for i in range(len(x)):
    for j in range(len(wall_velocity)):
        for k in range(len(reynolds_number)):
            if mpi:
                final_velocity = milestone_7.main(total_time_steps=total_time_steps, x=x[i], y=y[i],
                                                  wall_velocity=wall_velocity[j], reynolds_num=reynolds_number[k],
                                                  num_procs_x=4, num_procs_y=2, visualise=False)
            else:
                final_velocity = milestone_6.main(total_time_steps=total_time_steps, x=x[i], y=y[i],
                                                  wall_velocity=wall_velocity[j], reynolds_num=reynolds_number[k],
                                                  visualise=False)
            values.append({'Grid_shape': [x[i], y[i]], 'Wall_velocity': wall_velocity[j],
                           "Reynolds_number": reynolds_number[k], "Velocity_field": final_velocity})

# If parallel implementation
if mpi:
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        visualise_sliding_lid_for_different_parameters(values=values, mpi=mpi)
else:
    visualise_sliding_lid_for_different_parameters(values=values)
