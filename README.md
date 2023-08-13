# High Performance Computing: Fluid Mechanics with Python

This repository contains implementations of various fluid mechanics simulations using the Lattice Boltzmann Method (LBM) for High Performance Computing (HPC) applications.

## Simulations

The different simulations implemented in this repository are:

- **Milestone 1 (m1):** Simple Lattice Boltzmann Method
- **Milestone 2 (m2):** Simple Lattice Boltzmann Method with Collision operator
- **Milestone 3 (swd):** Shear Wave Decay
- **Milestone 4 (cf):** Couette Flow
- **Milestone 5 (pf):** Poiseuille Flow
- **Milestone 6 (sls):** Sliding Lid Serial
- **Milestone 7 (slp):** Sliding Lid Parallelization

## Running Simulations

To run any of these simulations, use the `run.py` script. 

Instructions how to run the experiments:

```
-E, '--experiment_name', type=str : The simulation experiment to run. For example (from above) "swd" implies Shear Wave Decay.
-X, type=int : The different lattice size in the x direction.
-Y, type=int : The different lattice size in the y direction.
-T, '--num_time_steps', type=int : The total time steps for which to run the simulation.
-O, '--omega', type=float : Relaxation Parameter.
--lx, type=int : The length of the domain in the x direction.
--ly, type=int : The length of the domain in the y direction.
--eps, type=float : Amplitude of the perturbation
-WV, '--wall_velocity', type=float : Velocity of the moving wall.
-DIPF, '--density_initial_poiseuille', type=float : Initial density for Poiseuille flow.
-DISWD, '--density_initial_sinus_density', type=float : Initial density for Shear Wave Decay of density.
-V, '--viscosity', type=float : Viscosity.
-RN, '--reynolds_number', type=int : Reynolds Number.
-NP, '--num_processes', type=int : Total number of processes.
-NPX, '--num_processes_x', type=int : Number of processes in x direction.
-NPY, '--num_processes_y', type=int : Number of processes in y direction.
```
Here are examples of how to run the Sliding Lid Serial and Parallel simulations:

- Sliding Lid Serial:

`python run.py -E sls -X 300 -Y 300 -T 100000 -RN 1000 -WV 0.1`

- Sliding Lid Parallel:

`python run.py -E slp -X 300 -Y 300 -T 100000 -RN 1000 -WV 0.1 -NP 9 -NPX 3 -NPY 3`


`Note:` There is no need to use `"mpirun"` for parallelizing the implementation; this is handled internally by the code.

## Test Cases

Some test cases have been provided to validate the implementations:

- `test_poiseuille_inlet_outlet.py`: Checks if the inlet and outlet flow rates of Poiseuille Flow are different.
- `test_parallel_vs_serial.py`: Compares results between parallel and serial Sliding Lid simulations.

`Note:` You need to run the Sliding Lid serial and parallel implementations at least once before running the second test.

## Scaling Test

A scaling test for the Sliding Lid parallel simulation is available, which measures and visualizes the MLUPS (Million Lattice Updates Per Second) for different combinations of parameters.

Instructions on how to run the scaling test:

```
-T, '--total_time_steps', type=int : Total number of time steps.
-P, '--different_num_processes', type=str : The different number of processes.
-X, type=str : The different lattice size in the x direction.
-Y, type=str : The different lattice size in the y direction.
```

Example command to run the scaling test:

`python scaling_test.py -T 10000 -P 100,200 -X 200,300 -Y 200,300`

`Note:` Similar to the other implementation, there is no need to use `"mpirun"` for parallelizing this test; it's handled internally.


