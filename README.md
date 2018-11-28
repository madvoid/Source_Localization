# Source Localization

Use particle swarm optimization (PSO) and possible MCMC to perform source localization in 3D unsteady environments. Work in progress. Being completed in python in case some sweet libraries need to be used in the future

## Data

For now, data will be stored in the git repository. If file sizes become too large, they will be removed. 

Five files are required from every QUIC run:

* `celltype.mat`
* `concentration.mat`
* `QP_params.inp`
* `QU_simparams.inp`
* `QP_source.inp`

The file `Domain_Maker.m` creates a file called `Data.mat` in a run's given folder. This `Data.mat` file is imported into the Python PSO code. Therefore, if a new QUIC simulation is run, `Domain_Maker.m` must be run before any other code is.

## Parameters to Experiment With

PSO has several different parameters and hyperparameters that can be modified, as well as several modifications that can be made. The following list presents, in no particular order, things that can be modified with this implementation of PSO.

* Velocity update constants `c1, c2`
* Velocity clamping options or inertia weight decrease
* Boundary offset
* Moving integer vs real number
* Incorporate wind into velocity update
* Velocity update rules for discrete domain
* What to do when drone is in building
  * Just move in direction that's not building
  * Recalculate velocity with random component
  * Move in different way, ray tracing?
* Adaptive weights, every time an unreachable position comes up weights change to move away from that position
* Have continuous position but convert to discrete index to check if in no-go zone
* Add mean wind vector or mean concentration gradient as vector to PSO update 
* Add "exploding factor" when mean wind changes to really explore area

## TODO

1. 