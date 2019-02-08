# Source Localization

Use particle swarm optimization (PSO) and possible MCMC to perform source localization in 3D unsteady environments. Work in progress. Being completed in python in case some sweet libraries need to be used in the future. While a few more commits will be pushed here, long-term future development will happen at https://github.com/UtahEFD/Source_Localization

## Data

For now, data will be stored in the git repository. If file sizes become too large, they will be removed. 

Five files are required from every QUIC run:

* `celltype.mat`
* `concentration.mat`
* `QP_params.inp`
* `QU_simparams.inp`
* `QP_source.inp`

The file `Domain_Maker.m` creates a file called `Data.mat` in a run's given folder. This `Data.mat` file is imported into the Python PSO code. Therefore, if a new QUIC simulation is run, `Domain_Maker.m` must be run before any other code is. 

Until 12/19/2018 there were several test functions in the `Domains` folder. However, these were removed from the repository as the PSO code became more and more tailored to QUIC data.

