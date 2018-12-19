# -------------------------------------------------------------------------------------------------
# Name: Simple1_Stats.py
#
# Author: Nipun Gunawardena
#
# Purpose: Run Simple1 PSO many times to get statistics
# -------------------------------------------------------------------------------------------------

import sys
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, '../')
from PSO import *
from MatRead import readQUICMat

if __name__ == "__main__":
    # Create save path
    basePath = '../Results/Simple1/Statistics/Simple1_'

    # Retrieve domain and data
    Simple1Domain, X, Y, Z, B, C, C_Plot = readQUICMat('../QUIC Data/Simple1/Data.mat')

    # Prepare
    numParticles = 50
    numIters = 50
    bestLocs = np.zeros((numIters, Simple1Domain.dimension))
    convIters = np.zeros(numIters)

    # Iterate
    for i in tqdm(range(numIters)):
        Simple1PSO = PSO(C, Simple1Domain, numberParticles=numParticles, maskArray=B, maximumIterations=300)
        Simple1PSO.run(checkNeighborhood=True, verbose=False)
        bestLocs[i,:] = Simple1PSO.bestPositionHistory[-1:,:]
        convIters[i] = np.argmin(Simple1PSO.bestFitnessHistory)

    def reject_outliers(data, m=2):
        return data[abs(data - np.mean(data)) < m * np.std(data)]

    # Plot and save
    fig, ax = plt.subplots()
    sns.distplot(convIters, rug=True, axlabel='Convergence Iterations')
    ax.set_title(f'Number of Particles: {numParticles} || Iterations: {numIters}')
    fig.savefig(basePath + 'ConvIters.pdf')

    fig, axes = plt.subplots(3, 1)
    ax = sns.distplot(reject_outliers(bestLocs[:,0]), rug=True, kde=False, axlabel='Found X Location', ax=axes[0])
    ax.axvline(x=Simple1Domain.sourceLoc[0], c='r')
    ax.set_title(f'Num Particles: {numParticles} || Iterations: {numIters} || Grid Spacing: {Simple1Domain.ds}')
    ax = sns.distplot(reject_outliers(bestLocs[:,1]), rug=True, kde=False, axlabel='Found Y Location', ax=axes[1])
    ax.axvline(x=Simple1Domain.sourceLoc[1], c='r')
    ax = sns.distplot(reject_outliers(bestLocs[:,2]), rug=True, kde=False, axlabel='Found Z Location', ax=axes[2])
    ax.axvline(x=Simple1Domain.sourceLoc[2], c='r')
    fig.savefig(basePath + 'FoundLocs.pdf')

    plt.show()
