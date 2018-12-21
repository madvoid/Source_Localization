# -------------------------------------------------------------------------------------------------
# Name: QUIC_Stats.py
#
# Author: Nipun Gunawardena
#
# Purpose: Run QUIC PSO many times to get statistics
# -------------------------------------------------------------------------------------------------

import sys
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, '../')
from PSO import *
from MatRead import readQUICMat

if __name__ == "__main__":
    # Create save path
    baseName = 'OKC'
    basePath = '../Results/' + baseName + '/Statistics/' + baseName + '_'

    # Retrieve domain and data
    quicDomain, X, Y, Z, B, C, C_Plot = readQUICMat('../QUIC Data/' + baseName + '/Data.mat')

    # Prepare
    numParticles = 50
    numIters = 100
    bestLocs = np.zeros((numIters, quicDomain.dimension))
    convIters = np.zeros(numIters)

    # Iterate
    for i in tqdm(range(numIters)):
        quicPSO = PSO(C, quicDomain, numberParticles=numParticles, maskArray=B, maximumIterations=300)
        quicPSO.run(checkNeighborhood=True, verbose=False)
        bestLocs[i,:] = quicPSO.bestPositionHistory[-1:,:]
        convIters[i] = np.argmin(quicPSO.bestFitnessHistory)

    # Calculate nondimensional distance
    dist = np.linalg.norm(bestLocs - quicDomain.sourceLoc, ord=2, axis=1) / np.linalg.norm(quicDomain.maxLims - quicDomain.minLims, ord=2)

    # Prepare CDF for dist
    distSort = np.sort(dist)
    pDist = np.array(range(len(dist))) / float(len(dist))

    # Plot and save
    sns.set()
    fig, ax = plt.subplots()
    sns.distplot(convIters, rug=True, axlabel='Convergence Iterations')
    ax.set_title(f'Number of Particles: {numParticles} || Iterations: {numIters}')
    fig.savefig(basePath + 'ConvIters.pdf')

    fig, axes = plt.subplots(2, 1)
    ax = sns.distplot(dist, rug=True, kde=False, axlabel='deltaS/L', ax=axes[0])
    ax.set_title(f'Number of Particles: {numParticles} || Iterations: {numIters} || Grid Spacing: {quicDomain.ds}')
    ax = sns.lineplot(x = distSort, y = pDist, ax=axes[1])
    ax.set_xlabel('deltaS/L')
    fig.savefig(basePath + 'FoundLocs.pdf')

    plt.show()
