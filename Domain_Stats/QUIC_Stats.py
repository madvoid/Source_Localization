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
    baseName = 'Simple3'
    basePath = '../Results/' + baseName + '/Statistics/' + baseName + '_'

    # Retrieve domain and data
    quicDomain, X, Y, Z, B, C, C_Plot = readQUICMat('../QUIC Data/' + baseName + '/Data.mat')

    # Prepare
    numParticles = 50
    numParticlesList = [10, 20, 30, 40, 50]
    numIters = 100
    bestLocs = np.zeros((numIters, quicDomain.dimension, len(numParticlesList)))
    convIters = np.zeros((numIters, len(numParticlesList)))

    # Iterate
    for jIdx, j in tqdm(enumerate(numParticlesList)):
        for i in tqdm(range(numIters)):
            quicPSO = PSO(C, quicDomain, numberParticles=j, maskArray=B, maximumIterations=300)
            quicPSO.run(checkNeighborhood=True, verbose=False)
            bestLocs[i,:,jIdx] = quicPSO.bestPositionHistory[-1:,:]
            convIters[i, jIdx] = np.argmin(quicPSO.bestFitnessHistory)

    # Calculate nondimensional distance
    dist = np.zeros((numIters, len(numParticlesList)))
    distSort = np.zeros(dist.shape)
    for jIdx, j in enumerate(numParticlesList):
        dist[:,jIdx] = np.linalg.norm(bestLocs[:,:,jIdx] - quicDomain.sourceLoc, ord=2, axis=1) / np.linalg.norm(quicDomain.maxLims - quicDomain.minLims, ord=2)

    # Prepare CDF for distOn
    for jIdx, j in enumerate(numParticlesList):
        distSort[:,jIdx] = np.sort(dist[:,jIdx])
    pDist = np.array(range(len(dist))) / float(len(dist))

    # Plot and save
    sns.set()
    fig, ax = plt.subplots()
    for i in range(len(numParticlesList)):
        sns.distplot(convIters[:,i], kde=False, norm_hist=False, label=str(numParticlesList[i])+'Particles')
    ax.set_title(f'Number of Iterations to Convergence')
    ax.set_xlabel('Convergence Iterations (Limited to 300)')
    ax.legend()
    fig.savefig(basePath + 'ConvIters.pdf')

    fig, ax = plt.subplots()
    for i in range(len(numParticlesList)):
        ax = sns.lineplot(x = distSort[:,i], y = pDist, ax=ax, label=str(numParticlesList[i])+'Particles')
    ax.set_xlabel('deltaS/L')
    ax.set_ylabel('Percent of Runs')
    ax.legend()
    fig.savefig(basePath + 'FoundLocs.pdf')

    fig, ax = plt.subplots()
    for i in range(len(numParticlesList)):
        plt.semilogx(distSort[:,i], pDist)
    ax.set_xlabel('deltaS/L')
    ax.set_ylabel('Percent of Runs')
    ax.legend([str(i)+'Particles' for i in numParticlesList])
    fig.savefig(basePath + 'FoundLocsLog.pdf')

    plt.show()
