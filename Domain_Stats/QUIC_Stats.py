# -------------------------------------------------------------------------------------------------
# Name: QUIC_Stats.py
#
# Author: Nipun Gunawardena
#
# Purpose: Run QUIC PSO many times to get statistics
# -------------------------------------------------------------------------------------------------

import os
import sys
import seaborn as sns
from tabulate import tabulate
from tqdm import tqdm

sys.path.insert(0, '../')
from PSO import *
from MatRead import readQUICMat

if __name__ == "__main__":
    # Create save path
    baseName = 'Simple_R'
    basePath = '../Results/' + baseName + '/Statistics/' + baseName + '_'

    # Retrieve domain and data
    quicDomain, X, Y, Z, B, C, C_Plot = readQUICMat('../QUIC Data/' + baseName + '/Data.mat')

    # Prepare
    numParticlesList = [10, 20, 30, 40, 50]
    numIters = 100
    bestLocs = np.zeros((numIters, quicDomain.dimension, len(numParticlesList)))
    bestIdx = np.zeros((numIters, quicDomain.dimension, len(numParticlesList)))
    bestFitness = np.zeros((numIters, len(numParticlesList)))
    convIters = np.zeros((numIters, len(numParticlesList)))
    maxIters = 300
    tStart = 0

    # Iterate
    for jIdx, j in tqdm(enumerate(numParticlesList)):
        for i in tqdm(range(numIters)):
            quicPSO = PSO(C, quicDomain, numberParticles=j, maskArray=B, maximumIterations=maxIters, tStartIndex=tStart)
            quicPSO.run(checkNeighborhood=True, verbose=False, checkStuckParticle=True)
            bpIter = np.argmin(quicPSO.bestFitnessHistory)
            convIters[i, jIdx] = bpIter
            bestLocs[i,:,jIdx] = quicPSO.bestPositionHistory[bpIter,:]
            bestFitness[i, jIdx] = np.min(quicPSO.bestFitnessHistory)

    # Calculate nondimensional distance
    dist = np.zeros((numIters, len(numParticlesList)))
    distSort = np.zeros(dist.shape)
    for jIdx, j in enumerate(numParticlesList):
        dist[:,jIdx] = np.linalg.norm(bestLocs[:,:,jIdx] - quicDomain.sourceLoc, ord=2, axis=1) / np.linalg.norm(quicDomain.maxLims - quicDomain.minLims, ord=2)

    # Create 2d representation of 3d concentration, use log scale to highlight differences
    C_Plot_2d = flattenPlotQuic(tStart, C_Plot)
    concentrationMap = 'inferno'
    lineMap = 'g'
    zMap = 'winter'
    sourceMap = 'g'
    sLocX = quicDomain.sourceLoc[0]
    sLocY = quicDomain.sourceLoc[1]

    # Clean data
    convIters[convIters == 0] = np.nan      # 0 convergence iterations means plume wasn't found
    dist[np.isnan(convIters)] = np.nan      # Change distances to nans to for non-found plumes

    # Prepare CDF for dist
    for jIdx, j in enumerate(numParticlesList):
        distSort[:,jIdx] = np.sort(dist[:,jIdx])
    pDist = np.array(range(1, len(dist)+1)) / float(len(dist))

    sns.set()
    fig, ax = plt.subplots()
    for i in range(len(numParticlesList)):
        v = ~np.isnan(convIters[:,i])
        sns.distplot(convIters[v,i], kde=False, norm_hist=False, label=f"{numParticlesList[i]} Particles ({sum(v)})")
    ax.set_title(f'(a)')
    ax.set_xlabel('Convergence Iterations (Limited to 300)')
    ax.legend()
    fig.savefig(basePath + 'ConvIters.pdf')

    fig, ax = plt.subplots()
    for i in range(len(numParticlesList)):
        v = ~np.isnan(convIters[:, i])
        ax = sns.lineplot(x=distSort[~np.isnan(distSort[:,i]), i], y=np.array(range(1, sum(v)+1)) / float(sum(v)), ax=ax, label=f"{numParticlesList[i]} Particles ({sum(v)})")
    ax.set_xlabel('$\Delta S/L$')
    ax.set_ylabel('Percent of Successful Runs')
    ax.set_title(f'(b)')
    ax.legend()
    fig.savefig(basePath + 'FoundLocs.pdf')

    fig, ax = plt.subplots()
    vSum = np.zeros(len(numParticlesList))
    for i in range(len(numParticlesList)):
        v = ~np.isnan(convIters[:, i])
        vSum[i] = sum(v)
        plt.semilogx(distSort[~np.isnan(distSort[:,i]), i], np.array(range(1, sum(v)+1)) / float(sum(v)))
    ax.set_xlabel('$\Delta S/L$')
    ax.set_ylabel('Percent of Successful Runs')
    ax.set_title(f'(c)')
    ax.legend([f"{numParticlesList[idx]} Particles ({int(vSum[idx])})" for idx, i in enumerate(numParticlesList)])
    fig.savefig(basePath + 'FoundLocsLog.pdf')

    # Plot 2D representation of field
    fig, ax = plt.subplots()
    ax.pcolormesh(X[:, :, 0], Y[:, :, 0], C_Plot_2d, cmap=concentrationMap, edgecolor='none')
    ax.scatter(sLocX, sLocY, c=sourceMap, marker='*', s=50)  # Actual best position
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Domain Map')
    fig.savefig(basePath + 'Map.pdf')

    plt.show()
    os.system('tput bel')
    os.system('say "Program Finished"')

