# -------------------------------------------------------------------------------------------------
# Name: QUIC_Stats.py
#
# Author: Nipun Gunawardena
#
# Purpose: Run QUIC PSO many times to get statistics on how the maximum number of iterations affects performance
# -------------------------------------------------------------------------------------------------

import os
import sys
import seaborn as sns
import numpy.ma as ma
from copy import copy
from tqdm import tqdm
import matplotlib.colors as colors

sys.path.insert(0, '../')
from PSO import *
from MatRead import readQUICMat

if __name__ == "__main__":
    # Create save path
    baseName = 'Quad_Center'
    basePath = '../Results/' + baseName + '/MaxIter Statistics/' + baseName + '_'

    # Retrieve domain and data
    quicDomain, X, Y, Z, B, C, C_Plot = readQUICMat('../QUIC Data/' + baseName + '/Data.mat')

    # Prepare
    maxIterList = [100, 200, 300, 400, 500]
    numIters = 100
    bestLocs = np.zeros((numIters, quicDomain.dimension, len(maxIterList)))
    bestIdx = np.zeros((numIters, quicDomain.dimension, len(maxIterList)))
    bestFitness = np.zeros((numIters, len(maxIterList)))
    convIters = np.zeros((numIters, len(maxIterList)))
    tStart = 4

    # Iterate
    for jIdx, j in tqdm(enumerate(maxIterList)):
        for i in tqdm(range(numIters)):
            quicPSO = PSO(C, quicDomain, numberParticles=30, maskArray=B, maximumIterations=j, tStartIndex=tStart)
            quicPSO.run(checkNeighborhood=True, verbose=False, checkStuckParticle=True)
            bpIter = np.argmin(quicPSO.bestFitnessHistory)
            convIters[i, jIdx] = bpIter
            bestLocs[i,:,jIdx] = quicPSO.bestPositionHistory[bpIter,:]
            bestFitness[i, jIdx] = np.min(quicPSO.bestFitnessHistory)

    # Calculate nondimensional distance
    dist = np.zeros((numIters, len(maxIterList)))
    distSort = np.zeros(dist.shape)
    for jIdx, j in enumerate(maxIterList):
        dist[:,jIdx] = np.linalg.norm(bestLocs[:,:,jIdx] - quicDomain.sourceLoc, ord=2, axis=1) / np.linalg.norm(quicDomain.maxLims - quicDomain.minLims, ord=2)

    # Create 2d representation of 3d concentration, use log scale to highlight differences
    C_Plot_2d = flattenPlotQuic(tStart, C_Plot, log=False)
    C_Mask = ma.masked_array(np.zeros(C_Plot_2d.shape), mask=~np.isnan(C_Plot_2d))
    cMin = np.nanmin(C_Plot_2d[C_Plot_2d > 0])
    cMax = np.nanmax(C_Plot_2d)

    lineMap = 'b'
    zMap = 'copper'
    sourceMap = 'g'
    concentrationMap = copy(plt.cm.plasma)
    concentrationMap.set_bad('w', 1.0)
    maskMap = 'Paired'
    sLocX = quicDomain.sourceLoc[0]
    sLocY = quicDomain.sourceLoc[1]

    # Clean data
    convIters[convIters == 0] = np.nan      # 0 convergence iterations means plume wasn't found
    dist[np.isnan(convIters)] = np.nan      # Change distances to nans to for non-found plumes

    # Function to convert v components to speed
    def vToSpeed(v):
        return np.sqrt(3*(v**2))

    # Prepare CDF for dist
    for jIdx, j in enumerate(maxIterList):
        distSort[:,jIdx] = np.sort(dist[:,jIdx])
    pDist = np.array(range(1, len(dist)+1)) / float(len(dist))

    sns.set()
    fig, ax = plt.subplots()
    for i in range(len(maxIterList)):
        v = ~np.isnan(convIters[:,i])
        sns.distplot(convIters[v,i], kde=False, norm_hist=False, label=f"Maximum Iterations = {maxIterList[i]} ({sum(v)})")
    ax.set_title(f'(a)')
    ax.set_xlabel('Convergence Iterations')
    ax.legend()
    fig.savefig(basePath + 'ConvIters.pdf')

    fig, ax = plt.subplots()
    for i in range(len(maxIterList)):
        v = ~np.isnan(convIters[:, i])
        ax = sns.lineplot(x=distSort[~np.isnan(distSort[:,i]), i], y=np.array(range(1, sum(v)+1)) / float(sum(v)), ax=ax, label=f"Maximum Iterations = {maxIterList[i]} ({sum(v)})")
    ax.set_xlabel('$\Delta S/L$')
    ax.set_ylabel('Percent of Successful Runs')
    ax.set_title(f'(b)')
    ax.legend()
    fig.savefig(basePath + 'FoundLocs.pdf')

    fig, ax = plt.subplots()
    vSum = np.zeros(len(maxIterList))
    for i in range(len(maxIterList)):
        v = ~np.isnan(convIters[:, i])
        vSum[i] = sum(v)
        plt.semilogx(distSort[~np.isnan(distSort[:,i]), i], np.array(range(1, sum(v)+1)) / float(sum(v)))
    ax.set_xlabel('$\Delta S/L$')
    ax.set_ylabel('Percent of Successful Runs')
    ax.set_title(f'(c)')
    ax.legend([f"Maximum Iterations = {i} ({int(vSum[idx])})" for idx, i in enumerate(maxIterList)])
    fig.savefig(basePath + 'FoundLocsLog.pdf')

    # Plot map with no points for easy analysis
    sns.reset_orig()
    sns.reset_defaults()
    fig, ax = plt.subplots()
    fig.set_size_inches(6.4, 5.6)
    ax.pcolormesh(X[:, :, 0], Y[:, :, 0], C_Mask, cmap='gray')
    conc = ax.pcolormesh(X[:, :, 0], Y[:, :, 0], C_Plot_2d, cmap=concentrationMap, edgecolor='none', norm=colors.LogNorm(vmin=cMin, vmax=cMax))
    manc = ax.pcolormesh(X[:, :, 0], Y[:, :, 0], C_Mask, cmap=maskMap)
    ax.scatter(sLocX, sLocY, c=sourceMap, marker='*', s=50)  # Actual best position
    cbar = plt.colorbar(conc, orientation='vertical')
    cbar.set_label('Concentration ($g/m^3$)')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('(e)')
    fig.savefig(basePath + 'Map.pdf')

    plt.show()
    # os.system('tput bel')
    # os.system('say "Program Finished"')

