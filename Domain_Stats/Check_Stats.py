# -------------------------------------------------------------------------------------------------
# Name: Check_Stats.py
#
# Author: Nipun Gunawardena
#
# Purpose: Run QUIC run many times to see how well it works with neighborhood/stuck check and without
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
    basePath = '../Results/' + baseName + '/Check Statistics/' + baseName + '_'

    # Retrieve domain and data
    quicDomain, X, Y, Z, B, C, C_Plot = readQUICMat('../QUIC Data/' + baseName + '/Data.mat')

    # Prepare
    numParticles = 30
    numIters = 100
    bestLocsOn = np.zeros((numIters, quicDomain.dimension))
    convItersOn = np.zeros(numIters)
    bestLocsOff = np.zeros((numIters, quicDomain.dimension))
    convItersOff = np.zeros(numIters)

    # Iterate with neighborhood checks and stuck checks
    for i in tqdm(range(numIters)):
        quicPSO = PSO(C, quicDomain, numberParticles=numParticles, maskArray=B, maximumIterations=300)
        quicPSO.run(checkNeighborhood=True, verbose=False, checkStuckParticle=True)
        bestLocsOn[i, :] = quicPSO.bestPositionHistory[-1:, :]
        convItersOn[i] = np.argmin(quicPSO.bestFitnessHistory)

    # Iterate withOUT neighborhood checks and stuck checks
    for i in tqdm(range(numIters)):
        quicPSO = PSO(C, quicDomain, numberParticles=numParticles, maskArray=B, maximumIterations=300)
        quicPSO.run(checkNeighborhood=False, verbose=False, checkStuckParticle=False)
        bestLocsOff[i, :] = quicPSO.bestPositionHistory[-1:, :]
        convItersOff[i] = np.argmin(quicPSO.bestFitnessHistory)

    # Calculate nondimensional distance
    distOn = np.linalg.norm(bestLocsOn[:, :] - quicDomain.sourceLoc, ord=2, axis=1) / np.linalg.norm(quicDomain.maxLims - quicDomain.minLims, ord=2)
    distOff = np.linalg.norm(bestLocsOff[:, :] - quicDomain.sourceLoc, ord=2, axis=1) / np.linalg.norm(quicDomain.maxLims - quicDomain.minLims, ord=2)

    # Prepare CDF for distOn
    distSortOn = np.sort(distOn)
    distSortOff = np.sort(distOff)
    pDist = np.array(range(len(distOn))) / float(len(distOn))

    # Plot and save
    sns.set()
    fig, ax = plt.subplots()
    sns.distplot(convItersOn, kde=False, norm_hist=False, label='Checks On')
    sns.distplot(convItersOff, kde=False, norm_hist=False, label='Checks Off')
    ax.set_title(f'(a)')
    ax.set_xlabel('Convergence Iterations (Limited to 300)')
    ax.legend()
    fig.savefig(basePath + 'Checks_' + 'ConvIters.pdf')

    fig, ax = plt.subplots()
    ax = sns.lineplot(x = distSortOn, y = pDist, ax=ax, label='Checks On')
    ax = sns.lineplot(x = distSortOff, y = pDist, ax=ax, label='Checks Off')
    ax.set_xlabel('deltaS/L')
    ax.set_ylabel('Percent of Runs')
    ax.set_title('(b)')
    ax.legend()
    fig.savefig(basePath + 'Checks_' + 'FoundLocs.pdf')

    fig, ax = plt.subplots()
    plt.semilogx(distSortOn, pDist)
    plt.semilogx(distSortOff, pDist)
    ax.set_xlabel('deltaS/L')
    ax.set_ylabel('Percent of Runs')
    ax.set_title('(c)')
    ax.legend(['Checks On', 'Checks Off'])
    fig.savefig(basePath + 'Checks_' + 'FoundLocsLog.pdf')

    plt.show()
