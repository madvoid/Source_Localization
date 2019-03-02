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
    baseName = 'Simple_R'
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

    # Clean data
    convItersOn[convItersOn == 0] = np.nan      # 0 convergence iterations means plume wasn't found
    distOn[np.isnan(convItersOn)] = np.nan      # Change distances to nans to for non-found plumes

    convItersOff[convItersOff == 0] = np.nan      # 0 convergence iterations means plume wasn't found
    distOff[np.isnan(convItersOff)] = np.nan      # Change distances to nans to for non-found plumes

    # Prepare CDF for distOn
    distSortOn = np.sort(distOn)
    distSortOff = np.sort(distOff)
    pDist = np.array(range(len(distOn))) / float(len(distOn))

    # Plot and save
    sns.set()
    fig, ax = plt.subplots()
    vOn = ~np.isnan(convItersOn)
    vOff = ~np.isnan(convItersOff)
    sns.distplot(convItersOn, kde=False, norm_hist=False, label=f'Checks On ({sum(vOn)})')
    sns.distplot(convItersOff, kde=False, norm_hist=False, label=f'Checks Off ({sum(vOff)})')
    ax.set_title(f'(a)')
    ax.set_xlabel('Convergence Iterations (Limited to 300)')
    ax.legend()
    fig.savefig(basePath + 'Checks_' + 'ConvIters.pdf')

    fig, ax = plt.subplots()
    ax = sns.lineplot(x = distSortOn, y = pDist, ax=ax, label=f'Checks On ({sum(vOn)})')
    ax = sns.lineplot(x = distSortOff, y = pDist, ax=ax, label=f'Checks Off ({sum(vOff)})')
    ax.set_xlabel('$\Delta S/L$')
    ax.set_ylabel('Percent of Runs')
    ax.set_title('(b)')
    ax.legend()
    fig.savefig(basePath + 'Checks_' + 'FoundLocs.pdf')

    fig, ax = plt.subplots()
    plt.semilogx(distSortOn, pDist)
    plt.semilogx(distSortOff, pDist)
    ax.set_xlabel('$\Delta S/L$')
    ax.set_ylabel('Percent of Runs')
    ax.set_title('(c)')
    ax.legend(['Checks On', 'Checks Off'])
    fig.savefig(basePath + 'Checks_' + 'FoundLocsLog.pdf')

    plt.show()
