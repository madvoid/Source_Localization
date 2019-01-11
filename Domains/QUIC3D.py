# -------------------------------------------------------------------------------------------------
# Name: QUIC3D.py
#
# Author: Nipun Gunawardena
#
# Purpose: Test PSO on a QUIC simulation
# -------------------------------------------------------------------------------------------------

import sys
import itertools
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d

sys.path.insert(0, '../')
from PSO import *
from MatRead import readQUICMat

if __name__ == "__main__":
    # Create save path
    # baseName = 'OKC'
    # baseName = 'Quad_Center'
    # baseName = 'Quad_Corner'
    # baseName = 'Quad_Edge'
    baseName = 'Simple3'
    basePath = '../Results/' + baseName + '/' + baseName + '_'
    timeVaryingFlag = False

    # Set start index
    if timeVaryingFlag:
        timeStartIndex = 0
    else:
        timeStartIndex = 0

    # Retrieve domain and data
    quicDomain, X, Y, Z, B, C, C_Plot = readQUICMat('../QUIC Data/' + baseName + '/Data.mat')

    # Initialize PSO Algorithm
    numParticles = 30
    quicPSO = PSO(C, quicDomain, numberParticles=numParticles, maskArray=B, maximumIterations=300, tStartIndex=timeStartIndex)

    # Prepare time-varying vars
    if timeVaryingFlag:
        deltaT = quicPSO.deltaT     # Pulls value from PSO class TODO: Make it input to PSO class
        currentPeriod = 0
        timeChanges = [(i+1)*quicDomain.avgTime for i in range(quicDomain.numPeriods)]

    # Run PSO
    quicPSO.run(checkNeighborhood=True, timeVarying=timeVaryingFlag, checkStuckParticle=True)

    # Calculate points to show
    stopBuf = 50
    stopPoint = np.argmin(quicPSO.bestFitnessHistory) + stopBuf  # When to stop iterations
    frameSave = np.round(np.linspace(0, stopPoint-stopBuf, num=6)).astype(int)
    frameCount = 0

    # Plot "built-in" plots
    fig = quicPSO.plotConvergence(stop=stopPoint)
    fig.savefig(basePath + 'Convergence.pdf')
    fig = quicPSO.plotDistanceNorm()
    fig.savefig(basePath + 'DistNorm.pdf')

    # Make plotting variables
    xMin = quicDomain.minLims[0]
    xMax = quicDomain.maxLims[0]
    yMin = quicDomain.minLims[1]
    yMax = quicDomain.maxLims[1]
    zMin = quicDomain.minLims[2]
    zMax = quicDomain.maxLims[2]
    sLocX = quicDomain.sourceLoc[0]
    sLocY = quicDomain.sourceLoc[1]
    sLocZ = quicDomain.sourceLoc[2]

    # Create 2d representation of 3d concentration, use log scale to highlight differences
    C_Plot_2d = flattenPlotQuic(timeStartIndex, C_Plot)

    # Create Colors
    concentrationMap = 'inferno'
    lineMap = 'g'
    zMap = 'winter'
    sourceMap = 'g'

    # Plot 2D representation of best points
    fig, ax = plt.subplots()
    ax.pcolormesh(X[:, :, 0], Y[:, :, 0], C_Plot_2d, cmap=concentrationMap, edgecolor='none')
    ax.plot(quicPSO.bestPositionHistory[0:stopPoint, 0], quicPSO.bestPositionHistory[0:stopPoint, 1], color=lineMap, linestyle=':',
            marker='.')
    ax.scatter(sLocX, sLocY, c=sourceMap, marker='*', s=10)  # Actual best position
    ax.set_title('Best Location Convergence')
    fig.savefig(basePath + 'Best_Path.pdf')
    plt.show()

    # Normalizing function for scatterplot colors
    def normalize(arr):
        return (arr - zMin) / (zMax - zMin)

    # Animated plot
    fig, ax = plt.subplots()
    ax.set(xlim=(xMin, xMax), ylim=(yMin, yMax))
    ax.set_title('Live Convergence')
    pcol = ax.pcolormesh(X[:, :, 0], Y[:, :, 0], C_Plot_2d, cmap=concentrationMap, edgecolor='none')
    ax.scatter(sLocX, sLocY, c=sourceMap, marker='*', s=50)  # Actual best position
    scat = ax.scatter(*quicPSO.getCurrentPoints(0)[:, 0:2].T, c=normalize(quicPSO.getCurrentPoints(0)[:, 2]), marker='.', cmap=zMap)
    scatCol = plt.get_cmap(zMap)                            # To change colors in animate
    if timeVaryingFlag:
        simStopPoint = quicPSO.stopIter

    def animate(i):
        global currentPeriod
        global frameCount
        scat.set_offsets(quicPSO.getCurrentPoints(i)[:, 0:2])   # New x,y positions
        scat.set_color(scatCol(normalize(quicPSO.getCurrentPoints(i)[:, 2])))   # New elevation
        if timeVaryingFlag:     # Change cost function
            currentTime = i*deltaT
            if currentTime >= timeChanges[currentPeriod] and (i < simStopPoint):
                currentPeriod += 1
                C_Plot_2d = flattenPlotQuic(currentPeriod, C_Plot)
                ax.pcolormesh(X[:, :, 0], Y[:, :, 0], C_Plot_2d, cmap=concentrationMap, edgecolor='none')
                scat.set_zorder(10)
            ax.set_title(f'Live Convergence :: Num Particles = {numParticles} :: Iteration {i} :: Time {currentTime} s')
        else:
            ax.set_title(f'Live Convergence :: Num Particles = {numParticles} :: Iteration {i}')
        if frameCount < len(frameSave):
            if i == frameSave[frameCount]:
                plt.savefig(basePath+'Frame'+str(frameCount)+'.png')
                frameCount += 1
        return scat,

    print("Making Video...")
    anim = FuncAnimation(fig, animate, interval=150, frames=stopPoint, blit=True)
    anim.save(basePath + '.mp4', extra_args=['-vcodec', 'libx264'])
    # anim.save('quic.gif', writer='imagemagick')

    plt.show()
    print("Script Finished")
