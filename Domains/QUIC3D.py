# -------------------------------------------------------------------------------------------------
# Name: QUIC3D.py
#
# Author: Nipun Gunawardena
#
# Purpose: Test PSO on a QUIC simulation
# -------------------------------------------------------------------------------------------------

import sys
import numpy.ma as ma
import matplotlib.colors as colors
from copy import copy
from matplotlib.animation import FuncAnimation

sys.path.insert(0, '../')
from PSO import *
from MatRead import readQUICMat

if __name__ == "__main__":
    # Create save path
    # baseName = 'OKC'
    # baseName = 'Quad_Corner'
    baseName = 'Quad_Center'
    # baseName = 'Quad_Edge'
    # baseName = 'Simple3'
    # baseName = 'Simple_R'
    basePath = '../Results/' + baseName + '/' + baseName + '_'
    timeVaryingFlag = True

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
    frameSave = np.round(np.linspace(0, stopPoint-stopBuf/2, num=6)).astype(int)
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
    C_Plot_2d = flattenPlotQuic(timeStartIndex, C_Plot, log=False)
    C_Mask = ma.masked_array(np.zeros(C_Plot_2d.shape), mask=~np.isnan(C_Plot_2d))
    cMin = np.nanmin(C_Plot_2d[C_Plot_2d > 0])
    cMax = np.nanmax(C_Plot_2d)

    # Create Colors
    # concentrationMap = 'viridis'
    lineMap = 'b'
    zMap = 'copper'
    sourceMap = 'g'
    concentrationMap = copy(plt.cm.plasma)
    concentrationMap.set_bad('w', 1.0)
    maskMap = 'Paired'


    # Plot 2D representation of best points
    fig, ax = plt.subplots()
    fig.set_size_inches(6.4, 5.6)
    ax.pcolormesh(X[:, :, 0], Y[:, :, 0], C_Mask, cmap='gray')
    conc = ax.pcolormesh(X[:, :, 0], Y[:, :, 0], C_Plot_2d, cmap=concentrationMap, edgecolor='none', norm=colors.LogNorm(vmin=cMin, vmax=cMax))
    manc = ax.pcolormesh(X[:, :, 0], Y[:, :, 0], C_Mask, cmap=maskMap)
    ax.plot(quicPSO.bestPositionHistory[0:stopPoint, 0], quicPSO.bestPositionHistory[0:stopPoint, 1], color=lineMap, linestyle=':',
            marker='.')
    ax.scatter(sLocX, sLocY, c=sourceMap, marker='*', s=50)  # Actual best position
    cbar = plt.colorbar(conc, orientation='horizontal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Best Location Convergence')
    fig.savefig(basePath + 'Best_Path.pdf')
    plt.show()

    # Normalizing function for scatterplot colors
    def normalize(arr):
        return (arr - zMin) / (zMax - zMin)

    # Animated plot
    fig, ax = plt.subplots()
    fig.set_size_inches(6.4, 5.6)
    ax.set(xlim=(xMin, xMax), ylim=(yMin, yMax))
    ax.set_title('Live Convergence')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    pcol = ax.pcolormesh(X[:, :, 0], Y[:, :, 0], C_Plot_2d, cmap=concentrationMap, edgecolor='none', norm=colors.LogNorm(vmin=cMin, vmax=cMax))
    manc = ax.pcolormesh(X[:, :, 0], Y[:, :, 0], C_Mask, cmap=maskMap)
    cbar = plt.colorbar(pcol, orientation='horizontal')
    cbar.set_label('Concentration ($g/m^3$)')
    ax.scatter(sLocX, sLocY, c=sourceMap, marker='*', s=50)  # Actual best position
    scat = ax.scatter(*quicPSO.getCurrentPoints(0)[:, 0:2].T, c=normalize(quicPSO.getCurrentPoints(0)[:, 2]), marker='.', cmap=zMap, s=75)
    cbar2 = plt.colorbar(scat)
    cbar2.set_label('UAV Elevation (m)')
    cbarTicks = np.linspace(0, 1, 10)
    cbar2.set_ticks(cbarTicks)
    cbarTicks = np.linspace(zMin, zMax, 10)
    cbarTickLabels = [f'{int(i):d}' for i in cbarTicks]
    cbar2.set_ticklabels(cbarTickLabels)
    scatCol = plt.get_cmap(zMap)                            # To change colors in animate
    if timeVaryingFlag:
        simStopPoint = quicPSO.stopIter
    # plt.show()    # Weird, if this is uncommented the figure frames do not save (line 158) for some reason

    def animate(i):
        global currentPeriod
        global frameCount
        scat.set_offsets(quicPSO.getCurrentPoints(i)[:, 0:2])   # New x,y positions
        scat.set_color(scatCol(normalize(quicPSO.getCurrentPoints(i)[:, 2])))   # New elevation
        if timeVaryingFlag:     # Change cost function
            currentTime = i*deltaT
            if currentTime >= timeChanges[currentPeriod] and (i < simStopPoint):
                currentPeriod += 1
                C_Plot_2d = flattenPlotQuic(currentPeriod, C_Plot, log=False)
                ax.pcolormesh(X[:, :, 0], Y[:, :, 0], C_Plot_2d, cmap=concentrationMap, edgecolor='none', norm=colors.LogNorm(vmin=cMin, vmax=cMax))
                ax.pcolormesh(X[:, :, 0], Y[:, :, 0], C_Mask, cmap=maskMap)
                ax.scatter(sLocX, sLocY, c=sourceMap, marker='*', s=50)  # Actual best position
                scat.set_zorder(10)
            ax.set_title(f'Live Convergence :: Num Particles = {numParticles} :: Iteration {i} :: Time {currentTime+600} s')
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
