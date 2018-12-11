# -------------------------------------------------------------------------------------------------
# Name: Simple3.py
#
# Author: Nipun Gunawardena
#
# Purpose: Test PSO on Simple3 QUIC simulation
# -------------------------------------------------------------------------------------------------

import sys
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d

sys.path.insert(0, '../')
from PSO import *
from MatRead import readQUICMat

if __name__ == "__main__":
    # Create save path
    basePath = '../Results/Simple3/Simple3_'

    # Create Colors
    concentrationMap = 'plasma'
    lineMap = 'g'
    zMap = 'Greens'

    # Retrieve domain and data
    Simple3Domain, X, Y, Z, B, C, C_Plot = readQUICMat('../QUIC Data/Simple3/Data.mat')

    # Initialize and run PSO Algorithm
    Simple3PSO = PSO(C, Simple3Domain, numberParticles=25, maximumIterations=300)
    Simple3PSO.run(checkNeighborhood=True)

    # Plot "built-in" plots
    fig = Simple3PSO.plotConvergence()
    fig.savefig(basePath + 'Convergence.pdf')
    fig = Simple3PSO.plotDistanceNorm()
    fig.savefig(basePath + 'DistNorm.pdf')

    # Make plotting variables
    xMin = Simple3Domain.minLims[0]
    xMax = Simple3Domain.maxLims[0]
    yMin = Simple3Domain.minLims[1]
    yMax = Simple3Domain.maxLims[1]
    zMin = Simple3Domain.minLims[2]
    zMax = Simple3Domain.maxLims[2]
    sLocX = Simple3Domain.sourceLoc[0]
    sLocY = Simple3Domain.sourceLoc[1]
    sLocZ = Simple3Domain.sourceLoc[2]

    # Create 2d representation of 3d concentration
    C_Plot_2d = np.mean(C_Plot, 2)

    # Create Colors
    concentrationMap = 'inferno'
    lineMap = 'g'
    zMap = 'winter'
    sourceMap = 'g'

    # Plot 2D representation of best points
    fig, ax = plt.subplots()
    ax.pcolormesh(X[:, :, 0], Y[:, :, 0], C_Plot_2d, cmap=concentrationMap, edgecolor='none')
    ax.plot(Simple3PSO.bestPositionHistory[:, 0], Simple3PSO.bestPositionHistory[:, 1], color=lineMap, linestyle=':',
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
    ax.pcolormesh(X[:, :, 0], Y[:, :, 0], C_Plot_2d, cmap=concentrationMap, edgecolor='none')
    ax.scatter(sLocX, sLocY, c=sourceMap, marker='*', s=50)  # Actual best position
    scat = ax.scatter(*Simple3PSO.getCurrentPoints(0)[:, 0:2].T, c=normalize(Simple3PSO.getCurrentPoints(0)[:, 2]), marker='.', cmap=zMap)
    stopPoint = np.argmin(Simple3PSO.bestFitnessHistory) + 25
    plt.show()

    scatCol = plt.get_cmap(zMap)

    def animate(i):
        scat.set_offsets(Simple3PSO.getCurrentPoints(i)[:, 0:2])
        # scat._sizes = Simple3PSO.getCurrentPoints(i)[:, 2]
        scat.set_color(scatCol(normalize(Simple3PSO.getCurrentPoints(i)[:, 2])))
        return scat,


    anim = FuncAnimation(fig, animate, interval=150, frames=stopPoint, blit=True)
    anim.save(basePath + '.mp4', extra_args=['-vcodec', 'libx264'])
    # anim.save('Simple1.gif', writer='imagemagick')

    plt.show()
