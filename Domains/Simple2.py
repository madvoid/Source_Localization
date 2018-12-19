# -------------------------------------------------------------------------------------------------
# Name: Simple2.py
#
# Author: Nipun Gunawardena
#
# Purpose: Test PSO on Simple2 QUIC simulation
# -------------------------------------------------------------------------------------------------

import sys
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d

sys.path.insert(0, '../')
from PSO import *
from MatRead import readQUICMat

if __name__ == "__main__":
    # Create save path
    basePath = '../Results/Simple2/Simple2_'

    # Retrieve domain and data
    Simple2Domain, X, Y, Z, B, C, C_Plot = readQUICMat('../QUIC Data/Simple2/Data.mat')

    # Initialize and run PSO Algorithm
    Simple2PSO = PSO(C, Simple2Domain, numberParticles=50, maskArray=B)
    Simple2PSO.run(checkNeighborhood=True)

    # Plot "built-in" plots
    fig = Simple2PSO.plotConvergence()
    fig.savefig(basePath + 'Convergence.pdf')
    fig = Simple2PSO.plotDistanceNorm()
    fig.savefig(basePath + 'DistNorm.pdf')

    # Make plotting variables
    xMin = Simple2Domain.minLims[0]
    xMax = Simple2Domain.maxLims[0]
    yMin = Simple2Domain.minLims[1]
    yMax = Simple2Domain.maxLims[1]
    zMin = Simple2Domain.minLims[2]
    zMax = Simple2Domain.maxLims[2]
    sLocX = Simple2Domain.sourceLoc[0]
    sLocY = Simple2Domain.sourceLoc[1]
    sLocZ = Simple2Domain.sourceLoc[2]

    # Create 2d representation of 3d concentration
    C_Plot_2d = np.mean(C_Plot[0], 2)

    # Create Colors
    concentrationMap = 'inferno'
    lineMap = 'g'
    zMap = 'winter'
    sourceMap = 'g'

    # Plot 2D representation of best points
    fig, ax = plt.subplots()
    ax.pcolormesh(X[:, :, 0], Y[:, :, 0], C_Plot_2d, cmap=concentrationMap, edgecolor='none')
    ax.plot(Simple2PSO.bestPositionHistory[:, 0], Simple2PSO.bestPositionHistory[:, 1], color=lineMap, linestyle=':',
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
    scat = ax.scatter(*Simple2PSO.getCurrentPoints(0)[:, 0:2].T, c=normalize(Simple2PSO.getCurrentPoints(0)[:, 2]), marker='.', cmap=zMap)
    stopPoint = np.argmin(Simple2PSO.bestFitnessHistory) + 25
    plt.show()

    scatCol = plt.get_cmap(zMap)

    def animate(i):
        scat.set_offsets(Simple2PSO.getCurrentPoints(i)[:, 0:2])
        # scat._sizes = Simple2PSO.getCurrentPoints(i)[:, 2]
        scat.set_color(scatCol(normalize(Simple2PSO.getCurrentPoints(i)[:, 2])))
        ax.set_title(f'Live Convergence :: Iteration {i}')
        return scat,


    anim = FuncAnimation(fig, animate, interval=150, frames=stopPoint, blit=True)
    anim.save(basePath + '.mp4', extra_args=['-vcodec', 'libx264'])
    # anim.save('Simple2.gif', writer='imagemagick')

    plt.show()