# -------------------------------------------------------------------------------------------------
# Name: Simple1.py
#
# Author: Nipun Gunawardena
#
# Purpose: Test PSO on Simple1 QUIC simulation
# -------------------------------------------------------------------------------------------------

import sys
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d

sys.path.insert(0, '../')
from PSO import *
from MatRead import readQUICMat

if __name__ == "__main__":
    # Create save path
    basePath = '../Results/Simple1/Simple1_'

    # Retrieve domain and data
    Simple1Domain, X, Y, Z, B, C, C_Plot = readQUICMat('../QUIC Data/Simple1/Data.mat')

    # Initialize and run PSO Algorithm
    Simple1PSO = PSO(C, Simple1Domain, numberParticles=50, maskArray=B, maximumIterations=300)
    Simple1PSO.run(checkNeighborhood=True)

    # Plot "built-in" plots
    fig = Simple1PSO.plotConvergence()
    fig.savefig(basePath + 'Convergence.pdf')
    fig = Simple1PSO.plotDistanceNorm()
    fig.savefig(basePath + 'DistNorm.pdf')

    # Make plotting variables
    xMin = Simple1Domain.minLims[0]
    xMax = Simple1Domain.maxLims[0]
    yMin = Simple1Domain.minLims[1]
    yMax = Simple1Domain.maxLims[1]
    zMin = Simple1Domain.minLims[2]
    zMax = Simple1Domain.maxLims[2]
    sLocX = Simple1Domain.sourceLoc[0]
    sLocY = Simple1Domain.sourceLoc[1]
    sLocZ = Simple1Domain.sourceLoc[2]

    # Create 2d representation of 3d concentration, use log scale to highlight differences
    with np.errstate(divide='ignore'):
        C_Plot_2d = np.log(np.mean(C_Plot[0], 2))
    C_Plot_2d[C_Plot_2d == -np.inf] = 0
    C_Plot_2d *= -1

    # Create Colors
    concentrationMap = 'inferno'
    lineMap = 'g'
    zMap = 'winter'
    sourceMap = 'g'

    # Plot 2D representation of best points
    fig, ax = plt.subplots()
    ax.pcolormesh(X[:, :, 0], Y[:, :, 0], C_Plot_2d, cmap=concentrationMap, edgecolor='none')
    ax.plot(Simple1PSO.bestPositionHistory[:, 0], Simple1PSO.bestPositionHistory[:, 1], color=lineMap, linestyle=':',
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
    scat = ax.scatter(*Simple1PSO.getCurrentPoints(0)[:, 0:2].T, c=normalize(Simple1PSO.getCurrentPoints(0)[:, 2]), marker='.', cmap=zMap)
    stopPoint = np.argmin(Simple1PSO.bestFitnessHistory) + 25
    plt.show()

    scatCol = plt.get_cmap(zMap)

    def animate(i):
        scat.set_offsets(Simple1PSO.getCurrentPoints(i)[:, 0:2])
        # scat._sizes = Simple1PSO.getCurrentPoints(i)[:, 2]
        scat.set_color(scatCol(normalize(Simple1PSO.getCurrentPoints(i)[:, 2])))
        ax.set_title(f'Live Convergence :: Iteration {i}')
        return scat,


    anim = FuncAnimation(fig, animate, interval=150, frames=stopPoint, blit=True)
    anim.save(basePath + '.mp4', extra_args=['-vcodec', 'libx264'])
    # anim.save('Simple1.gif', writer='imagemagick')

    plt.show()
