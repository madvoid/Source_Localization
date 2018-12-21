# -------------------------------------------------------------------------------------------------
# Name: OKC.py
#
# Author: Nipun Gunawardena
#
# Purpose: Test PSO on QUIC simulation
# -------------------------------------------------------------------------------------------------

import sys
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d

sys.path.insert(0, '../')
from PSO import *
from MatRead import readQUICMat

if __name__ == "__main__":
    # Create save path
    basePath = '../Results/OKC/OKC_'

    # Retrieve domain and data
    OKCDomain, X, Y, Z, B, C, C_Plot = readQUICMat('../QUIC Data/OKC/Data.mat')

    # Initialize and run PSO Algorithm
    OKCPSO = PSO(C, OKCDomain, numberParticles=50, maskArray=B, maximumIterations=300)
    OKCPSO.run(checkNeighborhood=True)

    # Plot "built-in" plots
    fig = OKCPSO.plotConvergence()
    fig.savefig(basePath + 'Convergence.pdf')
    fig = OKCPSO.plotDistanceNorm()
    fig.savefig(basePath + 'DistNorm.pdf')

    # Make plotting variables
    xMin = OKCDomain.minLims[0]
    xMax = OKCDomain.maxLims[0]
    yMin = OKCDomain.minLims[1]
    yMax = OKCDomain.maxLims[1]
    zMin = OKCDomain.minLims[2]
    zMax = OKCDomain.maxLims[2]
    sLocX = OKCDomain.sourceLoc[0]
    sLocY = OKCDomain.sourceLoc[1]
    sLocZ = OKCDomain.sourceLoc[2]

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
    ax.plot(OKCPSO.bestPositionHistory[:, 0], OKCPSO.bestPositionHistory[:, 1], color=lineMap, linestyle=':',
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
    scat = ax.scatter(*OKCPSO.getCurrentPoints(0)[:, 0:2].T, c=normalize(OKCPSO.getCurrentPoints(0)[:, 2]), marker='.', cmap=zMap)
    stopPoint = np.argmin(OKCPSO.bestFitnessHistory) + 25
    plt.show()

    scatCol = plt.get_cmap(zMap)

    def animate(i):
        scat.set_offsets(OKCPSO.getCurrentPoints(i)[:, 0:2])
        # scat._sizes = OKCPSO.getCurrentPoints(i)[:, 2]
        scat.set_color(scatCol(normalize(OKCPSO.getCurrentPoints(i)[:, 2])))
        ax.set_title(f'Live Convergence :: Iteration {i}')
        return scat,


    anim = FuncAnimation(fig, animate, interval=150, frames=stopPoint, blit=True)
    anim.save(basePath + '.mp4', extra_args=['-vcodec', 'libx264'])
    # anim.save('OKC.gif', writer='imagemagick')

    plt.show()
