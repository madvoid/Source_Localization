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
    Simple1PSO = PSO(C, Simple1Domain, numberParticles=50, maskArray=B)
    Simple1PSO.run()

    # Plot "built-in" plots
    fig = Simple1PSO.plotConvergence()
    fig.savefig(basePath + 'Convergence.pdf')

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

    C_Plot_2d = np.mean(C_Plot, 2)

    # Plot 2D representation of best points
    fig, ax = plt.subplots()
    ax.pcolormesh(X[:, :, 0], Y[:, :, 0], C_Plot_2d, cmap='viridis', edgecolor='none')
    ax.plot(Simple1PSO.bestPositionHistory[:, 0], Simple1PSO.bestPositionHistory[:, 1], color='r', linestyle=':',
            marker='.')
    ax.scatter(sLocX, sLocY, c='k', marker='*', s=10)  # Actual best position
    ax.set_title('Best Location Convergence')
    fig.savefig(basePath + 'Best_Path.pdf')

    # Animated plot
    fig, ax = plt.subplots()
    ax.set(xlim=(xMin, xMax), ylim=(yMin, yMax))
    ax.set_title('Live Convergence')
    ax.pcolormesh(X[:, :, 0], Y[:, :, 0], C_Plot_2d, cmap='viridis', edgecolor='none')
    ax.scatter(sLocX, sLocY, c='k', marker='*', s=50)  # Actual best position
    scat = ax.scatter(*Simple1PSO.getCurrentPoints(0)[:, 0:2].T, c='r', marker='.',
                      s=Simple1PSO.getCurrentPoints(0)[:, 2])
    stopPoint = np.argmin(Simple1PSO.bestFitnessHistory) + 25


    def animate(i):
        scat.set_offsets(Simple1PSO.getCurrentPoints(i)[:, 0:2])
        scat._sizes = Simple1PSO.getCurrentPoints(i)[:, 2]
        return scat,


    anim = FuncAnimation(fig, animate, interval=150, frames=stopPoint, blit=True)
    anim.save(basePath + '.mp4', extra_args=['-vcodec', 'libx264'])
    # anim.save('Simple1.gif', writer='imagemagick')

    plt.show()
