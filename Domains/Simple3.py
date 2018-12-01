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

    # Retrieve domain and data
    Simple3Domain, X, Y, Z, B, C, C_Plot = readQUICMat('../QUIC Data/Simple3/Data.mat')

    # Initialize and run PSO Algorithm
    Simple3PSO = PSO(C, Simple3Domain, numberParticles=25)
    Simple3PSO.run()

    # Plot "built-in" plots
    fig = Simple3PSO.plotConvergence()
    fig.savefig(basePath + 'Convergence.pdf')

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

    C_Plot_2d = np.mean(C_Plot, 2)

    # Plot 2D representation of best points
    fig, ax = plt.subplots()
    ax.pcolormesh(X[:, :, 0], Y[:, :, 0], C_Plot_2d, cmap='viridis', edgecolor='none')
    ax.plot(Simple3PSO.bestPositionHistory[:, 0], Simple3PSO.bestPositionHistory[:, 1], color='r', linestyle=':',
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
    scat = ax.scatter(*Simple3PSO.getCurrentPoints(0)[:, 0:2].T, c='r', marker='.',
                      s=Simple3PSO.getCurrentPoints(0)[:, 2])
    stopPoint = np.argmin(Simple3PSO.bestFitnessHistory) + 25


    def animate(i):
        scat.set_offsets(Simple3PSO.getCurrentPoints(i)[:, 0:2])
        scat._sizes = Simple3PSO.getCurrentPoints(i)[:, 2]
        return scat,


    anim = FuncAnimation(fig, animate, interval=150, frames=stopPoint, blit=True)
    anim.save(basePath + '.mp4', extra_args=['-vcodec', 'libx264'])
    # anim.save('Simple3.gif', writer='imagemagick')

    plt.show()
