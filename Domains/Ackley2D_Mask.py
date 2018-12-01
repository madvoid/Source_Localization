# -------------------------------------------------------------------------------------------------
# Name: Ackley2D_Mask.py
#
# Author: Nipun Gunawardena
#
# Purpose: Test PSO "library" on Ackley function with "buildings", i.e. undefined areas
# -------------------------------------------------------------------------------------------------

import sys
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d

sys.path.insert(0, '../')
from PSO import *


@np.vectorize
def ackley(x1, x2):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1 ** 2 + x2 ** 2))) - np.exp(
        0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + 20 + np.exp(1)


if __name__ == "__main__":
    # Create save path
    basePath = '../Results/AckleyMask/AckleyMask_'

    # Create domain
    xMin = -32
    xMax = 32
    yMin = -30
    yMax = 30
    xPoints = 108
    yPoints = 100
    x = np.linspace(xMin, xMax, num=xPoints)
    y = np.linspace(yMin, yMax, num=yPoints)
    (X, Y) = np.meshgrid(x, y, indexing='xy')  # Create meshgrid
    C = ackley(X, Y)  # Create discrete Ackley function

    # Create "building"
    B = np.full(C.shape, False)
    B[70:75, 36:72] = True
    C_Plot = np.copy(C)
    C_Plot[B == True] = np.nan

    # Create domain instance
    minIdx = np.unravel_index(np.argmin(C), C.shape)
    AckleyDomain = DomainInfo([xMin, yMin], [xMax, yMax],
                              [(xMax - xMin) / (xPoints - 1), (yMax - yMin) / (yPoints - 1)], [xPoints, yPoints], 3600,
                              900, 4, [X[minIdx], Y[minIdx]])

    # Initialize PSO Algorithm
    AckleyPSO = PSO(C, AckleyDomain, numberParticles=50, maskArray=B)

    # Plot 2D representation of masked array with particles to see if they are in building
    fig, ax = plt.subplots()
    ax.pcolormesh(X, Y, C_Plot, cmap='viridis', edgecolor='none')
    ax.scatter(X[minIdx], Y[minIdx], c='k', marker='*', s=50)  # Actual best position
    ax.plot(*AckleyPSO.getCurrentPoints(0).T, 'r.')
    ax.set_title('2D Preview')

    # Run PSO Algorithm
    AckleyPSO.run()

    # Plot "built-in" plots
    fig = AckleyPSO.plotConvergence()
    fig.savefig(basePath + 'Convergence.pdf')

    # Plot 2D representation of best points
    fig, ax = plt.subplots()
    ax.pcolormesh(X, Y, C_Plot, cmap='viridis', edgecolor='none')
    ax.plot(AckleyPSO.bestPositionHistory[:, 0], AckleyPSO.bestPositionHistory[:, 1], color='r', linestyle=':',
            marker='.')
    ax.scatter(X[minIdx], Y[minIdx], c='k', marker='*', s=50)  # Actual best position
    ax.set_title('Best Location Convergence')
    fig.savefig(basePath + 'Best_Path.pdf')

    # Animated plot
    fig, ax = plt.subplots()
    ax.set(xlim=(xMin, xMax), ylim=(yMin, yMax))
    ax.set_title('Live Convergence')
    ax.pcolormesh(X, Y, C_Plot, cmap='viridis', edgecolor='none')
    ax.scatter(X[minIdx], Y[minIdx], c='k', marker='*', s=50)  # Actual best position
    dots, = ax.plot(*AckleyPSO.getCurrentPoints(0).T, 'r.')
    stopPoint = np.argmin(AckleyPSO.bestFitnessHistory) + 15


    def animate(i):
        dots.set_data(*AckleyPSO.getCurrentPoints(i).T)
        return dots,


    anim = FuncAnimation(fig, animate, interval=150, frames=stopPoint, blit=True)
    anim.save(basePath + '.mp4', extra_args=['-vcodec', 'libx264'])
    # anim.save('Ackley.gif', writer='imagemagick')

    # Finish up
    plt.show()