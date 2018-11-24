# -------------------------------------------------------------------------------------------------
# Name: Eggholder.py
#
# Author: Nipun Gunawardena
#
# Purpose: Test PSO "library" on Eggholder function
#
# Notes: x* = (512, 404.2319), f(x*) = -959.6407
# -------------------------------------------------------------------------------------------------

import sys
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d

sys.path.insert(0, '../')
from PSO import *




@np.vectorize
def Eggholder(x1, x2):
    return -(x2+47)*np.sin(np.sqrt(np.abs(x2 + x1/2 + 47))) - x1*np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))




if __name__ == "__main__":

    # Create save path
    basePath = '../Results/Eggholder/Eggholder_'

    # Create domain
    xMin = -512
    xMax = 512
    yMin = -512
    yMax = 512
    xPoints = 130
    yPoints = 130
    x = np.linspace(xMin, xMax, num=xPoints)
    y = np.linspace(yMin, yMax, num=yPoints)
    (X, Y) = np.meshgrid(x, y, indexing='xy')  # Create meshgrid
    C = Eggholder(X, Y)  # Create discrete Eggholder function

    # Create domain instance
    minIdx = np.unravel_index(np.argmin(C), C.shape)
    EggholderDomain = DomainInfo([xMin, yMin], [xMax, yMax],
                              [(xMax - xMin) / (xPoints - 1), (yMax - yMin) / (yPoints - 1)], [xPoints, yPoints], 3600,
                              900, 4, [X[minIdx], Y[minIdx]])

    # Initialize and run PSO Algorithm
    maxIter = 5000
    EggholderPSO = PSO(C, EggholderDomain, numberParticles=50, maximumIterations=maxIter)
    EggholderPSO.run()

    # Plot "built-in" plots
    fig = EggholderPSO.plotConvergence()
    fig.savefig(basePath+'Convergence.pdf')

    # Plot 2D representation of best points
    fig, ax = plt.subplots()
    ax.pcolormesh(X, Y, C, cmap='viridis', edgecolor='none')
    ax.scatter(X[minIdx], Y[minIdx], c='k', marker='*', s=50)  # Actual best position
    ax.plot(EggholderPSO.bestPositionHistory[:, 0], EggholderPSO.bestPositionHistory[:, 1], color='r', linestyle=':', marker='.')
    ax.set_title('Best Location Convergence')
    fig.savefig(basePath+'Best_Path.pdf')

    # Plot 3D representation of function
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, C, cmap='viridis', edgecolor=(0,0,0,0.25))
    ax.set_title('3D Surface')
    fig.savefig(basePath+'Surface.pdf')

    # Animated plot
    fig, ax = plt.subplots()
    ax.set(xlim=(xMin, xMax), ylim=(yMin, yMax))
    ax.set_title('Live Convergence')
    ax.pcolormesh(X, Y, C, cmap='viridis', edgecolor='none')
    ax.scatter(X[minIdx], Y[minIdx], c='k', marker='*', s=50)  # Actual best position
    dots, = ax.plot(*EggholderPSO.getCurrentPoints(0).T, 'r.')
    stopPoint = np.argmin(EggholderPSO.bestFitnessHistory) + 15

    def animate(i):
        dots.set_data(*EggholderPSO.getCurrentPoints(i).T)
        return dots,

    anim = FuncAnimation(fig, animate, interval=150, frames=maxIter//5, blit=True)
    anim.save(basePath+'.mp4', extra_args=['-vcodec', 'libx264'])
    # anim.save('Eggholder.gif', writer='imagemagick')

    # Finish up
    plt.show()
