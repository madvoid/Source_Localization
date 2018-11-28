# -------------------------------------------------------------------------------------------------
# Name: Bukin6.py
#
# Author: Nipun Gunawardena
#
# Purpose: Test PSO "library" on Bukin6 function
#
# Notes: f(x*) = 0, x* = (-10,1)
# -------------------------------------------------------------------------------------------------

import sys
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d

sys.path.insert(0, '../')
from PSO import *




@np.vectorize
def bukin6(x1, x2):
    return 100*np.sqrt(np.abs(x2 - 0.01*x1**2)) + 0.01*np.abs(x1+10)




if __name__ == "__main__":

    # Create save path
    basePath = '../Results/Bukin6/Bukin6_'

    # Create domain
    xMin = -15
    xMax = -5
    yMin = -3
    yMax = 3
    xPoints = 105
    yPoints = 100
    x = np.linspace(xMin, xMax, num=xPoints)
    y = np.linspace(yMin, yMax, num=yPoints)
    (X, Y) = np.meshgrid(x, y, indexing='xy')  # Create meshgrid
    C = bukin6(X, Y)  # Create discrete Bukin6 function

    # Create domain instance
    minIdx = np.unravel_index(np.argmin(C), C.shape)
    Bukin6Domain = DomainInfo([xMin, yMin], [xMax, yMax],
                              [(xMax - xMin) / (xPoints - 1), (yMax - yMin) / (yPoints - 1)], [xPoints, yPoints], 3600,
                              900, 4, [X[minIdx], Y[minIdx]])

    # Initialize and run PSO Algorithm
    Bukin6PSO = PSO(C, Bukin6Domain, numberParticles=25)
    Bukin6PSO.run()

    # Plot "built-in" plots
    fig = Bukin6PSO.plotConvergence()
    fig.savefig(basePath+'Convergence.pdf')

    # Plot 2D representation of best points
    fig, ax = plt.subplots()
    ax.pcolormesh(X, Y, C, cmap='viridis', edgecolor='none')
    ax.scatter(X[minIdx], Y[minIdx], c='k', marker='*', s=50)  # Actual best position
    ax.plot(Bukin6PSO.bestPositionHistory[:, 0], Bukin6PSO.bestPositionHistory[:, 1], color='r', linestyle=':', marker='.')
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
    dots, = ax.plot(*Bukin6PSO.getCurrentPoints(0).T, 'r.')
    stopPoint = np.argmin(Bukin6PSO.bestFitnessHistory) + 15

    def animate(i):
        dots.set_data(*Bukin6PSO.getCurrentPoints(i).T)
        return dots,

    anim = FuncAnimation(fig, animate, interval=250, frames=stopPoint, blit=True)
    anim.save(basePath+'.mp4', extra_args=['-vcodec', 'libx264'])
    # anim.save('Bukin6.gif', writer='imagemagick')

    # Finish up
    plt.show()
