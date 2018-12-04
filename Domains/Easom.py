# -------------------------------------------------------------------------------------------------
# Name: Easom.py
#
# Author: Nipun Gunawardena
#
# Purpose: Test PSO "library" on Easom function
# -------------------------------------------------------------------------------------------------

import sys
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d

sys.path.insert(0, '../')
from PSO import *




@np.vectorize
def easom(x1, x2):
    return -np.cos(x1)*np.cos(x2)*np.exp(-(x1-np.pi)**2 - (x2-np.pi)**2)




if __name__ == "__main__":

    # Create save path
    basePath = '../Results/Easom/Easom_'

    # Create domain
    xMin = -32
    xMax = 32
    yMin = -30
    yMax = 30
    xPoints = 105
    yPoints = 100
    x = np.linspace(xMin, xMax, num=xPoints)
    y = np.linspace(yMin, yMax, num=yPoints)
    (X, Y) = np.meshgrid(x, y, indexing='xy')  # Create meshgrid
    C = easom(X, Y)  # Create discrete Easom function

    # Create domain instance
    minIdx = np.unravel_index(np.argmin(C), C.shape)
    EasomDomain = DomainInfo([xMin, yMin], [xMax, yMax],
                              [(xMax - xMin) / (xPoints - 1), (yMax - yMin) / (yPoints - 1)], [xPoints, yPoints], 3600,
                              900, 4, [X[minIdx], Y[minIdx]])

    # Initialize and run PSO Algorithm
    EasomPSO = PSO(C, EasomDomain, numberParticles=10)
    EasomPSO.run()

    # Plot "built-in" plots
    fig = EasomPSO.plotConvergence()
    fig.savefig(basePath+'Convergence.pdf')

    # Plot 2D representation of best points
    fig, ax = plt.subplots()
    ax.pcolormesh(X, Y, C, cmap='viridis', edgecolor='none')
    ax.plot(EasomPSO.bestPositionHistory[:, 0], EasomPSO.bestPositionHistory[:, 1], color='r', linestyle=':', marker='.')
    ax.scatter(X[minIdx], Y[minIdx], c='k', marker='*', s=50)  # Actual best position
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
    dots, = ax.plot(*EasomPSO.getCurrentPoints(0).T, 'r.')
    stopPoint = np.argmin(EasomPSO.bestFitnessHistory) + 15

    def animate(i):
        dots.set_data(*EasomPSO.getCurrentPoints(i).T)
        return dots,

    anim = FuncAnimation(fig, animate, interval=150, frames=stopPoint, blit=True)
    anim.save(basePath+'.mp4', extra_args=['-vcodec', 'libx264'])
    # anim.save('Easom.gif', writer='imagemagick')

    # Finish up
    plt.show()
