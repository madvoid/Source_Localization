# -------------------------------------------------------------------------------------------------
# Name: Ackley.py
#
# Author: Nipun Gunawardena
#
# Purpose: Test PSO "library" on Ackley function
# -------------------------------------------------------------------------------------------------

import sys
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d

sys.path.insert(0, '../')
from PSO import *




@np.vectorize
def ackley(x1, x2, x3):
    return -20 * np.exp(-0.2 * np.sqrt((1/3) * (x1 ** 2 + x2 ** 2 + x3 ** 2))) - np.exp(
        (1/3) * ( np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2) + np.cos(2 * np.pi * x3) )) + 20 + np.exp(1)




if __name__ == "__main__":

    # Create save path
    basePath = '../Results/Ackley3D/Ackley_3D_'

    # Create domain
    xMin = -32
    xMax = 32
    yMin = -30
    yMax = 30
    zMin = -31
    zMax = 31
    xPoints = 108
    yPoints = 96
    zPoints = 48
    x = np.linspace(xMin, xMax, num=xPoints)
    y = np.linspace(yMin, yMax, num=yPoints)
    z = np.linspace(zMin, zMax, num=zPoints)
    (X, Y, Z) = np.meshgrid(x, y, z, indexing='xy')  # Create meshgrid
    C = ackley(X, Y, Z)  # Create discrete Ackley function

    # Create domain instance
    minIdx = np.unravel_index(np.argmin(C), C.shape)
    AckleyDomain = DomainInfo([xMin, yMin, zMin], [xMax, yMax, zMax],
                              [(xMax - xMin) / (xPoints - 1), (yMax - yMin) / (yPoints - 1), (zMax - zMin) / (zPoints - 1)], [xPoints, yPoints, zPoints], 3600,
                              900, 4, [X[minIdx], Y[minIdx], Z[minIdx]])

    # Initialize and run PSO Algorithm
    AckleyPSO = PSO(C, AckleyDomain, numberParticles=10)
    AckleyPSO.run()

    # Plot "built-in" plots
    fig = AckleyPSO.plotConvergence()
    fig.savefig(basePath+'Convergence.pdf')

    # 3D Path Plot
    dVal = 6    # Downsample points by 6 to make plot readable
    Xnew = reslice3D(X, dVal)
    Ynew = reslice3D(Y, dVal)
    Znew = reslice3D(Z, dVal)
    Cnew = rebin(np.exp(C), dVal)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set(xlim=(xMin, xMax), ylim=(yMin, yMax), zlim=(zMin, zMax))
    ax.set_title('Best Location Convergence')
    ax.scatter(X[minIdx], Y[minIdx], Z[minIdx], c='k', marker='*', s=50)    # Actual best position
    ax.plot(AckleyPSO.bestPositionHistory[:,0], AckleyPSO.bestPositionHistory[:,1], AckleyPSO.bestPositionHistory[:,2], zdir='z', c='r', linestyle='-') # Best path
    ax.scatter(Xnew.flatten(), Ynew.flatten(), Znew.flatten(), c=Cnew.flatten(), alpha=0.075, cmap='viridis', marker='s', s=150)    # 3D scatter
    fig.savefig(basePath+'Best_Path.pdf')

    # Animated plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set(xlim=(xMin, xMax), ylim=(yMin, yMax), zlim=(zMin, zMax))
    ax.set_title('Live Convergence')
    ax.scatter(X[minIdx], Y[minIdx], Z[minIdx], c='k', marker='*', s=50)    # Actual best position
    ax.scatter(Xnew.flatten(), Ynew.flatten(), Znew.flatten(), c=Cnew.flatten(), alpha=0.075, cmap='viridis', marker='s', s=150)
    dots, = ax.plot(*AckleyPSO.getCurrentPoints(0).T, 'r.')     # https://stackoverflow.com/a/27047043
    stopPoint = np.argmin(AckleyPSO.bestFitnessHistory) + 15

    def animate(i):
        # https://stackoverflow.com/a/41609238
        newPos = AckleyPSO.getCurrentPoints(i)
        dots.set_data(newPos[:,0], newPos[:,1])
        dots.set_3d_properties(newPos[:,2])
        return dots,

    anim = FuncAnimation(fig, animate, interval=150, frames=stopPoint, blit=True)
    anim.save(basePath+'.mp4', extra_args=['-vcodec', 'libx264'])
    # # anim.save('Ackley.gif', writer='imagemagick')

    # Finish up
    plt.show()
