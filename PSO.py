# -------------------------------------------------------------------------------------------------
# Name: PSO.py
#
# Author: Nipun Gunawardena
#
# Purpose: Create PSO library for use with discrete grids. Running by itself will show a demo using
#          Ackley function.
# -------------------------------------------------------------------------------------------------

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


class DomainInfo:
    def __init__(self, lowerLimitArr, upperLimitArr, spacingArr, cellCountArr, duration, averageTime,
                 numPeriods, sourceLocation):
        """
        Initialize DomainInfo class with values

        :param lowerLimitArr: The lower limits of the domain xMin, yMin, zMin
        :param upperLimitArr: The upper limits of the domain xMax, yMax, zMax
        :param spacingArr: The grid spacing of the array dx, dy, dz
        :param cellCountArr: The number of cells in each dimension
        :param duration: The total simulation time
        :param averageTime: The time for each averaging period within the duration
        :param numPeriods: The number of periods within a duration
        :param sourceLocation: The location of the source
        """
        self.minLims = np.array(lowerLimitArr)  # (xMin, yMin, zMin)
        self.maxLims = np.array(upperLimitArr)  # (xMax, yMax, zMax)
        self.ds = np.array(spacingArr)  # (dx, dy, dz)
        self.cellCounts = np.array(cellCountArr)  # (xCells, yCells, zCells) ! Need to be careful with this one
        self.duration = duration  # Seconds
        self.avgTime = averageTime  # Seconds
        self.numPeriods = numPeriods
        self.sourceLoc = np.array(sourceLocation)  # Not in cell units, in x,y,z units

        # Derived variables
        self.dimension = len(self.minLims)  # Dimension of problem. Should be 2 or 3 in most cases

    def __repr__(self):
        return f'Domain from {self.minLims} to {self.maxLims} with {self.ds} spacing'


class Particle:
    def __init__(self, domainClass):
        """
        Initialize particle for PSO

        :param domainClass: DomainInfo instance about current domain
        """
        self.domain = domainClass  # Current domain
        self.position = np.random.uniform(self.domain.minLims, self.domain.maxLims)  # Initial position
        self.positionIndex = self.getIndex(self.position)  # Index of initial position
        self.velocity = np.random.random_sample(
            self.domain.dimension) * 0.1  # Initial velocity, multiply by 0.1 so it doesn't explode
        self.pBestLoc = self.position  # Best location of particle

    def getIndex(self, point):
        """
        Get matrix index of a given point in the domain

        :param point: Point you need index of in x,y,z coords
        :return: The index of the input point
        """
        return ((point - self.domain.minLims) / self.domain.ds).astype(int)

    def getFitness(self, costArray):
        pass


class PSO:
    def __init__(self, costArray, numberParticles, maximumIterations, domainClass):
        self.maxIter = maximumIterations
        self.numParticles = numberParticles
        self.costArray = costArray
        self.particles = [Particle(domainClass) for i in range(self.numParticles)]


if __name__ == "__main__":
    @np.vectorize
    def ackley(x1, x2):
        return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1 ** 2 + x2 ** 2))) - np.exp(
            0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + 20 + np.exp(1)


    # Create domain
    xMin = -32
    xMax = 32
    yMin = -25
    yMax = 25
    xPoints = 105
    yPoints = 100
    x = np.linspace(xMin, xMax, num=xPoints)
    y = np.linspace(yMin, yMax, num=yPoints)
    (X, Y) = np.meshgrid(x, y, indexing='xy')  # Create meshgrid
    C = ackley(X, Y)  # Create discrete Ackley function

    # Create domain instance
    minIdx = np.unravel_index(np.argmin(C), C.shape)
    AckleyDomain = DomainInfo([xMin, yMin], [xMax, yMax],
                              [(xMax - xMin) / (xPoints - 1), (yMax - yMin) / (yPoints - 1)], [xPoints, yPoints], 3600,
                              900, 4, [X[minIdx], Y[minIdx]])

    # Plot Ackley function
    fig, ax = plt.subplots()
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(X, Y, C, cmap='viridis', edgecolor='none')
    ax.pcolormesh(X, Y, C)
    plt.show()

    # Initialize PSO Algorithm
    AckleyPSO = PSO(C, 25, 1000, AckleyDomain)
    pass
