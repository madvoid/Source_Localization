# -------------------------------------------------------------------------------------------------
# Name: PSO.py
#
# Author: Nipun Gunawardena
#
# Purpose: Create PSO library for use with discrete grids. Running by itself will show a demo using
#          Ackley function.
# -------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt




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
        self.position = np.random.uniform(self.domain.minLims, self.domain.maxLims)  # Initial position, 1xnDim array
        self.positionIndex = self.getIndex(
            self.position)  # Index of initial position, 1xnDim tuple (is a tuple for accessing elements)
        self.velocity = np.random.random_sample(
            self.domain.dimension) * 0.1  # Initial velocity, multiply by 0.1 so no explode, same size as position
        self.pBestPos = self.position  # Best location of particle
        self.currentFitness = None     # Current fitness of particle

    def getIndex(self, point):
        """
        Get matrix index of a given point in the domain

        :param point: Point you need index of in x,y,z coords
        :return: The index of the input point. This is to be used with the meshgrid matrices, not the 1-dim matrices!
        """
        indexRaw = ((point - self.domain.minLims) / self.domain.ds).astype(int)
        indexRaw[0], indexRaw[1] = indexRaw[1], indexRaw[0]  # Switch first and second spots since x,y need to be flipped when accessing 2D array since x corresponds to columns and y corresponds to rows
        return tuple(indexRaw)

    def updateFitness(self, costArray):
        """
        Update fitness of current particle by accessing cost function array

        :param costArray: Cost function array, will be passed in
        :return: current fitness value in case it's needed outside of class
        """
        self.currentFitness = costArray[self.positionIndex]
        return self.currentFitness




class PSO:
    def __init__(self, costArray, domainClass, numberParticles=25, maximumIterations=1000):
        self.costArray = costArray          # Array of values of cost function
        self.domain = domainClass           # DomainInfo instance about current domain
        self.maxIter = maximumIterations    # Maximum number of iterations
        self.numParticles = numberParticles # Number of particles
        self.particles = [Particle(domainClass) for _ in range(self.numParticles)]  # Initialize particles
        self.bestParticle = None            # Don't get best particle yet
        for particle in self.particles:     # Find cost of all particles
            particle.updateFitness(self.costArray)
        self.getBestParticle()              # Find best particle

    def getBestParticle(self):
        bestParticleIndex = np.argmin([particle.currentFitness for particle in self.particles])
        self.bestParticle = self.particles[bestParticleIndex]




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
    AckleyPSO = PSO(C, AckleyDomain)
    pass
