# -------------------------------------------------------------------------------------------------
# Name: PSO.py
#
# Author: Nipun Gunawardena
#
# Purpose: PSO library for use with discrete grids
# -------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


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
    def __init__(self, domainClass, costArray, maximumIterations):
        """
        Initialize particle for PSO

        :param domainClass: DomainInfo instance about current domain
        """
        self.domain = domainClass  # Current domain
        self.position = np.random.uniform(self.domain.minLims, self.domain.maxLims)  # Initial position, 1xnDim array
        # self.positionIndex = self.getIndex()  # Index of initial position,
        self.velocity = np.random.random_sample(
            self.domain.dimension) * 0.1  # Initial velocity, multiply by 0.1 so no explode, same size as position
        self.currentFitness = self.updateFitness(costArray)  # Current fitness of particle
        self.pBestPosition = self.position  # Best location of particle
        self.pBestFitness = self.currentFitness
        self.fitnessHistory = np.zeros(maximumIterations)
        self.positionHistory = np.zeros([maximumIterations, self.domain.dimension])
        self.fitnessHistory[0] = self.currentFitness
        self.positionHistory[0, :] = self.position

    def getIndex(self):
        """
        Get matrix index of a given point in the domain

        :return: The index of current position. This is to be used with the meshgrid matrices, not the 1-dim matrices! 1xnDim tuple (is a tuple for accessing elements)
        """
        indexRaw = ((self.position - self.domain.minLims) / self.domain.ds).astype(int)
        indexRaw[0], indexRaw[1] = indexRaw[1], indexRaw[
            0]  # Switch first and second spots since x,y need to be flipped when accessing 2D array since x corresponds to columns and y corresponds to rows
        return tuple(indexRaw)

    def updateFitness(self, costArray):
        """
        Update fitness of current particle by accessing cost function array

        :param costArray: Cost function array, will be passed in
        :return: current fitness value in case it's needed outside of class
        """
        self.currentFitness = costArray[self.getIndex()]
        return self.currentFitness

    def updateHistory(self, currentIteration):
        """
        Update history arrays with fitness and position

        :param currentIteration: Current iteration within PSO algorithm
        :return: None
        """
        self.fitnessHistory[currentIteration] = self.currentFitness
        self.positionHistory[currentIteration, :] = self.position








class PSO:
    def __init__(self, costArray, domainClass, numberParticles=25, maximumIterations=1000):
        self.costArray = costArray  # Array of values of cost function
        self.domain = domainClass  # DomainInfo instance about current domain
        self.maxIter = maximumIterations  # Maximum number of iterations
        self.numParticles = numberParticles  # Number of particles
        self.particles = [Particle(domainClass, costArray, self.maxIter) for _ in range(self.numParticles)]  # Initialize particles
        self.globalBest = self.particles[0]  # Just set best particle to first one for now
        self.globalBestIndex = 0
        self.getGlobalBest()  # Find best particle
        self.bestPositionHistory = np.zeros([self.maxIter, self.domain.dimension])  # Keep track of best position found
        self.bestFitnessHistory = np.zeros(self.maxIter)  # Keep track of best fitness so far
        self.bestPositionHistory[0, :] = self.globalBest.position
        self.bestFitnessHistory[0] = self.globalBest.currentFitness

    def getGlobalBest(self):
        """
        Find best particle out of all particles. Updates best particle as well

        :return: None
        """
        bestParticleIndex = np.argmin([particle.currentFitness for particle in self.particles])
        bestParticle = self.particles[bestParticleIndex]
        if bestParticle.currentFitness < self.globalBest.currentFitness:
            self.globalBestIndex = bestParticleIndex
            self.globalBest = deepcopy(bestParticle)

    def generateVelocity(self, particle):
        """
        Generate new velocity for particle class

        :param particle: Instance of Particle class
        :return: New velocity, ndarray of size 1xnDim. Particle is NOT updated in this function
        """
        # Establish Constants
        c1 = 2
        c2 = 2
        k = 0.05    # Velocity clamping constant
        vMax = k * (self.domain.maxLims - self.domain.minLims) / 2
        vMin = -1 * vMax

        # Create new velocity and clamp it
        R1 = np.random.rand(self.domain.dimension)
        R2 = np.random.rand(self.domain.dimension)
        newVel = particle.velocity + c1 * (particle.pBestPosition - particle.position) * R1 + c2 * (
                self.globalBest.position - particle.position) * R2
        newVel[newVel > vMax] = vMax[newVel > vMax]
        newVel[newVel < vMin] = vMin[newVel < vMin]
        return newVel

    def checkPosition(self, position):
        """
        Check a given position to see if particle can move to it

        :param position: New position value, ndarray of size 1xnDim
        :return: Boolean indicating whether position is "allowed". True is allowed, false is not allowed
        """
        inLims = all(position > self.domain.minLims) and all(
            position < self.domain.maxLims)  # Will be true if within boundary
        if not inLims:
            return False
        else:
            return True
        # TODO: Add nogo zones, set of nogo indices

    def run(self):
        """
        Run the particle swarm algorithm. All parameters are set in __init__

        :return: None
        """
        for i in range(1, self.maxIter):
            for particle in self.particles:
                # Generate new velocity
                while True:
                    newVel = self.generateVelocity(particle)
                    newPos = particle.position + newVel
                    if self.checkPosition(newPos):
                        break

                # Update velocity, position, fitness
                particle.velocity = newVel
                particle.position = newPos
                particle.updateFitness(self.costArray)
                particle.updateHistory(i)
                if particle.currentFitness <= particle.pBestFitness:  # Minimizing, not maximizing!!
                    particle.pBestFitness = particle.currentFitness
                    particle.pBestPosition = particle.position.copy()

            # Update global best and history
            print(" ")
            self.getGlobalBest()
            self.bestPositionHistory[i, :] = self.globalBest.position
            self.bestFitnessHistory[i] = self.globalBest.currentFitness

            # Print
            print(
                f"Iteration: {i} || Best Position: {self.globalBest.position} || Best Fitness: {self.globalBest.currentFitness}")

    def plotConvergence(self):
        """
        Plot best fitness of all particles vs iteration

        :return: Handle for figure
        """
        fig, ax = plt.subplots()
        ax.plot(self.bestFitnessHistory)
        ax.set_title('Convergence')
        plt.show()
        return fig

    def getCurrentPoints(self, iteration):
        """
        Return the position of all the particles at a given iteration. Useful for animating plots

        :param iteration: Iteration of PSO where position for all particles is desired
        :return: Array of positions of all particles at given iteration
        """
        return np.array([p.positionHistory[iteration,:] for p in self.particles])



