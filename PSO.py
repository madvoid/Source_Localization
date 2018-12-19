# -------------------------------------------------------------------------------------------------
# Name: PSO.py
#
# Author: Nipun Gunawardena
#
# Purpose: PSO library for use with discrete grids
#
# Notes: May need to switch equalities to np.allclose()
# -------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.spatial.distance import cdist


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
        self.cellCounts = np.array(cellCountArr)  # (xCells, yCells, zCells) !! Need to be careful with this one !!
        self.cellCounts[0], self.cellCounts[1] = self.cellCounts[1], self.cellCounts[
            0]  # Change from (xCells, yCells, zCells) to (yCells, xCells, zCells). Done in separate line to maintain backwards compatibility
        self.duration = duration  # Seconds
        self.avgTime = averageTime  # Seconds
        self.numPeriods = numPeriods
        self.sourceLoc = np.array(sourceLocation)  # Not in cell units, in x,y,z units

        # Derived variables
        self.dimension = len(self.minLims)  # Dimension of problem. Should be 2 or 3 in most cases

    def __repr__(self):
        return f'Domain from {self.minLims} to {self.maxLims} with {self.ds} spacing'


class Particle:
    def __init__(self, domainClass, costArray, maximumIterations, maskArray=None):
        """
        Initialize particle for PSO

        :param domainClass: DomainInfo instance about current domain
        :param costArray: Cost array to be minimized
        :param maximumIterations: Maximum number of iterations. Used to initialize history arrays
        :param maskArray: Boolean array that is true where there is a building and false where there isn't
        """
        self.domain = domainClass  # Current domain
        self.position = np.random.uniform(self.domain.minLims, self.domain.maxLims)  # Initial position, 1xnDim array
        if maskArray is not None:
            index = self.getIndex()
            while maskArray[index]:
                self.position = np.random.uniform(self.domain.minLims, self.domain.maxLims)
                index = self.getIndex()
        self.velocity = np.random.random_sample(
            self.domain.dimension) * 0.1  # Initial velocity, multiply by 0.1 so no explode, same size as position
        self.currentFitness = self.updateFitness(costArray)  # Current fitness of particle
        self.pBestPosition = self.position  # Best location of particle
        self.pBestFitness = self.currentFitness  # Fitness of personal best position
        self.fitnessHistory = np.zeros(maximumIterations)  # History of particle fitness
        self.positionHistory = np.zeros([maximumIterations, self.domain.dimension])  # History of particle position
        self.fitnessHistory[0] = self.currentFitness
        self.positionHistory[0, :] = self.position
        self.isStuck = False  # Keep track if it's stuck in personal best position

    def __repr__(self):
        return f'Particle at position {self.position} and velocity {self.velocity}'

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

    def updateVelAndPos(self, newVelocity, newPosition, costArray):
        """
        Update velocity and position with new velocity and position

        :param newVelocity: New velocity value
        :param newPosition: New position value
        :param costArray: Cost array for new fitness value
        :return: None
        """
        self.velocity = newVelocity
        self.position = newPosition
        self.updateFitness(costArray)


class PSO:
    def __init__(self, costArray, domainClass, numberParticles=25, maximumIterations=1000, maskArray=None):
        """
        Initialize PSO class

        :param costArray: Array to minimize
        :param domainClass: Information about the domain
        :param numberParticles: Number of particles to run PSO with
        :param maximumIterations: Maximum iterations for PSO
        :param maskArray: Boolean array that is true where there is a building and false where there isn't
        """
        self.costArray = costArray[0]  # Array of values of cost function for first time step
        self.totalCostArray = costArray  # Cost function, all time steps
        self.currentTimeStep = 0  # Current time step
        self.maskArray = maskArray
        self.domain = domainClass  # DomainInfo instance about current domain
        self.maxIter = maximumIterations  # Maximum number of iterations
        self.numParticles = numberParticles  # Number of particles
        self.particles = [Particle(domainClass, self.costArray, self.maxIter, maskArray=maskArray) for _ in
                          range(self.numParticles)]  # Initialize particles
        self.globalBest = self.particles[0]  # Just set best particle to first one for now
        self.globalBestIndex = 0
        self.getGlobalBest()  # Find best particle
        self.bestPositionHistory = np.zeros([self.maxIter, self.domain.dimension])  # Keep track of best position found
        self.bestFitnessHistory = np.zeros(self.maxIter)  # Keep track of best fitness so far
        self.bestPositionHistory[0, :] = self.globalBest.position
        self.bestFitnessHistory[0] = self.globalBest.currentFitness
        self.stuckCheckVal = 5  # Number of times particle should be in neighborhood before force leave

    def __repr__(self):
        return f'PSO instance with {self.numParticles} particles'

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

    def getIndex(self, position):
        """
        Get matrix index of a given point in the domain. Duplicate of function in particle class

        :return: The index of current position. 1xnDim tuple (is a tuple for accessing elements) This is to be used with the meshgrid matrices, not the 1-dim matrices!
        """
        indexRaw = ((position - self.domain.minLims) / self.domain.ds).astype(int)
        indexRaw[0], indexRaw[1] = indexRaw[1], indexRaw[
            0]  # Switch first and second spots since x,y need to be flipped when accessing 2D array since x corresponds to columns and y corresponds to rows
        return tuple(indexRaw)

    def getPosition(self, index):
        """
        Get position on grid given a matrix row and column

        :param index: Matrix row and column as tuple (r,c) or (r, c, s)
        :return: Position as ndarray [x, y] or [x, y, z]
        """
        index = np.array(index)  # Convert to array so can update
        index[0], index[1] = index[1], index[0]  # Switch row and column so in order x, y instead of y, x
        pos = index * self.domain.ds + self.domain.minLims  # Convert to position, inverse of getIndex() math
        return pos

    def checkPosition(self, position):
        """
        Check a given position to see if particle can move to it

        :param position: New position value, ndarray of size 1xnDim
        :return: Boolean indicating whether position is "allowed". True is allowed, false is not allowed
        """
        inLims = all(position > self.domain.minLims) and all(
            position < self.domain.maxLims)  # Will be true if within boundary
        if not inLims:  # Check to see if in domain
            return False
        if self.maskArray is not None:
            if self.maskArray[self.getIndex(position)]:  # Check to see if in building
                return False
        return True

    def rotateVector(self, velocity):
        """
        Rotate vector clockwise in case it runs into building
        2d vectors are rotated +90 degrees
        3d vectors are rotated +90 degrees in xy plane and 180 degrees in z direction

        :param velocity: Velocity that needs to be rotated
        :return: Rotated velocity
        """
        if self.domain.dimension == 2:
            return np.matmul(np.array([[0, -1], [1, 0]]), velocity)
        elif self.domain.dimension == 3:
            rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]])
            return np.matmul(rot, velocity)
        else:
            raise ValueError('Domain is not 2 or 3 dimensions!')

    def getDistanceNorm(self, iteration):
        """
        Calculate norm of matrix of distances between all particles. Can be slow

        :param iteration: Iteration to check
        :return:
        """
        a = self.getCurrentPoints(iteration)
        return np.linalg.norm(cdist(a, a, metric='euclidean'), ord='fro')

    def checkNeighborhood(self, particle):
        """
        Check immediate neighborhood gridpoints of a given particle position

        :param particle: Particle to check neighbors of
        :return: Better position to move to from neighbors, if it exists
        """
        oPosition = particle.position  # Save current position to return if search returns nothing
        bFlag = False  # Flag to indicate whether better position was found or not
        pVal = particle.currentFitness  # Present value
        if self.domain.dimension == 2:  # Check dimension
            (rIdx, cIdx) = particle.getIndex()  # Get index of current position
            rBest, cBest = rIdx, cIdx
            if all(np.array([rIdx, cIdx]) > np.array([0, 0])) and all(np.array([rIdx, cIdx]) < (
                    self.domain.cellCounts - 1)):  # If not edge, check neighbors. Otherwise return same position
                for r in range(rIdx - 1, rIdx + 2):  # Iterate through rows
                    for c in range(cIdx - 1, cIdx + 2):  # Iterate through cols
                        if self.costArray[r, c] < pVal:  # If neighbor has smaller value than current
                            if self.checkPosition(
                                    self.getPosition((r, c))):  # If neighbor is allowed, update best neighbor
                                bFlag = True
                                pVal = self.costArray[r, c]
                                rBest, cBest = r, c
                            else:
                                continue  # If neighbor not allowed, go to next
                if bFlag:
                    return self.getPosition((rBest, cBest))
                else:
                    return oPosition
            else:
                return particle.position
        elif self.domain.dimension == 3:  # See 2D part for comments
            (rIdx, cIdx, sIdx) = particle.getIndex()
            rBest, cBest, sBest = rIdx, cIdx, sIdx
            if all(np.array([rIdx, cIdx, sIdx]) > np.array([0, 0, 0])) and all(
                    np.array([rIdx, cIdx, sIdx]) < (self.domain.cellCounts - 1)):
                for r in range(rIdx - 1, rIdx + 2):
                    for c in range(cIdx - 1, cIdx + 2):
                        for s in range(sIdx - 1, sIdx + 2):
                            if self.costArray[r, c, s] < pVal:
                                if self.checkPosition(self.getPosition((r, c, s))):
                                    bFlag = True
                                    pVal = self.costArray[r, c, s]
                                    rBest, cBest, sBest = r, c, s
                                else:
                                    continue
                if bFlag:
                    return self.getPosition((rBest, cBest, sBest))
                else:
                    return oPosition
            else:
                return particle.position
        else:
            raise ValueError('Domain is not 2 or 3 dimensions!')

    def generateVelocity(self, particle):
        """
        Generate new velocity for particle class

        :param particle: Instance of Particle class
        :return: New velocity, ndarray of size 1xnDim. Particle is NOT updated in this function
        """
        # Establish Constants
        if particle.pBestFitness == 0.0:  # QUIC specific addition, don't weight empty spots heavily
            c1 = 0.5
            c2 = 3.5
        else:
            c1 = 2  # Local Best
            c2 = 2  # Global Best

        # Inertial weight decay
        omega = 0.05

        # Velocity clamping
        k = 0.05  # Velocity clamping constant, this makes a big difference!!!
        vMax = 0.05 * (self.domain.maxLims - self.domain.minLims)
        vMin = -1 * vMax

        # Create new velocity and clamp it
        R1 = np.random.rand(self.domain.dimension)
        R2 = np.random.rand(self.domain.dimension)
        cogComp = c1 * (particle.pBestPosition - particle.position) * R1
        socComp = c2 * (self.globalBest.position - particle.position) * R2
        if (all(cogComp == 0) and all(socComp == 0)) or (
                particle.isStuck):  # If a particle is in the global best position, or stuck in personal best, jump around a bit. Otherwise do regular PSO
            particle.isStuck = False
            newVel = np.random.uniform(vMin, vMax, particle.velocity.shape)
        else:
            newVel = particle.velocity + cogComp + socComp
            newVel = newVel * k
        # TODO: Play with this section to change behavior
        # newVel[newVel > vMax] = vMax[newVel > vMax]
        # newVel[newVel < vMin] = vMin[newVel < vMin]
        return newVel

    def run(self, checkNeighborhood=False, verbose=True):
        """
        Run the particle swarm algorithm. All parameters are set in __init__

        :return: None
        """
        # Set altVector to function used to produce alternate velocity vector if original is not allowed
        # All functions should have same inputs, for now functions are a part of this class
        # May change in future
        altVector = self.rotateVector

        self.distNorm = np.ones(self.maxIter)
        self.distNorm[0] = self.getDistanceNorm(0)

        # Start iterations
        for i in range(1, self.maxIter):
            for pIdx, particle in enumerate(self.particles):

                # Generate new velocity
                newVel = self.generateVelocity(particle)
                newPos = particle.position + newVel
                if not self.checkPosition(newPos):
                    while True:
                        newVel = altVector(newVel)
                        newPos = particle.position + newVel
                        if self.checkPosition(newPos):
                            break

                # Update velocity, position, fitness
                particle.updateVelAndPos(newVel, newPos, self.costArray)
                if checkNeighborhood:
                    newPos = self.checkNeighborhood(particle)
                    particle.updateVelAndPos(newVel, newPos, self.costArray)
                particle.updateHistory(i)
                if particle.currentFitness <= particle.pBestFitness:  # Minimizing, not maximizing!!
                    particle.pBestFitness = particle.currentFitness
                    particle.pBestPosition = particle.position.copy()

                # Check to see if particle is within the dx, dy, dz for a given amount of times
                stuckCheck = np.diff(particle.positionHistory[i - self.stuckCheckVal:i], axis=0)
                if (stuckCheck < self.domain.ds).all():
                    particle.isStuck = True

            # Update global best and history
            self.getGlobalBest()
            self.bestPositionHistory[i, :] = self.globalBest.position
            self.bestFitnessHistory[i] = self.globalBest.currentFitness

            # Get idea of distance between particles
            self.distNorm[i] = self.getDistanceNorm(i)

            # Print
            if verbose:
                print(
                    f"Iteration: {i} || Best Position: {self.globalBest.position} || Best Fitness: {self.globalBest.currentFitness}")

        # Finish up
        if verbose:
            print("\nFinished Iterations")

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

    def plotDistanceNorm(self):
        fig, ax = plt.subplots()
        ax.plot(self.distNorm)
        ax.set_title('Distance Norm')
        plt.show()
        return fig

    def getCurrentPoints(self, iteration):
        """
        Return the position of all the particles at a given iteration. Useful for animating plots

        :param iteration: Iteration of PSO where position for all particles is desired
        :return: Array of positions of all particles at given iteration, size nPoints x nDims
        """
        return np.array([p.positionHistory[iteration, :] for p in self.particles])


def rebin(ndarray, dVal, operation='mean'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or averaging. Number of output dimensions
    must match number of input dimensions and new dimensions must evenly divide into
    the old ones. To be used to improve visualizations of PSO

    Slightly modified from https://stackoverflow.com/a/29042041 by @Nipun
    Simple walkthrough found at https://scipython.com/blog/binning-a-2d-array-in-numpy/

    :param ndarray: Array to be downsampled
    :param dVal: Downscale factor for new array, dVal should be an even factor of every dimension of original array
    :param operation: Downsample using mean or sum
    :return: New array
    """

    new_shape = tuple([i // dVal for i in ndarray.shape])

    # Check inputs
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))

    # Reshape into ndim + 1 array and operate along new dimension
    compression_pairs = [(d, c // d) for d, c in zip(new_shape,
                                                     ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1 * (i + 1))
    return ndarray


def reslice3D(X, dVal):
    """
    Reslice a meshgrid output array to match the output of the rebin() function, only to be used with 3D arrays

    :param X: Meshgrid output
    :param dVal: Factor to reduce by. Should be same factor as used in rebin
    :return:
    """
    return X[::dVal, ::dVal, ::dVal]
