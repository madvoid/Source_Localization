# -------------------------------------------------------------------------------------------------
# Name: MatRead.py
#
# Author: Nipun Gunawardena
#
# Purpose: Read .mat file, specifically one made by Domain_Maker.m for QUIC data
#
# Notes: Doesn't do anything when run by itself
# -------------------------------------------------------------------------------------------------

import numpy as np
from scipy.io import loadmat
from PSO import DomainInfo


def readQUICMat(filename):
    """
    Read a .mat file created by Domain_Maker.m for QUIC data

    :param filename: Path to .mat file to read
    :return: Tuple (QuicDomain, X, Y, Z, B, C, C_Plot) containing vars needed by PSO. See code comments for details.
    """
    # Load .mat
    M = loadmat(filename, squeeze_me=True)

    # Unpack variables
    if M['C'].ndim == 1:  # Time varying case, set C to to first time step  TODO: Update to handle time varying case, save other time steps
        C = M['C'][0]
    elif M['C'].ndim == 3:  # Constant time case
        C = M['C']
    bldgData = M['domain']  # Building matrix (B)
    avgTime = M['avgTime']  # Averaging time for each period in total time
    duration = M['duration']  # Total duration of simulation
    dx = M['dx']  # Grid spacing - x
    dy = M['dy']  # Grid spacing - y
    dz = M['dz']  # Grid spacing - z
    numPeriods = M['numPeriods']  # Number of averaging periods in simulation
    sourceLoc = M['sourceLoc']  # (x,y,z) coordinate of source
    xCells = M['xCells']  # Number of cells - x
    yCells = M['yCells']  # Number of cells - y
    zCells = M['zCells']  # Number of cells - z

    # Create domain class
    QuicDomain = DomainInfo([dx, dy, dz], [xCells * dx, yCells * dy, zCells * dz], [dx, dy, dz],
                            [xCells, yCells, zCells], duration, avgTime, numPeriods, sourceLoc)

    # Create X, Y, Z for plotting
    x = np.arange(dx, xCells * dx + dx, step=dx)
    y = np.arange(dy, yCells * dy + dy, step=dy)
    z = np.arange(dz, zCells * dz + dz, step=dz)
    (X, Y, Z) = np.meshgrid(x, y, z, indexing='xy')

    # Change concentrations so there is 0 instead of nan for cells with no particulate
    C[np.isnan(C)] = 0

    # Create building logical matrix
    B = np.logical_not(np.isnan(bldgData))

    # Create plotting concentration matrix
    C_Plot = np.copy(C)
    try:  # If this happens it probably means there are no buildings
        C_Plot[B == True] = np.nan
    except:
        C_Plot = C

    # Invert concentrations
    # Concentrations are positive but PSO minimizes so invert concentrations
    # Don't do the same for C_Plot so visualizations are positive
    C = C * (-1)

    # Return values
    return (QuicDomain, X, Y, Z, B, C, C_Plot)


if __name__ == "__main__":
    matFile = 'QUIC Data/Simple1/Data.mat'
    readQUICMat(matFile)
