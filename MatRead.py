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
from copy import deepcopy
from PSO import DomainInfo
from scipy.io import loadmat


def readQUICMat(filename):
    """
    Read a .mat file created by Domain_Maker.m for QUIC data

    :param filename: Path to .mat file to read
    :return: Tuple (QuicDomain, X, Y, Z, B, C, C_Plot) containing vars needed by PSO. See code comments for details.
    """
    # Load .mat
    M = loadmat(filename, squeeze_me=True)

    # Unpack variables
    if M['C'].ndim == 3:  # Non time varying case, wrap in array so looks like time varying case
        C = np.empty(shape=(1,), dtype=object)
        C[0] = M['C']
    else:  # Time Varying case
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
    for i in C:
        i[np.isnan(i)] = 0

    # Create building logical matrix
    B = np.logical_not(np.isnan(bldgData))

    # Create plotting concentration matrix
    C_Plot = deepcopy(C)
    try:  # If exception happens it probably means there are no buildings
        for i in C_Plot:
            i[B == True] = np.nan
    except Exception:
        pass

    # Invert concentrations
    # Concentrations are positive but PSO minimizes so invert concentrations
    # Don't do the same for C_Plot so visualizations are positive
    for idx, i in enumerate(C):
        i = i * (-1)
        C[idx] = i  # For some reason need to reassign instead of just operating on i as done in lines 56-58

    # Return values
    return (QuicDomain, X, Y, Z, B, C, C_Plot)


if __name__ == "__main__":
    matFile = 'QUIC Data/Simple1/Data.mat'
    readQUICMat(matFile)
