# -------------------------------------------------------------------------------------------------
# Name: Percent_Plume.py
#
# Author: Nipun Gunawardena
#
# Purpose: Given a QUIC domain, calculate the perecentage of the domain which is taken by plume
#          Also print other domain stats
# -------------------------------------------------------------------------------------------------

import sys
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, '../')
from PSO import *
from MatRead import readQUICMat

if __name__ == "__main__":
    # Create save path
    baseName = 'Simple3'
    basePath = '../Results/' + baseName + '/Statistics/' + baseName + '_'

    # Retrieve domain and data
    quicDomain, X, Y, Z, B, C, C_Plot = readQUICMat('../QUIC Data/' + baseName + '/Data.mat')

    timeIndex = 0

    # Count
    numElements = np.prod(C[0].shape)
    totalNans = np.isnan(C_Plot[0]).sum()
    print(f"Domain name: {baseName}")
    print(f"Number of time steps: {len(C)}")
    print(f"Cell counts: {quicDomain.cellCounts}")
    print(f"Total elements in domain: {numElements}")
    print(f"Total NaN elements in domain: {totalNans} ({totalNans/numElements*100}%)")
    print(f"ds (m): {quicDomain.ds}")
    print(f"Min lims (m): {quicDomain.minLims}")
    print(f"Max lims (m): {quicDomain.maxLims}")
    print(f"Source location (m): {quicDomain.sourceLoc}")
    print(f"Maximum building height (m): {np.max(np.argmax(~B, axis=2))*quicDomain.ds[2]}")
    print(f"---------------------------")
    for t in range(len(C)):
        totalNonzeros = np.count_nonzero(C_Plot[t]) - totalNans
        print(f"Time index: {t}")
        print(f"\tTotal nonzero elements in domain (Not including buildings): {totalNonzeros} ({totalNonzeros/(numElements-totalNans)*100}%)")
