from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt


def createSquare(center, size, plot=True):
    return createRect(center, size, size, plot)


def createRect(center, sizev, sizeh, plot=True):
    rect = ConvexHull(np.array([(center[0]-sizeh, center[1]+sizev), (center[0]+sizeh, center[1] +
                      sizev), (center[0]+sizeh, center[1]-sizev), (center[0]-sizeh, center[1]-sizev)]))
    if plot:
        for simplex in rect.simplices:
            plt.plot(rect.points[simplex, 0], rect.points[simplex, 1], 'k-')
    return rect
