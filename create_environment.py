from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt


def createSquare(center, size):
    return createRect(center, size, size)


def createRect(center, sizev, sizeh):
    rect = ConvexHull(np.array([(center[0]-sizeh, center[1]+sizev), (center[0]+sizeh, center[1] +
                      sizev), (center[0]+sizeh, center[1]-sizev), (center[0]-sizeh, center[1]-sizev)]))
    for simplex in rect.simplices:
        plt.plot(rect.points[simplex, 0], rect.points[simplex, 1], 'k-')
    return rect

