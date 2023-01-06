from create_environment import createRect, createSquare
import biped
import matplotlib.pyplot as plt
import gurobipy as gp
import numpy as np

region_centers = np.array([[0, 0], [2, 0], [4, 0], [6, 0], [6, 2], [6, 4], [4, 4],
                           [2, 4], [0, 4], [0, 6], [0, 8], [2, 8], [4, 8], [6, 8], [6, 10], [6, 12]])

hulls = []

for center in region_centers:
    hulls.append(createSquare(center, 0.5))

model = gp.Model('optimizer')
biped.get_constraints(
    model, hulls, start=region_centers[0], end=region_centers[-1], no_regions=len(region_centers))

plt.axis('scaled')
plt.show()
