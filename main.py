from create_environment import createRect, createSquare
import biped
import matplotlib.pyplot as plt
import gurobipy as gp
import numpy as np

# back-and-forth path
# region_centers = np.array([[0, 0], [2, 0], [4, 0], [6, 0], [6, 2], [6, 4], [4, 4],
#                            [2, 4], [0, 4], [0, 6], [0, 8], [2, 8], [4, 8], [6, 8], [6, 10], [6, 12]])

# region_centers = np.array(
#     [[0, 0], [-0.5, 2], [0.5, 2], [2, 2.5], [0, 4], [3, 4], [1, 5.5]])

# region_centers = np.array([[3, 1],
#                            [4.95, 1],
#                            [5, 3.1],
#                            [6, 4],
#                            [7, 5.05],
#                            [5.95, 6.05],
#                            [5.95, 8],
#                            [7.9, 8.15],
#                            [9.8, 8.2],
#                            [9.85, 9.95],
#                            [9.8, 11.75],
#                            [8.05, 11.9],
#                            [2, 2.05],
#                            [2, 3.95],
#                            [2, 6.05],
#                            [1.85, 8.05],
#                            [1.9, 9.95],
#                            [4.95, 9],
#                            [7, 1.1],
#                            [6.95, 2.9],
#                            [3.8, 5.1],
#                            [3.85, 7.65],
#                            [6.15, 11.9]])

# region_centers = np.array([[-1.5, 2.0], [0.5, 0.5], [1.5, 1.5], [2.5, 3.5]])

# region_centers = np.array(
#     [[-2.0, 2.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, -2.0], [3.0, -3.0]])

# region_centers = np.array(
#     [[-2.5, 2.5], [-1.5, 1.5], [-0.5, 0.5], [0.5, -0.5], [1.5, -1.5], [2.5, -2.5]])

region_centers = np.array([[0.5, 2.5], [-1.5, -1.5], [1.0, 0.0], [3.5, 2.0], [-2.5, 3.5], [-3.5, -0.5], [2.5, 3.5], [1.5, -3.5], [3.0, 0.5], [-2.0, -2.0], [-0.5, 2.0], [0.0, -3.0], [-1.0, 1.5], [2.0, -1.0], [
                          3.0, 3.0], [-2.5, 1.0], [1.5, -0.5], [-3.5, 2.5], [-0.5, -1.5], [2.0, 3.0], [-1.5, 3.0], [0.5, -2.5], [2.5, 1.0], [-2.0, 0.5], [1.0, 2.5], [-3.0, 3.0], [0.5, 1.5], [-1.0, -0.5], [3.0, -3.0], [1.5, 2.5], [-3.5, 0.5], [-3.0, -2.5]])

hulls = []

hulls = [createSquare(center, 0.4) for center in region_centers]

model = gp.Model('optimizer')
biped.get_constraints(
    model, hulls, start=region_centers[0], end=region_centers[-1], no_regions=len(region_centers))

plt.axis('scaled')
plt.show()
