from create_environment import createRect, createSquare
import biped
import matplotlib.pyplot as plt
import gurobipy as gp
import numpy as np

# back-and-forth path
# np.array([[0, 0], [2, 0], [4, 0], [6, 0], [6, 2], [6, 4], [4, 4],
#                            [2, 4], [0, 4], [0, 6], [0, 8], [2, 8], [4, 8], [6, 8], [6, 10], [6, 12]])

# np.array(
#     [[0, 0], [-0.5, 2], [0.5, 2], [2, 2.5], [0, 4], [3, 4], [1, 5.5]])

# np.array([[3, 1],
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

#np.array([[-1.5, 2.0], [0.5, 0.5], [1.5, 1.5], [2.5, 3.5]])

# np.array(
#     [[-2.0, 2.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, -2.0], [3.0, -3.0]])

# np.array(
#     [[-2.5, 2.5], [-1.5, 1.5], [-0.5, 0.5], [0.5, -0.5], [1.5, -1.5], [2.5, -2.5]])

# np.array([[0.5, 2.5], [-1.5, -1.5], [1.0, 0.0], [3.5, 2.0], [-2.5, 3.5], [-3.5, -0.5], [2.5, 3.5], [1.5, -3.5], [3.0, 0.5], [-2.0, -2.0], [-0.5, 2.0], [0.0, -3.0], [-1.0, 1.5], [2.0, -1.0], [
#                           3.0, 3.0], [-2.5, 1.0], [1.5, -0.5], [-3.5, 2.5], [-0.5, -1.5], [2.0, 3.0], [-1.5, 3.0], [0.5, -2.5], [2.5, 1.0], [-2.0, 0.5], [1.0, 2.5], [-3.0, 3.0], [0.5, 1.5], [-1.0, -0.5], [3.0, -3.0], [1.5, 2.5], [-3.5, 0.5], [-3.0, -2.5]])


test_cases = {}
no_steps = {}
step_dist = {}


no_steps["Fig_1"] = 46  # 19, 46
step_dist["Fig_1"] = 0.6  # 1, 0.6
test_cases["Fig_1"] = np.array([[18.8, 1.7],
                                [18.8, 3.4],
                                [18.85, 5.55],
                                [18.65, 8.15],
                                [19.55, 3.25],
                                [20.4, 4],
                                [21.15, 4.65],
                                [21.65, 5.45],
                                [21.8, 6.3],
                                [21.8, 7.65],
                                [21.8, 8.55],
                                [19.75, 6.05],
                                [19.9, 6.5],
                                [20.1, 7.25],
                                [20.2, 8],
                                [18.2, 5.45],
                                [17.85, 5.85],
                                [17.85, 6.4],
                                [17.5, 7.65],
                                [17.6, 3.7],
                                [16.2, 4.25],
                                [16.1, 6.1],
                                [16, 7.15],
                                [16.05, 5.25],
                                [15.95, 7.85],
                                [15.55, 8.9],
                                [16.2, 10.1],
                                [16.65, 11.2],
                                [21.95, 9.35],
                                [22.25, 9.75],
                                [23.2, 10.7],
                                [24.2, 11.45],
                                [19.3, 9.25],
                                [19.45, 10.5],
                                [18.6, 4.4],
                                [18.75, 2.65],
                                [24.1, 12.4],
                                [23.2, 13.3],
                                [22.55, 13.7],
                                [17.05, 11.8],
                                [17.65, 12],
                                [18.2, 12.05],
                                [19.05, 12.7],
                                [19.95, 13.3],
                                [19.85, 11.5],
                                [20.2, 12.35],
                                [20.95, 8],
                                [21.15, 6.25],
                                [20.95, 7],
                                [21.85, 10.55],
                                [21.9, 11.3],
                                [22.05, 12.2],
                                [20.15, 9.1],
                                [22.8, 14.65]])

no_steps["Fig_2"] = 32  # path doesn't take entire spiral
step_dist["Fig_2"] = 1.5
test_cases["Fig_2"] = test_cases["big_Spiral"] = np.array([[18.65, 14.8],
                                                           [15.75, 14.85],
                                                           [21.1, 14.25],
                                                           [23.65, 12.85],
                                                           [24.9, 10.7],
                                                           [25.25, 8.35],
                                                           [24.8, 5.95],
                                                           [23.3, 4.5],
                                                           [21.05, 3.55],
                                                           [18.45, 3.1],
                                                           [16.05, 3],
                                                           [13.9, 3.55],
                                                           [11.9, 4.7],
                                                           [10.35, 6.35],
                                                           [9.85, 8.1],
                                                           [10.25, 9.95],
                                                           [11.7, 11.45],
                                                           [13.4, 11.75],
                                                           [15.05, 11.65],
                                                           [16.7, 11.25],
                                                           [18.05, 10.75],
                                                           [19.2, 9.85],
                                                           [19.8, 8.5],
                                                           [19.25, 7.5],
                                                           [17.8, 7.1],
                                                           [16.8, 7.15],
                                                           [15.25, 7.45],
                                                           [17.1, 15.1],
                                                           [19.9, 14.7],
                                                           [22.35, 13.45],
                                                           [24.35, 11.7],
                                                           [25.05, 9.55],
                                                           [25, 7.25],
                                                           [23.7, 5.1],
                                                           [22.15, 4.15],
                                                           [19.7, 3.17],
                                                           [17.25, 3.15],
                                                           [14.9, 3.2],
                                                           [12.9, 4.25],
                                                           [11.25, 5.5],
                                                           [10, 7.25],
                                                           [10, 8.8],
                                                           [10.8, 10.6],
                                                           [12.3, 11.55],
                                                           [15.05, 6.3],
                                                           [15.8, 5.6],
                                                           [16.85, 5.15],
                                                           [18.3, 5.2],
                                                           [19.55, 5.4],
                                                           [20.6, 5.9],
                                                           [21.2, 6.65],
                                                           [21.65, 7.45],
                                                           [21.9, 8.5],
                                                           [21.75, 9.5],
                                                           [21.5, 10.45],
                                                           [21, 11.35],
                                                           [20.4, 12.05],
                                                           [19.5, 12.5]])

no_steps["Fig_3"] = 29
step_dist["Fig_3"] = 1.8
test_cases["Fig_3"] = test_cases["Spiral"] = np.array([[14.25, 14.6],
                                                       [16.15, 14.95],
                                                       [18.45, 14.95],
                                                       [20.19, 15.41],
                                                       [20.25, 14],
                                                       [21.55, 11.9],
                                                       [21.35, 10.05],
                                                       [21.81, 13.37],
                                                       [20.15, 8.9],
                                                       [18.5, 8.3],
                                                       [17.05, 8.35],
                                                       [17.3, 15.8],
                                                       [15.95, 8.85],
                                                       [14.85, 10.1],
                                                       [14.7, 11.65],
                                                       # these two extra regions on the end change the node count from 1 to 2027!!
                                                       [17.0, 11.65],
                                                       [18.7, 11.65], ])

no_steps["Fig_4"] = 1
step_dist["Fig_4"] = 3
test_cases["Fig_4"] = np.array([[6.75, 8.85],
                                [17.7, 8.2],
                                [14.65, 8.25],
                                [11.9, 8.35],
                                [9.25, 8.55],
                                [22.55, 10.55],
                                [20.55, 8.55],
                                [20.65, 12.7],
                                [17.85, 12.05],
                                [15.6, 12.6],
                                [12.6, 13.1],
                                [10.35, 13.6],
                                [7.75, 13.4],
                                [4.95, 12.95],
                                [2.4, 12.35]])

step_dist["Fig_5"] = 1.8
no_steps["Fig_5"] = 1
test_cases["Fig_5"] = np.array([[15.775, 0.95],
                                [14.625, 2.05],
                                [16.775, 2.1],
                                [17.825, 3.25],
                                [18.775, 4.55],
                                [19.575, 5.7],
                                [20.475, 7.05],
                                [15.525, 3],
                                [13.525, 3.1],
                                [12.575, 4.2],
                                [14.175, 4.3],
                                [16.125, 4.35],
                                [17.325, 5.7],
                                [18.475, 7.1],
                                [15.825, 7.2],
                                [15.125, 5.65],
                                [13.025, 5.65],
                                [14.025, 7.25],
                                [12.425, 7.2],
                                [11.525, 5.7],
                                [10.525, 7.1],
                                [9.725, 8.5],
                                [9.025, 9.8],
                                [8.275, 11.15],
                                [21.175, 8.2],
                                [21.675, 9.3],
                                [22.425, 10.7],
                                [22.925, 11.85],
                                [19.425, 8.25],
                                [17.325, 8.1],
                                [14.925, 8.2],
                                [13.025, 8.3],
                                [11.425, 8.3],
                                [10.475, 9.6],
                                [12.025, 9.6],
                                [13.625, 9.6],
                                [16.025, 9.6],
                                [18.225, 9.55],
                                [19.875, 9.55],
                                [20.575, 11.1],
                                [18.675, 11.35],
                                [16.975, 11.4],
                                [15.175, 11.4],
                                [13.225, 11.25],
                                [10.525, 11.55],
                                [15.325, 10.55],
                                [21.125, 12.55],
                                [19.725, 12.9],
                                [18.525, 12.9],
                                [16.825, 12.95],
                                [14.675, 12.95],
                                [12.825, 13],
                                [11.175, 12.85],
                                [8.775, 12.85],
                                [11.925, 11.55],
                                [15.875, 14.35],
                                [13.875, 14.35],
                                [11.775, 14.35],
                                [17.975, 14.5],
                                [16.775, 15.3],
                                [15.525, 15.4],
                                [13.075, 15.35],
                                [13.825, 15.35],
                                [13.875, 16.95],
                                [15.875, 16.95],
                                [14.925, 18.05]])


no_steps["Fig_6"] = 1
step_dist["Fig_6"] = 1

env = "Fig_3"


if env == "Fig_6":
    hulls = [createRect([0, 5], 12, 4)]

elif env == "Fig_5":
    region_centers = test_cases[env]
    hulls = [createSquare(center, 0.3) for center in region_centers]
# else:
    # region_centers = test_cases[env]
    # hulls = [createSquare(center, 0.3) for center in region_centers]

    # env_filename = env + "_log_testin.txt"
model = gp.Model('optimizer')
# model.Params.LogFile = env_filename
# open(env_filename, "w").write("")

# if env != "Fig_6":
#     biped.get_constraints(
#         model, hulls, start=region_centers[0], end=region_centers[-1], no_regions=len(region_centers), steps_taken=no_steps[env], reachable_distance=step_dist[env], logfile=env_filename)
# else:
#     biped.get_constraints(model, hulls, start=np.array([0, 0]), end=np.array([
#                           0, 10]), no_regions=1, steps_taken=no_steps[env], reachable_distance=step_dist[env], logfile=env_filename)

biped.get_constraints(model)
plt.axis('scaled')
plt.show()
