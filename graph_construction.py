from scipy.spatial import ConvexHull
import numpy as np
import math
import matplotlib.pyplot as plt


class HullSection(ConvexHull):
    def __init__(self, parent: ConvexHull, vertices: np.array(float)):
        super().__init__(vertices)
        self.parent = parent


def construct_graph(regions: list[ConvexHull], reach_dist):
    env_graph = {}


# linear approximation of region is always a subset of the actual reachable region,
# so it will increase efficacy at higher point count, while guaranteeing reachability.


def linearise_reachable_region(reachable_distance, no_points, centre=[0, 0], foot='right', offset=0.1):
    x = []
    y = []
    if foot == 'right':
        initial_angle = math.pi/2
        color = 'blue'
        min_x_sep = -offset
    else:
        initial_angle = 3*math.pi/2
        color = 'green'
        min_x_sep = offset
    for i in range(no_points+1):
        x.append(min_x_sep + centre[0] + math.cos(initial_angle +
                 math.pi*(i/no_points)) * reachable_distance)
        y.append(centre[1] + math.sin(initial_angle +
                 math.pi*(i/no_points))*reachable_distance)
    # add first points at the end for plot to look closed.
    x.append(min_x_sep + centre[0] +
             math.cos(initial_angle) * reachable_distance)
    y.append(centre[1] + math.sin(initial_angle)*reachable_distance)

    plt.plot(x, y, "-", color=color, alpha=0.4)
    plt.axis("scaled")
    # plt.show()
    return ConvexHull(np.array(list(zip(x, y))))


# linearise_reachable_region(2, 10, [1, 1], 7*math.pi/6)
