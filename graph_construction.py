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


def linearise_reachable_region(reachable_distance, no_points, centre=[0, 0], initial_angle=3*math.pi/2):
    x = []
    y = []
    for i in range(no_points+1):
        x.append(math.cos(initial_angle + math.pi*(i/no_points)))
        y.append(math.sin(initial_angle + math.pi*(i/no_points)))

    # c = ConvexHull(np.array(list(zip(x, y))))
    plt.plot(x, y)
    plt.axis("scaled")
    plt.show()


linearise_reachable_region(0, 10)
