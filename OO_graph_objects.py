from scipy.spatial import ConvexHull
from scipy.spatial import HalfspaceIntersection
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from graph_construction import plot_hull
from create_environment import createSquare

walkable_regions = []


class HullSection(ConvexHull):
    def __init__(self, parent_hull: ConvexHull, vertices: np.array(float), foot_in_this_hull="left"):  # default is important
        super().__init__(vertices)

        self.parent_hull = parent_hull
        self.foot_in = foot_in_this_hull
        self.vision = VisionHull(
            self, "right" if foot_in_this_hull == "left" else "left")

    def get_vertices(self):
        return [self.points[v] for v in self.vertices]

    def get_children_hulls(self, environment: list[ConvexHull], explored):
        return self.vision.intersect_with_world(environment, explored)

    def contains(self, point: list[float],  tolerance=1e-12):
        return all(
            (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
            for eq in self.equations)


class HullNode():
    def __init__(self, hull: HullSection | None, parent, children):
        self.hull = hull
        self.parent = parent
        self.children: list[HullNode] = children

    def add_child(self, child: HullSection):
        self.children.append(HullNode(hull=child, parent=self, children=[]))

    def add_children(self, children):
        for child in children:
            self.add_child(child)


class VisionHull(ConvexHull):
    def __init__(self, source: HullSection | list[float], foot):
        self.source = source
        self.foot = foot

        if isinstance(self.source, HullSection):
            extremities = [self.source.points[v] for v in self.source.vertices]
            super().__init__(np.vstack([self.linearise_reachable_region(
                x) for x in extremities]))

        else:
            super().__init__(self.linearise_reachable_region(self.source))

    def linearise_reachable_region(self, centre=[0, 0]):
        # global foot
        global no_points
        global reachable_distance
        global offset
        x = []
        y = []
        if self.foot == 'right':
            initial_angle = math.pi/2
            min_x_sep = -offset
        else:
            initial_angle = 3*math.pi/2
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

        return np.array(list(zip(x, y)))

    def hull_intersection(self, env_region: ConvexHull):

        halfspaces = np.concatenate((env_region.equations,
                                    self.equations))

        norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1),
                                 (halfspaces.shape[0], 1))
        c = np.zeros((halfspaces.shape[1],))
        c[-1] = -1
        A = np.hstack((halfspaces[:, :-1], norm_vector))
        b = - halfspaces[:, -1:]
        res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
        x = res.x[:-1]
        try:
            hs = HalfspaceIntersection(halfspaces, x)
            x, y = zip(*hs.intersections)
            hull_section = HullSection(
                env_region, np.array(list(zip(x, y))), self.foot)
        except Exception as e:
            # print("halfspace not good", e)
            return None
        return hull_section

    def intersect_with_world(self, environment: list[ConvexHull], explored: list[HullSection]) -> list[HullSection]:
        res = []
        for hull in environment:
            i = self.hull_intersection(hull)
            if i:
                same_parent_regions = [
                    x for x in explored if x.parent_hull is i.parent_hull]
                for explored_region in same_parent_regions:

                    distinction_check_volume = ConvexHull(
                        np.vstack([i.points, explored_region.points]))
                    larger_region = i if i.volume > explored_region.volume else explored_region

                    if distinction_check_volume.volume == larger_region.volume:
                        if larger_region is i:

                            res.append(i)
                            continue
                        else:
                            # res.append(explored_region)
                            pass

                if not same_parent_regions:
                    res.append(i)

        # intersection of convex sets is convex
        return res


# env = np.array([[0, 0], [-0.5, 1], [0.5, 1]])

env = np.array([[18.8, 1.7],  # 0.6
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

world = [createSquare(center, 0.3, False) for center in env]

config = [1, 3, "right", 0.1]

[reachable_distance, no_points, foot, offset] = config

# graph_pathfind(hulls,
#                env[0], env[-1])


# initial_vision = VisionHull(start, foot)
# initial_sections = initial_vision.intersect_with_world(world)
# root = HullNode(hull=None, parent=None, children=[])

# current = root.children

# for initial_section in initial_sections:
#     root.add_child(initial_section)

# new: list[HullNode] = []

def cleanup(regions: list[HullSection], world):
    for hull in world:
        same_parent_regions = [
            x for x in regions if x.parent_hull is hull]
        for region in same_parent_regions:
            for region2 in same_parent_regions:
                distinction_check_volume = ConvexHull(
                    np.vstack([region2.points, region.points]))
                larger_region = region2 if region2.volume > region.volume else region
                if distinction_check_volume.volume == larger_region.volume and not (region is region2):
                    if larger_region is region2:
                        # print("removing hullsection with vertices",
                        # _region.get_vertices())a
                        if region in regions:
                            regions.remove(region)
    return regions


def search(world, start, end, config):
    step = 1
    global foot
    explored_walkable_regions = []
    explored_walkable_regions_right = []
    explored_walkable_regions_left = []
    initial_vision = VisionHull(start, foot)
    initial_sections = initial_vision.intersect_with_world(world, [])
    root = HullNode(hull=None, parent=None, children=[])

    current = root.children
    for initial_section in initial_sections:
        root.add_child(initial_section)
    while True:
        print(len(current))
        for node in current:
            if node.hull.foot_in == "left":
                explored_walkable_regions_left.append(node.hull)
            elif node.hull.foot_in == "right":
                explored_walkable_regions_right.append(node.hull)

        explored_walkable_regions_left = cleanup(
            explored_walkable_regions_left, world)
        explored_walkable_regions_right = cleanup(
            explored_walkable_regions_right, world)

        # for region in world:
        #     plot_hull(region)
        # for region in explored_walkable_regions_left:
        #     plot_hull(region, color="blue")
        # for region in explored_walkable_regions_right:
        #     plot_hull(region, color="green")
        # for region in current:
        #     plot_hull(region.hull, color="pink")
        print("taking step", step)
        for region in current:
            if region.hull.contains(end):
                print("acquired")
                return
        new = []
        for region in current:
            region_vision = region.hull.get_children_hulls(
                world, explored_walkable_regions_left if region.hull.foot_in == "left" else explored_walkable_regions_right)
            region.add_children(region_vision)
            # plot_hull(region.hull.vision, color="orange")
            new.extend(region.children)
        # plt.axis("scaled")
        # plt.show()
        current = new
        step += 1


start = env[0]
end = env[-1]

search(world, start, end, config)
