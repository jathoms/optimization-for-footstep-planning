from scipy.spatial import ConvexHull
from scipy.spatial import HalfspaceIntersection
import numpy as np
import math
import gurobipy as gp
import matplotlib.pyplot as plt
from scipy.optimize import linprog

from create_environment import createSquare


class HullSection(ConvexHull):
    def __init__(self, parent_hull: ConvexHull, vertices: np.array(float)):
        super().__init__(vertices)
        self.parent_hull = parent_hull
        self.vision = VisionHull(self)

    def get_vertices(self):
        return [self.points[v] for v in self.vertices]

    def get_children_hulls(self, environment: list[ConvexHull]):
        return self.vision.intersect_with_world(environment)

    def contains(self, point: list[float],  tolerance=1e-12):
        return all(
            (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
            for eq in self.equations)


class HullNode():
    def __init__(self, hull: HullSection | None, parent, children):
        self.hull = hull
        self.parent = parent
        self.children = children

    def add_child(self, child: HullSection):
        self.children.append(HullNode(hull=child, parent=self, children=[]))

    def add_children(self, children):
        for child in children:
            self.add_child(child)


class VisionHull(ConvexHull):
    def __init__(self, source: HullSection | list[float]):
        self.source = source

        if isinstance(self.source, HullSection):
            extremities = [self.source.points[v] for v in self.source.vertices]
            super().__init__(np.vstack([self.linearise_reachable_region(
                x) for x in extremities]))
            plot_hull(self, color="green")
        else:
            super().__init__(self.linearise_reachable_region(self.source))
            plot_hull(self, color="red")

    def linearise_reachable_region(self, centre=[0, 0]):
        global foot
        global no_points
        global reachable_distance
        global offset
        x = []
        y = []
        if foot == 'right':
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
            hull_section = HullSection(env_region, np.array(list(zip(x, y))))
        except Exception as e:
            print("halfspace not good", e)
            return None
        return hull_section

    def intersect_with_world(self, environment: list[ConvexHull]) -> list[HullSection]:
        res = []
        for hull in environment:
            # order is important, as the first argument is treated as part of the environment
            i = self.hull_intersection(hull)
            if i:
                # plot_hull(i, title="intersection", color="red")
                # plt.show()
                # same_parent_regions = [
                #     x for x in walkable_regions if x.parent is i.parent]
                # for region in walkable_regions:
                #     if region not in same_parent_regions:
                #         continue
                #     distinction_check_volume = ConvexHull(
                #         np.vstack([i.points, region.points]))
                #     larger_region = i if i.volume > region.volume else region
                #     # plot_hull(distinction_check_volume, color="red")
                #     # plot_hull(hull)
                #     if distinction_check_volume.volume == larger_region.volume:
                #         if larger_region is i:
                #             print("removing hullsection with vertices",
                #             2      region.get_vertices())
                #             walkable_regions.remove(region)
                #             walkable_regions.append(i)
                #             plot_hull(i, color="red")

                #         elif larger_region is region:
                #             plot_hull(region, color="red")
                #             pass
                # else:
                # walkable_regions.append(i)
                res.append(i)
                # plot_hull(i, color="red")
                continue
                # if not same_parent_regions:
                #     plot_hull(i, color="red")
                #     walkable_regions.append(i)
        # intersection of convex hulls is convex

        return res


a = 0


def linearise_reachable_region(centre=[0, 0], foot2="right"):
    # linear approximation of region is always a subset of the actual reachable region,
    # so it will increase efficacy at higher point count, while guaranteeing reachability at any n.
    global foot
    global no_points
    global reachable_distance
    global offset
    # global a
    # print("started linearisation", a)
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

    # plt.plot(x, y, "-", color=color, alpha=0.4)
    # plt.axis("scaled")
    # plt.show()

    # print("finished linearisation", a)
    # a += 1
    return ConvexHull(np.array(list(zip(x, y))))


calls = 1
appends = 1


def intersect_reachable_region(environment: list[ConvexHull], r_region: ConvexHull, walkable_regions: list[HullSection]) -> list[HullSection] | None:
    global calls
    global appends
    for hull in environment:
        # order is important, as the first argument is treated as part of the environment
        i = intersection(hull, r_region)
        if i:
            for hull2 in environment:
                plot_hull(hull2)
            # plot_hull(i, title="intersection", color="red")
            # plt.show()
            same_parent_regions = [
                x for x in walkable_regions if x.parent is i.parent]
            for region in walkable_regions:
                if region not in same_parent_regions:
                    continue
                distinction_check_volume = ConvexHull(
                    np.vstack([i.points, region.points]))
                larger_region = i if i.volume > region.volume else region
                # plot_hull(distinction_check_volume, color="red")
                # plot_hull(hull)
                if distinction_check_volume.volume == larger_region.volume:
                    if larger_region is i:
                        print("removing hullsection with vertices",
                              region.get_vertices())
                        walkable_regions.remove(region)
                        walkable_regions.append(i)
                        plot_hull(i, color="red")

                    elif larger_region is region:
                        plot_hull(region, color="red")
                        pass
                else:
                    walkable_regions.append(i)
                    plot_hull(i, color="red")
                    continue
            if not same_parent_regions:
                plot_hull(i, color="red")
                walkable_regions.append(i)
    # intersection of convex hulls is convex

    return walkable_regions


def reachable_from_region(region: ConvexHull) -> ConvexHull:

    extremities = [region.points[v] for v in region.vertices]

    return ConvexHull(np.vstack([linearise_reachable_region(
        x).points for x in extremities]))


a = 1


def intersection(env_region: ConvexHull, reachable_region: ConvexHull):

    halfspaces = np.concatenate((env_region.equations,
                                reachable_region.equations))

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
        hull_section = HullSection(env_region, np.array(list(zip(x, y))))
    except Exception as e:
        print(e)
        return None
    return hull_section


def cleanup(env: list[ConvexHull], current_walkable_regions: list[HullSection]):
    global a
    remove = []
    new_walkable_regions = []
    for r in env:
        subregions_of_r = [
            region for region in current_walkable_regions if region.parent is r]
        subregions_of_r = sorted(subregions_of_r, key=lambda x: x.volume)
        for idx, sr in enumerate(subregions_of_r):
            for smaller_sr in subregions_of_r[idx+1:]:
                # hacky solution for now
                if ConvexHull(np.vstack([sr.points, smaller_sr.points])).volume == sr.volume:
                    print("removing", a)
                    a += 1
                    subregions_of_r.remove(smaller_sr)
        new_walkable_regions.extend(subregions_of_r)
    print(len(current_walkable_regions), len(new_walkable_regions))
    return new_walkable_regions


def concat_list(l):
    return [item for sublist in l for item in sublist]


# def simplified_hull(hull_list_list, parent):
#     hull_list_list = [hull_list for hull_list in hull_list_list if hull_list]
#     return [HullSection(parent, np.vstack([hull.points for hull in hull_list])) for hull_list in hull_list_list]

def next_step(env, current_walkable_regions):
    # for hullsection in current_walkable_regions:
    #     print(hullsection.points)
    # current_walkable_regions = cleanup(env, current_walkable_regions)
    return concat_list([intersect_reachable_region(env,
                                                   reachable_from_region(hull), current_walkable_regions)
                        for hull in current_walkable_regions])


def graph_pathfind(environment: list[ConvexHull], start_point, end_point):
    global foot
    global no_points
    global reachable_distance
    global offset
    steps = 0
    checked_vertices = []
    reachable_region = linearise_reachable_region(
        start_point)
    new_walkable_regions = intersect_reachable_region(
        environment, reachable_region, [])
    # for hull in new_walkable_regions:
    #     plot_hull(hull, color="red")
    # for hull in environment:
    # plot_hull(hull)
    # plt.show()
    for _ in range(20):
        plt.axis("scaled")
        plt.show()
        print(steps, "steps")
        steps += 1
        new_walkable_regions = next_step(
            environment, new_walkable_regions)
        skip = len(new_walkable_regions)
        print(skip)
        foot = "right" if foot == "left" else "left"

    plt.axis("scaled")
    plt.show()
    return
    for i in range(2):
        foot = "right" if foot == "left" else "left"
        reachable_regions = [reachable_from_region(
            region, reachable_distance, no_points, foot, offset) for region in new_walkable_regions]

        walkable_regions = [item for sublist in [intersect_reachable_region(
            environment, region) for region in reachable_regions] for item in sublist]
        for region in walkable_regions:
            # plot_hull(region, f"Reachable distance after {i+1} step(s)", "red")
            if ConvexHull(region.points + end_point) == region:
                print(f"found in {i} steps")
                return


def plot_hull(hull, title="", color="black"):
    plt.title(title)
    vertices = hull.points
    for simplex in hull.simplices:
        plt.plot(vertices[simplex, 0], vertices[simplex, 1], 'k-', color=color)
    # plt.show()
    return


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

env = np.array([[0, 0], [-0.5, 1], [0.5, 1]])

world = [createSquare(center, 0.3) for center in env]

config = [1, 10, "right", 0.1]

[reachable_distance, no_points, foot, offset] = config

# graph_pathfind(hulls,
#                env[0], env[-1])

start_point = env[0]
end_point = env[-1]

initial_vision = VisionHull(start_point).intersect_with_world(world)
foot = "right" if foot == "left" else "left"

root = HullNode(hull=None, parent=None, children=[])
print('1')
for initial_section in initial_vision:
    plot_hull(initial_section, color='red')
    root.add_child(initial_section)

for hull in world:
    plot_hull(hull)
print('3')
plt.axis("scaled")
plt.show()

while False:

    new_walk_regions = []
    for region in walk_regions:
        print('3,4')
        plot_hull(region, color="red")
        print('3.5')
        plt.show()
        print('4')
        new_walk_regions.extend(region.get_children_hulls(world))
        print('5')
    walk_regions = new_walk_regions
    foot = "right" if foot == "left" else "left"
    for region in walk_regions:
        if region.contains(end_point):
            print("success")
            break
