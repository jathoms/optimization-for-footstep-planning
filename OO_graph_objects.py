from scipy.spatial import ConvexHull
from scipy.spatial import HalfspaceIntersection
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import linprog

import time
from hierarchy_tree import hierarchy_pos
import networkx as nx


walkable_regions = []
check_for_redundant_regions = True
plot = False
plot_graph = False
regionparentmap = {}

config = [0.6, 16, "left", 0.1]

[reachable_distance, no_points, default_foot, offset] = config


class HullSection(ConvexHull):
    def __init__(self, parent_hull: ConvexHull, vertices: np.array(float), foot_in_this_hull="left"):  # default is important
        super().__init__(vertices)

        self.parent_hull = parent_hull
        self.foot_in = foot_in_this_hull
        self.vision = VisionHull(
            self, "right" if foot_in_this_hull == "left" else "left")

    def get_vertices(self):
        return [self.points[v] for v in self.vertices]

    def contains(self, point: list[float],  tolerance=1e-12):
        return all(
            (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
            for eq in self.equations)


class HullNode():
    def __init__(self, hull: HullSection | None, parent, children, depth: int, starthull=None):
        self.hull = hull if hull else HullSection(
            parent_hull=starthull, vertices=starthull.points)
        self.parent = parent
        self.children: list[HullNode] = children
        self.depth = depth

    def add_child_from_hullsection_intersection_with_world(self, child: HullSection):
        child = HullNode(hull=child, parent=self,
                         children=[], depth=self.depth+1)
        self.children.append(
            child)
        return child

    def add_child_from_hullsection_intersection_with_node(self, child: HullSection, node):
        child.parent_hull = node.hull.parent_hull
        child = HullNode(hull=child, parent=self,
                         children=[], depth=self.depth+1)
        self.children.append(
            child)
        return child

    def add_child_node(self, node):
        self.children.append(node)
        node.parent = self

    def get_children_hulls(self, environment: list[ConvexHull], explored):
        return self.hull.vision.intersect_with_world(environment, explored)

    def add_children(self, children):
        for child in children:
            self.add_child_from_hullsection_intersection_with_world(child)

    def remove_child(self, child):
        if child in self.children:
            self.children.remove(child)


class RootHull():
    def __init__(self, pos, env_region, foot_in_this_hull):
        self.parent_hull = env_region
        self.foot_in = foot_in_this_hull
        self.vision = VisionHull(
            pos, "right" if foot_in_this_hull == "left" else "left")

    def contains(a, b):
        return False


class RootNodePoint(HullNode):
    def __init__(self, pos, starthull, foot):
        self.hull = RootHull(pos, starthull, foot)
        self.parent = None
        self.children: list[HullNode] = []
        self.depth = 0
        self.pos = pos


class VisionHull(ConvexHull):
    def __init__(self, source: HullSection | list[float], foot):
        self.source = source
        self.foot = foot

        if isinstance(self.source, HullSection):
            extremities = [self.source.points[v] for v in self.source.vertices]
            super().__init__(np.vstack([self.linearise_reachable_region(
                x) for x in extremities]))
            # plot_hull(self, color="red")

        else:
            super().__init__(self.linearise_reachable_region(self.source))

    def linearise_reachable_region(self, centre=[0, 0]):
        # global foot
        global no_points
        global reachable_distance
        global offset
        global add_redundant_regions
        x = []
        y = []

        offset_angle = math.asin((offset) /
                                 math.sqrt(reachable_distance**2 + offset**2))

        if self.foot == 'left':
            initial_angle = math.pi/2 + offset_angle
        else:
            initial_angle = 3*math.pi/2 + offset_angle

        for i in range(no_points+1):
            x.append(centre[0] + math.cos(initial_angle +
                                          (math.pi - 2*offset_angle)*(i/no_points)) * reachable_distance)
            y.append(centre[1] + math.sin(initial_angle +
                                          (math.pi - 2*offset_angle)*(i/no_points))*reachable_distance)
        # add first points at the end for plot to look closed.
        x.append(centre[0] +
                 math.cos(initial_angle) * reachable_distance)
        y.append(centre[1] + math.sin(initial_angle)*reachable_distance)

        return np.array(list(zip(x, y)))

    def hull_intersection(self, env_region: ConvexHull):
        t1 = time.perf_counter()
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
            # print("intersection took: ", time.perf_counter() - t1)
            return None
        # print("intersection took: ", time.perf_counter() - t1)
        return hull_section

    def intersect_with_world(self, environment: list[ConvexHull], explored: list[HullSection]) -> list[HullSection]:
        newfrontier = []
        # explored_hulls = [e.hull for e in]
        for hull in environment:
            i = self.hull_intersection(hull)
            if i:
                if not check_for_redundant_regions:
                    newfrontier.append(i)
                    continue
                same_parent_regions = [
                    x for x in explored+newfrontier if x.parent_hull is i.parent_hull]
                if not same_parent_regions:
                    newfrontier.append(i)
                    continue
                if all([self.adds_to_region(explored_region, i) for explored_region in same_parent_regions]):
                    newfrontier.append(i)
                for explored_region in same_parent_regions:
                    if self.adds_to_region(explored_region, i):
                        continue
                    else:
                        regionparentmap.setdefault(explored_region, [])
                        regionparentmap[explored_region].append(i)
        # intersection of convex sets is convex, so we are chilling
        return newfrontier

    def adds_to_region(self, explored_region, i):
        if i.foot_in != explored_region.foot_in:
            return True
        distinction_check_volume = ConvexHull(
            np.vstack([i.points, explored_region.points]))

        if distinction_check_volume.volume > explored_region.volume:
            return True
        return False


# env = np.array([[0, 0], [-0.5, 1], [0.5, 1]])

# env = np.array([[18.8, 1.7],  # 0.6 env3
#                 [18.8, 3.4],
#                 [18.85, 5.55],
#                 [18.65, 8.15],
#                 [19.55, 3.25],
#                 [20.4, 4],
#                 [21.15, 4.65],
#                 [21.65, 5.45],
#                 [21.8, 6.3],
#                 [21.8, 7.65],
#                 [21.8, 8.55],
#                 [19.75, 6.05],
#                 [19.9, 6.5],
#                 [20.1, 7.25],
#                 [20.2, 8],
#                 [18.2, 5.45],
#                 [17.85, 5.85],
#                 [17.85, 6.4],
#                 [17.5, 7.65],
#                 [17.6, 3.7],
#                 [16.2, 4.25],
#                 [16.1, 6.1],
#                 [16, 7.15],
#                 [16.05, 5.25],
#                 [15.95, 7.85],
#                 [15.55, 8.9],
#                 [16.2, 10.1],
#                 [16.65, 11.2],
#                 [21.95, 9.35],
#                 [22.25, 9.75],
#                 [23.2, 10.7],
#                 [24.2, 11.45],
#                 [19.3, 9.25],
#                 [19.45, 10.5],
#                 [18.6, 4.4],
#                 [18.75, 2.65],
#                 [24.1, 12.4],
#                 [23.2, 13.3],
#                 [22.55, 13.7],
#                 [17.05, 11.8],
#                 [17.65, 12],
#                 [18.2, 12.05],
#                 [19.05, 12.7],
#                 [19.95, 13.3],
#                 [19.85, 11.5],
#                 [20.2, 12.35],
#                 [20.95, 8],
#                 [21.15, 6.25],
#                 [20.95, 7],
#                 [21.85, 10.55],
#                 [21.9, 11.3],
#                 [22.05, 12.2],
#                 [20.15, 9.1],
#                 [22.8, 14.65]])


# env = np.array([[18.65, 14.8],  # 1.5 spiral?
#                 [15.75, 14.85],
#                 [21.1, 14.25],
#                 [23.65, 12.85],
#                 [24.9, 10.7],
#                 [25.25, 8.35],
#                 [24.8, 5.95],
#                 [23.3, 4.5],
#                 [21.05, 3.55],
#                 [18.45, 3.1],
#                 [16.05, 3],
#                 [13.9, 3.55],
#                 [11.9, 4.7],
#                 [10.35, 6.35],
#                 [9.85, 8.1],
#                 [10.25, 9.95],
#                 [11.7, 11.45],
#                 [13.4, 11.75],
#                 [15.05, 11.65],
#                 [16.7, 11.25],
#                 [18.05, 10.75],
#                 [19.2, 9.85],
#                 [19.8, 8.5],
#                 [19.25, 7.5],
#                 [17.8, 7.1],
#                 [16.8, 7.15],
#                 [15.25, 7.45],
#                 [17.1, 15.1],
#                 [19.9, 14.7],
#                 [22.35, 13.45],
#                 [24.35, 11.7],
#                 [25.05, 9.55],
#                 [25, 7.25],
#                 [23.7, 5.1],
#                 [22.15, 4.15],
#                 [19.7, 3.17],
#                 [17.25, 3.15],
#                 [14.9, 3.2],
#                 [12.9, 4.25],
#                 [11.25, 5.5],
#                 [10, 7.25],
#                 [10, 8.8],
#                 [10.8, 10.6],
#                 [12.3, 11.55],
#                 [15.05, 6.3],
#                 [15.8, 5.6],
#                 [16.85, 5.15],
#                 [18.3, 5.2],
#                 [19.55, 5.4],
#                 [20.6, 5.9],
#                 [21.2, 6.65],
#                 [21.65, 7.45],
#                 [21.9, 8.5],
#                 [21.75, 9.5],
#                 [21.5, 10.45],
#                 [21, 11.35],
#                 [20.4, 12.05],
#                 [19.5, 12.5]])


# env = np.array([[14.25, 14.6],
#                 [16.15, 14.95],
#                 [18.45, 14.95],
#                 [20.19, 15.41],
#                 [20.25, 14],
#                 [21.55, 11.9],
#                 [21.35, 10.05],
#                 [21.81, 13.37],
#                 [20.15, 8.9],
#                 [18.5, 8.3],
#                 [17.05, 8.35],
#                 [17.3, 15.8],
#                 [15.95, 8.85],
#                 [14.85, 10.1],
#                 [14.7, 11.65],
#                 [17.0, 11.65],
#                 [18.7, 11.65], ])

# env = np.array([[6.75, 8.85],
#                 [17.7, 8.2],
#                 [14.65, 8.25],
#                 [11.9, 8.35],
#                 [9.25, 8.55],
#                 [22.55, 10.55],
#                 [20.55, 8.55],
#                 [20.65, 12.7],
#                 [17.85, 12.05],
#                 [15.6, 12.6],
#                 [12.6, 13.1],
#                 [10.35, 13.6],
#                 [7.75, 13.4],
#                 [4.95, 12.95],
#                 [2.4, 12.35]])


# world = [createSquare(center, 0.3, False) for center in env]


[reachable_distance, no_points, foot, offset] = config


def cleanup(regions: list[HullSection], world):
    # print(len(regions), "b")
    for hull in world:
        same_parent_regions = [
            x for x in regions if x.parent_hull is hull]
        for region in same_parent_regions:
            for region2 in same_parent_regions:
                distinction_check_volume = ConvexHull(
                    np.vstack([region2.points, region.points]))
                larger_region = region2 if region2.volume >= region.volume else region
                if distinction_check_volume.volume == larger_region.volume and not (region is region2):
                    if larger_region is region2:
                        if region in regions:
                            regions.remove(region)
    # print(len(regions))
    return regions


def cleanup_nodes(regions: list[HullNode], world):
    # print(len(regions), "b")
    for hull in world:
        same_parent_regions = [
            x for x in regions if x.hull.parent_hull is hull]
        for region in same_parent_regions:
            for region2 in same_parent_regions:
                distinction_check_volume = ConvexHull(
                    np.vstack([region2.hull.points, region.hull.points]))
                larger_region = region2.hull if region2.hull.volume >= region.hull.volume else region.hull
                if distinction_check_volume.volume == larger_region.volume and not (region is region2):
                    if larger_region is region2.hull:
                        if region in regions:
                            regions.remove(region)

    # print(len(regions))
    return regions


def steal_child(other_parent, child):
    if child in other_parent.children:
        return
    if child not in child.parent.children:
        return
    child.parent.children.remove(child)
    other_parent.children.append(child)
    child.parent = other_parent


def cleanup_nodes_change_parent(regions: list[HullNode], world):
    # print(len(regions), "b")
    for hull in world:
        same_parent_regions = [
            x for x in regions if x.hull.parent_hull is hull]
        for region in same_parent_regions:
            for region2 in same_parent_regions:
                distinction_check_volume = ConvexHull(
                    np.vstack([region2.hull.points, region.hull.points]))
                larger_region = region2 if region2.hull.volume >= region.hull.volume else region
                if distinction_check_volume.volume == larger_region.hull.volume and not (region is region2):
                    if larger_region is region2:
                        if region in regions:
                            regions.remove(region)
                            steal_child(larger_region.parent, region)
                        # alalregions.remove(region)


def cleanup_nodes_add_parent(regions: list[HullNode], world):
    # print(len(regions), "b")
    global regionparentmap
    for hull in world:
        same_parent_regions = [
            x for x in regions if x.hull.parent_hull is hull]
        for region in same_parent_regions:
            for region2 in same_parent_regions:
                distinction_check_volume = ConvexHull(
                    np.vstack([region2.hull.points, region.hull.points]))
                larger_region = region2 if region2.hull.volume >= region.hull.volume else region
                if distinction_check_volume.volume == larger_region.hull.volume and not (region is region2):
                    if larger_region is region2:
                        if region in regions:
                            larger_region.parent.children.append(region)
                            regionparentmap.setdefault(region, [])
                            regionparentmap[region].append(
                                larger_region.parent)
                            regions.remove(region)
                        # alalregions.remove(region)


def is_explored(region, explored):
    for ex in explored:
        if region.hull.parent_hull == ex.parent_hull:
            return False
        distinction_check_volume = ConvexHull(
            np.vstack([ex.points, region.hull.points]))
        larger_region = ex if ex.volume >= region.hull.volume else region.hull
        if distinction_check_volume.volume == larger_region.volume and not (region.hull is ex):
            if larger_region is ex:
                return True
    return False


# search from start to end, or end to start depending on reverse
def search(world, start, end, config, reverse=False):

    step = 2  # first step is counted as initial position of non-starting foot, first region generated is technically step 2
    global foot
    explored_walkable_regions_right = []
    explored_walkable_regions_left = []
    initial_vision = VisionHull(start, foot)
    initial_sections = initial_vision.intersect_with_world(world, [])
    if reverse:
        root = RootNodePoint(end, world[-1], default_foot)
    else:
        root = RootNodePoint(start, world[0], default_foot)
    for initial_section in initial_sections:
        root.add_child_from_hullsection_intersection_with_world(
            initial_section)
    current = root.children

    if check_for_redundant_regions:
        if foot == "left":
            explored_walkable_regions_left.extend(initial_sections)
        elif region.hull.foot_in == "right":
            explored_walkable_regions_right.extend(initial_sections)

    while True:

        print(f"spawned {len(current)} children")

        explored_walkable_regions_left = cleanup(
            explored_walkable_regions_left, world)
        explored_walkable_regions_right = cleanup(
            explored_walkable_regions_right, world)
        if check_for_redundant_regions:
            # current = cleanup_nodes(current, world)
            cleanup_nodes_change_parent(current, world)
        if plot:
            for region in world:
                plot_hull(region)
            for region in explored_walkable_regions_left:
                plot_hull(region, color="blue")
            for region in explored_walkable_regions_right:
                plot_hull(region, color="green")
            for region in current:
                plot_hull(region.hull, color="pink")
            plt.plot(start[0], start[1], "o", color='brown')
            plt.plot(end[0], end[1], "o", color="red")
            # for region in current:
            #     plot_hull(region.hull.vision, color='cyan', alpha=0.5)
            plt.axis("scaled")
            plt.show()
            draw_graph(root)
            pass
        print("taking step", step)
        for region in current:
            if region.hull.contains(end):
                if plot or plot_graph:
                    plt.axis("scaled")
                    plt.show()
                    path = draw_graph(root, region, True)

                print("acquired")
                return root, region
        new = []
        for region in current:

            # if is_explored(region, explored_walkable_regions_left if region.hull.foot_in == "left" else explored_walkable_regions_right):
            #     continue

            # t1 = time.perf_counter()
            region_vision = region.get_children_hulls(
                world, explored_walkable_regions_left if region.hull.foot_in == "right" else explored_walkable_regions_right)

            if region.hull.foot_in == "right":
                explored_walkable_regions_left.extend(region_vision)
            elif region.hull.foot_in == "left":
                explored_walkable_regions_right.extend(region_vision)
            region.add_children(region_vision)
            # plot_hull(region.hull.vision, color="orange")
            new.extend(region.children)

        if len(current) == 0:
            print('infeasible')
            return None
        # for r in world:
        #     plot_hull(r)
        # plot_hull(current[0].hull.vision, color='cyan', alpha=0.5)

        # for n in current[0].children:
        #     plot_hull(n.hull, color='pink')
        # plot_hull(current[0].hull, color='red')
        # plt.axis('scaled')
        # plt.show()
        current = new
        step += 1


def draw_graph(root: HullNode, end=None, highlight_path=False):
    path = []
    gr = nx.Graph()

    gr.add_node(root)

    def populate(root):
        if root is None:
            return
        for child in root.children:
            gr.add_node(child)
            gr.add_edge(root, child)
            populate(child)

        if highlight_path and root is end:
            node = root
            while node:
                path.append(node)
                node = node.parent

    populate(root)

    edge_colors = [
        "#00B2B2" if u in path and v in path else "#000000" for u, v in gr.edges]

    print(gr)
    plt.figure(dpi=500, figsize=(2, 1))
    if nx.is_tree(gr):
        pos = hierarchy_pos(gr, root)
        nx.draw(gr, pos=pos, node_size=0.5, edge_color=edge_colors, width=0.5,
                node_color=["#00B2B2" if node in path else '#B200B2' for node in gr])
    else:
        nx.draw(gr, node_size=5, edge_color=edge_colors, width=1,
                node_color=["#00B2B2" if node in path else '#B200B2' for node in gr])
    # colours = {root: "#FF0000", end: "000000"}

    # nx.draw(gr, node_color=[
    #         "#FF0000" if node.depth == 0 or node.depth == steps else "#000000" for node in gr], node_size=15)

    # nx.draw(gr, node_color=[
    #     "#FF0000" if node.depth == 0 or node.depth == steps + 1 else "#000000" for node in gr], node_size=15)
    plt.show()

    return path


def fill_graph_from_root(root):

    gr = nx.Graph()

    def populate(root):
        if root is None:
            return
        for child in root.children:
            gr.add_node(child)
            gr.add_edge(root, child)
            populate(child)

    populate(root)

    return gr


def plot_hull(hull, title="", color="black", alpha=1):
    if isinstance(hull, RootHull):
        return
    plt.title(title)
    vertices = hull.points
    for simplex in hull.simplices:
        plt.plot(vertices[simplex, 0], vertices[simplex, 1],
                 '-', color=color, alpha=alpha)

    # plt.show()
    return


def create_new_graph_in_search_of_other_graph(world, start_point, start_hull: ConvexHull, existing_graph: nx.Graph, start_foot):
    step = 2  # first step is counted as initial position of non-starting foot, first region generated is technically step 2
    foot = start_foot
    explored_walkable_regions_right = []
    explored_walkable_regions_left = []
    initial_vision = VisionHull(start_point, foot)
    initial_sections = initial_vision.intersect_with_world(world, [])
    root = RootNodePoint(start_point, start_hull, start_foot)
    for initial_section in initial_sections:
        root.add_child_from_hullsection_intersection_with_world(
            initial_section)
    current = root.children
    if check_for_redundant_regions:
        if foot == "left":
            explored_walkable_regions_left.extend(initial_sections)
        elif foot == "right":
            explored_walkable_regions_right.extend(initial_sections)

    while True:

        print(f"spawned {len(current)} children")

        explored_walkable_regions_left = cleanup(
            explored_walkable_regions_left, world)
        explored_walkable_regions_right = cleanup(
            explored_walkable_regions_right, world)
        if check_for_redundant_regions:
            cleanup_nodes_change_parent(current, world)
        if plot:
            for region in world:
                plot_hull(region)
            for region in explored_walkable_regions_left:
                plot_hull(region, color="blue")
            for region in explored_walkable_regions_right:
                plot_hull(region, color="green")
            for region in current:
                plot_hull(region.hull, color="pink")
            plt.plot(start_point[0], start_point[1], "o", color='brown')
            # for region in current:
            #     plot_hull(region.hull.vision, color='cyan', alpha=0.5)
            plt.axis("scaled")
            plt.show()
            draw_graph(root)
            pass
        print("taking step", step)
        for region in current:
            for node in existing_graph:
                if isinstance(node, RootNodePoint):
                    continue
                intersection = region.hull.vision.hull_intersection(node.hull)
                if intersection and intersection.foot_in == node.hull.foot_in:
                    new_node = region.add_child_from_hullsection_intersection_with_node(
                        intersection, node)
                    if plot:
                        for env_region in world:
                            plot_hull(env_region)
                        plot_hull(region.hull, color='#00B2B2')
                        plot_hull(node.hull, color='red')
                        plot_hull(region.hull.vision, color='cyan')
                        plot_hull(intersection, color='blue')
                        plt.axis("scaled")
                        # plt.show()
                        # draw_graph(root, intersection, True)
                    print("acquired")
                    return root, new_node, node  # root new, leaf new, node old
        new = []
        for region in current:
            region_vision = region.get_children_hulls(
                world, explored_walkable_regions_left if region.hull.foot_in == "right" else explored_walkable_regions_right)

            if region.hull.foot_in == "right":
                explored_walkable_regions_left.extend(region_vision)
            elif region.hull.foot_in == "left":
                explored_walkable_regions_right.extend(region_vision)
            region.add_children(region_vision)
            # plot_hull(region.hull.vision, color="orange")
            new.extend(region.children)

        if len(current) == 0:
            print('infeasible')
            return None
        current = new
        step += 1
