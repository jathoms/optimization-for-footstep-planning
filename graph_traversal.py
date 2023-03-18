from OO_graph_objects import *
from scipy.spatial import ConvexHull
import heapq
from collections import deque
from time import perf_counter
import timeit
import create_environment as create
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
from hierarchy_tree import hierarchy_pos
import biped_mip_traverse_given_regions as mip
import gurobipy as gp

env = np.array([[14.25, 14.6],  # 1.8 spiral?
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
                [17.0, 11.65],
                [18.7, 11.65], ])

# env = np.array([[6.75, 8.85],  # fig4, 3
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


# env = np.array([[15.775, 0.95],
#                 [14.625, 2.05],
#                 [16.775, 2.1],
#                 [17.825, 3.25],
#                 [18.775, 4.55],
#                 [19.575, 5.7],
#                 [20.475, 7.05],
#                 [15.525, 3],
#                 [13.525, 3.1],
#                 [12.575, 4.2],
#                 [14.175, 4.3],
#                 [16.125, 4.35],
#                 [17.325, 5.7],
#                 [18.475, 7.1],
#                 [15.825, 7.2],
#                 [15.125, 5.65],
#                 [13.025, 5.65],
#                 [14.025, 7.25],
#                 [12.425, 7.2],
#                 [11.525, 5.7],
#                 [10.525, 7.1],
#                 [9.725, 8.5],
#                 [9.025, 9.8],
#                 [8.275, 11.15],
#                 [21.175, 8.2],
#                 [21.675, 9.3],
#                 [22.425, 10.7],
#                 [22.925, 11.85],
#                 [19.425, 8.25],
#                 [17.325, 8.1],
#                 [14.925, 8.2],
#                 [13.025, 8.3],
#                 [11.425, 8.3],
#                 [10.475, 9.6],
#                 [12.025, 9.6],
#                 [13.625, 9.6],
#                 [16.025, 9.6],
#                 [18.225, 9.55],
#                 [19.875, 9.55],
#                 [20.575, 11.1],
#                 [18.675, 11.35],
#                 [16.975, 11.4],
#                 [15.175, 11.4],
#                 [13.225, 11.25],
#                 [10.525, 11.55],
#                 [15.325, 10.55],
#                 [21.125, 12.55],
#                 [19.725, 12.9],
#                 [18.525, 12.9],
#                 [16.825, 12.95],
#                 [14.675, 12.95],
#                 [12.825, 13],
#                 [11.175, 12.85],
#                 [8.775, 12.85],
#                 [11.925, 11.55],
#                 [15.875, 14.35],
#                 [13.875, 14.35],
#                 [11.775, 14.35],
#                 [17.975, 14.5],
#                 [16.775, 15.3],
#                 [15.525, 15.4],
#                 [13.075, 15.35],
#                 [13.825, 15.35],
#                 [13.875, 16.95],
#                 [15.875, 16.95],
#                 [14.925, 18.05]])

# env = np.array([[18.65, 14.8],  # big spiral, 1.5
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

# env = np.array([[18.8, 1.7],  # 0.6
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


def fast_traverse_no_change(start: HullNode, end: HullNode):
    nodes_traversed = 0
    path = deque([])
    while end is not start:
        path.appendleft(end)
        end = end.parent
        nodes_traversed += 1
    path.appendleft(end)
    print(f'fast traverse in {nodes_traversed}')
    return path


def fast_traverse_no_change_backwards(start: HullNode, end: HullNode):
    nodes_traversed = 0
    path = deque([])
    while end is not start:
        path.append(end)
        end = end.parent
        nodes_traversed += 1
    path.append(end)
    print(f'fast traverse in {nodes_traversed}')
    return path


def bfs_traverse(root: HullNode, end: HullNode):
    nodes_traversed = 0
    frontier = deque([])
    frontier.append(root)

    reached = set()
    reached.add(root)
    t1 = perf_counter()
    while frontier:
        # print("bfs:", perf_counter() - t1)
        t1 = perf_counter()
        current = frontier.pop()
        if current is end:
            print(f'bfs in {nodes_traversed}')
            return end
        for next in current.children:
            if next not in reached:
                frontier.append(next)
                reached.add(next)
                nodes_traversed += 1


def a_star_traverse(root: HullNode, end: HullNode):
    nodes_traversed = 0
    frontier = []
    for child in root.children:
        heapq.heappush(frontier, child)

    reached = set()
    reached.add(root)
    t1 = perf_counter()
    while frontier:
        # print("a*:", perf_counter() - t1)
        t1 = perf_counter()
        current = heapq.heappop(frontier)
        if current is end:
            print(f'a* in {nodes_traversed}')
            return end
        for next in current.children:
            if next not in reached:
                heapq.heappush(frontier, next)
                reached.add(next)
                nodes_traversed += 1


def greedy_traverse(root: HullNode, end: HullNode):
    nodes_traversed = 0
    path = []

    reached = set()
    reached.add(root)
    current = min([child for child in root.children])
    path.append(current)

    t1 = perf_counter()
    while end not in reached:
        if not current:
            print("got nowhere")
            return reached
        # print("greedy:", perf_counter() - t1)
        t1 = perf_counter()
        available = [
            child for child in current.children if child not in reached]
        if not available:
            current = current.parent
            continue
        next = min(available)
        current = next
        path.append(next)
        reached.add(next)
        nodes_traversed += 1
    print(f'greedy in {nodes_traversed}, path length: {len(path)+1}')
    return end


def centroid(c: HullNode):
    return np.mean(c.hull.points[c.hull.vertices, :], axis=0)


def get_distance(start: HullNode, end):
    return np.sqrt(np.sum(((centroid(start) - end) ** 2)))


def get_distance_region(start: HullNode, end: HullNode):
    return


world = [createSquare(center, 0.3, False) for center in env]

# world = [create.createRect([0, 5], 12, 4)]

# env = np.array([[0, 0], [0, 12]])

t1 = perf_counter()

startpos, endpos = env[0], env[-1]

root, end = search(world, startpos, endpos, [])

print(f"graph construction took {perf_counter() - t1}")

i = 0
regiondict = {}
for region in world:
    regiondict[region] = i
    i += 1


setattr(HullNode, "__lt__", lambda self, other: get_distance(
    self, endpos) + self.depth < get_distance(other, endpos) + other.depth)


# t1 = perf_counter()
# steporder = fast_traverse_no_change(root, end)
# print(perf_counter() - t1)

# t1 = perf_counter()
# bfs_traverse(root, end)
# print(perf_counter() - t1)

# t1 = perf_counter()
# a_star_traverse(root, end)
# print(perf_counter() - t1)

# t1 = perf_counter()
# greedy_traverse(root, end)
# print(perf_counter() - t1)


def draw_graph():

    gr = nx.Graph()

    gr.add_node(root)

    def populate(root):
        if root is None:
            return
        for child in root.children:
            gr.add_node(child)
            gr.add_edge(root, child)
            populate(child)

    populate(root)

    print(gr)

    pos = hierarchy_pos(gr, root)
    nx.draw(gr, pos=pos, node_size=10,
            node_color=["#00B2B2" if node is end else '#B200B2' for node in gr])

    # colours = {root: "#FF0000", end: "000000"}

    # nx.draw(gr, node_color=[
    #         "#FF0000" if node.depth == 0 or node.depth == steps else "#000000" for node in gr], node_size=15)

    # nx.draw(gr, node_color=[
    #     "#FF0000" if node.depth == 0 or node.depth == steps + 1 else "#000000" for node in gr], node_size=15)
    plt.show()


def run_mip_with_graph(root, end):

    model = gp.Model()

    t1 = perf_counter()

    steporder = fast_traverse_no_change_backwards(root, end)
    steps = len(steporder)
    contact_points = mip.get_footstep_positions(
        model, world, startpos, endpos, offset, reachable_distance, steporder, regiondict, "hopefully.txt")

    print(perf_counter() - t1, " to get footstep positions")

    for region in world:
        plot_hull(region)

    for idx, point in enumerate(contact_points):
        x, y = point
        # plot footstep positions and linearised version of reachable region from each step
        if idx <= steps-1:
            plt.plot([step[0] for step in contact_points[idx:idx+2]],
                     [step[1] for step in contact_points[idx:idx+2]], ":", color="black")
        if idx % 2 == 0:
            plt.plot(x, y, marker="x", markersize=10,
                     markerfacecolor="blue", markeredgecolor="blue")
        else:
            plt.plot(x, y, marker="x", markersize=10,
                     markerfacecolor="green", markeredgecolor="green")

        plt.plot(endpos[0], endpos[1], marker="o", markersize=6, markeredgecolor="red", markerfacecolor="red",
                 alpha=0.5)
        plt.plot(startpos[0], startpos[1], marker="o", markersize=6, markeredgecolor="green", markerfacecolor="green",
                 alpha=0.5)
    plt.show()


# run_mip_with_graph(root, end)

for r in world:
    plot_hull(r)

p = np.array([[16.8, 11.95], [16.7, 11.95], [16.7, 11.35], [16.8, 11.35]])

c = ConvexHull(p)

s = HullSection(c, p, 'right')

v = VisionHull(source=s, foot='left')

# plot_hull(v, color='green')

i = v.intersect_with_world(world, [])

print(i)

for ii in i:
    plot_hull(ii, color='pink')

plt.show()
