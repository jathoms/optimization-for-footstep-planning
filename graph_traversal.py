from OO_graph_objects import *
import heapq
from collections import deque
from time import perf_counter
import create_environment as create
import biped_mip_traverse_given_regions as mip
import gurobipy as gp
import biped
import random

reverse_search = True
plot = True
fig6 = True

# env = np.array([[14.25, 14.6],  # 1.8 spiral?
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

# env = np.array([[17.325, 12.7],  # fu
#                 [2.325, 1.55],
#                 [38.925, 5.7],
#                 [16.675, 11.75],
#                 [16.075, 12.3],
#                 [16.175, 13.3],
#                 [17.025, 13.85],
#                 [17.925, 13.7],
#                 [18.675, 12.85],
#                 [18.425, 12.1],
#                 [17.825, 11.5],
#                 [2.275, 2.5],
#                 [1.275, 2.3],
#                 [1.275, 1.4],
#                 [1.875, 0.8],
#                 [2.725, 0.8],
#                 [3.175, 1.45],
#                 [3.025, 2.45],
#                 [38.375, 6.45],
#                 [39.275, 6.6],
#                 [40.025, 6.15],
#                 [40.025, 5.5],
#                 [39.475, 5],
#                 [38.175, 5.1],
#                 [38.075, 5.9],
#                 [16.825, 8.95],
#                 [16.825, 8.35],
#                 [16.825, 7.8],
#                 [16.775, 7.35],
#                 [16.875, 6.95],
#                 [17.275, 9.1],
#                 [17.625, 9.05],
#                 [17.875, 9.05],
#                 [17.225, 8.1],
#                 [17.575, 8.05],
#                 [17.875, 8.05],
#                 [19.225, 8.85],
#                 [19.225, 8.5],
#                 [19.175, 7.9],
#                 [19.375, 7.15],
#                 [19.925, 7],
#                 [20.125, 7.2],
#                 [20.275, 7.75],
#                 [20.275, 8.3],
#                 [20.275, 8.8],
#                 [22.225, 8.95],
#                 [21.925, 8.9],
#                 [21.825, 8.9],
#                 [21.625, 8.75],
#                 [21.375, 8.25],
#                 [21.375, 8],
#                 [21.375, 7.65],
#                 [21.525, 7.15],
#                 [21.775, 7],
#                 [22.175, 7],
#                 [22.525, 7.1],
#                 [23.775, 8.95],
#                 [23.775, 8.45],
#                 [23.775, 8.15],
#                 [23.725, 7.6],
#                 [23.875, 7.35],
#                 [23.875, 6.9],
#                 [24.775, 8.75],
#                 [24.525, 8.45],
#                 [24.325, 8],
#                 [24.275, 7.8],
#                 [24.675, 7.55],
#                 [25.125, 7.15],
#                 [25.225, 6.85],
#                 [17.375, 5],
#                 [17.675, 4.65],
#                 [17.775, 4.35],
#                 [18.325, 5],
#                 [18.175, 4.55],
#                 [17.775, 4.05],
#                 [17.475, 3.65],
#                 [17.325, 3.2],
#                 [20.025, 4.85],
#                 [19.675, 4.7],
#                 [19.375, 4.35],
#                 [19.375, 3.9],
#                 [19.575, 3.35],
#                 [20.225, 3.05],
#                 [20.775, 3.35],
#                 [20.825, 4.1],
#                 [20.775, 4.55],
#                 [20.425, 4.75],
#                 [22.025, 4.85],
#                 [22.025, 4.65],
#                 [22.025, 4.2],
#                 [22.025, 3.85],
#                 [22.025, 3.55],
#                 [22.125, 3.2],
#                 [22.425, 3.2],
#                 [22.675, 3.2],
#                 [22.925, 3.4],
#                 [22.925, 3.95],
#                 [22.925, 4.4],
#                 [22.975, 4.9],
#                 [25.525, 4.95],
#                 [26.825, 5.05],
#                 [24.975, 4.05],
#                 [25.025, 3.8],
#                 [25.375, 3.3],
#                 [25.775, 2.9],
#                 [26.275, 2.8],
#                 [26.775, 2.75],
#                 [27.375, 2.85],
#                 [27.575, 3.35],
#                 [27.825, 4],
#                 [1.775, 2.05],
#                 [2.775, 2],
#                 [2.625, 1.25],
#                 [1.675, 1.3],
#                 [17.075, 13.35],
#                 [17.725, 13],
#                 [17.875, 12.45],
#                 [17.275, 12.2],
#                 [16.775, 12.6],
#                 [16.825, 13.1],
#                 [39.525, 6.1],
#                 [38.675, 5.2],
#                 [38.925, 6.25],
#                 [39.575, 5.5],
#                 [7.575, 8.55],
#                 [7.625, 9.15],
#                 [6.925, 8.95],
#                 [6.925, 8.55],
#                 [7.175, 8],
#                 [7.875, 8],
#                 [8.225, 8.4],
#                 [8.275, 9],
#                 [8.175, 9.75],
#                 [7.275, 9.75],
#                 [6.575, 9.45],
#                 [6.425, 8.9],
#                 [6.425, 8.25],
#                 [6.675, 7.75],
#                 [7.325, 7.55],
#                 [8.375, 7.55],
#                 [8.675, 8.25],
#                 [7.875, 7.55],
#                 [8.775, 8.85],
#                 [8.775, 9.55],
#                 [33.125, 14.5],
#                 [32.825, 15.05],
#                 [32.525, 14.6],
#                 [32.625, 14.15],
#                 [33.075, 14.05],
#                 [33.775, 14.1],
#                 [33.775, 14.6],
#                 [33.425, 15.25],
#                 [34.175, 15.3],
#                 [34.375, 14.85],
#                 [34.375, 14.35],
#                 [34.125, 13.8],
#                 [33.225, 13.7],
#                 [32.425, 13.95],
#                 [31.975, 14.8],
#                 [33.025, 15.7],
#                 [32.675, 13.5],
#                 [32.525, 15.5],
#                 [33.775, 13.55],
#                 [2.975, 3.05],
#                 [4.025, 4.05],
#                 [4.875, 5.15],
#                 [5.925, 5.95],
#                 [7.075, 7.1],
#                 [9.025, 10.95],
#                 [10.025, 10.95],
#                 [10.975, 11],
#                 [11.975, 11],
#                 [12.875, 10.9],
#                 [13.825, 10.9],
#                 [14.775, 10.85],
#                 [15.875, 10.85],
#                 [16.925, 10.9],
#                 [18.925, 14.1],
#                 [19.925, 14],
#                 [20.875, 13.2],
#                 [21.875, 13.95],
#                 [22.825, 15],
#                 [23.975, 14.15],
#                 [24.725, 13.25],
#                 [25.825, 14],
#                 [26.775, 15.05],
#                 [28.025, 14.05],
#                 [28.825, 13.15],
#                 [29.875, 14.1],
#                 [30.675, 15],
#                 [31.725, 14.1],
#                 [35.075, 14.95],
#                 [35.825, 14.05],
#                 [35.925, 13],
#                 [34.975, 11.9],
#                 [34.075, 11],
#                 [34.025, 10.05],
#                 [34.975, 9.2],
#                 [36.175, 8.25],
#                 [36.925, 7.25],
#                 [36.925, 6.3],
#                 [37.575, 5.6]])

env = np.array([[6.75, 8.85],  # fig4, 3
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


# env = np.array([[15.775, 0.95],  # funnel, 1.8
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

# env = np.array([[15.775, 0.95],  # fig 5, 1.8
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


def fast_traverse_no_change(root: HullNode, end: HullNode):
    nodes_traversed = 0
    path = deque([])
    while end is not root:
        path.appendleft(end)
        end = end.parent
        nodes_traversed += 1
    path.appendleft(end)
    # print(f'fast traverse in {nodes_traversed}')
    return path


def fast_traverse_no_change_backwards(root: HullNode, end: HullNode):
    nodes_traversed = 0
    path = deque([])
    while end is not root:
        path.append(end)
        end = end.parent
        nodes_traversed += 1
    path.append(root)
    # print(f'fast traverse in {nodes_traversed}')
    return path


def fast_traverse_from_node_to_root(root: HullNode, node: HullNode):
    nodes_traversed = 0
    path = deque([])
    while node is not root:
        path.append(node)
        node = node.parent
        nodes_traversed += 1
    path.append(root)
    # print(f'fast traverse in {nodes_traversed}')
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
if fig6:
    world = [create.createRect([0, 5], 12, 4)]

    env = np.array([[0, 0], [0, 12]])

t1 = perf_counter()

startpos, endpos = env[0], env[-1]

setattr(HullNode, "__lt__", lambda self, other: get_distance(
    self, endpos) + self.depth < get_distance(other, endpos) + other.depth)


if reverse_search:

    root, end = search(world, endpos, startpos, [], reverse=reverse_search)

else:
    root, end = search(world, startpos, endpos, [], reverse=reverse_search)

constr_time = perf_counter() - t1
print(f"graph construction took {constr_time}")

i = 0
region_id_dict = {}
for region in world:
    region_id_dict[region] = i
    i += 1


# t1 = perf_counter()
# if reverse_search:
#     steporder = fast_traverse_no_change_backwards(root, end)
# else:
#     steporder = fast_traverse_no_change(root, end)
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


def run_mip_with_graph(root, end, foot=foot):

    model = gp.Model()

    if reverse_search:
        steporder = fast_traverse_no_change_backwards(root, end)
    else:
        steporder = fast_traverse_no_change(root, end)
    steps = len(steporder)

    # plot_path(steporder)

    start_foot = foot if (not reverse_search or steps % 2 == 1) else (
        "left" if foot == "right" else "right")

    contact_points = mip.get_footstep_positions(
        model, world, startpos, endpos, offset, reachable_distance, steporder, region_id_dict, start_foot, "hopefully.txt")

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
    plt.axis('scaled')
    plt.show()
    return model


def replan_mip_with_graph(model: gp.Model, node, root, foot, new_startpos, endpos):

    steporder = fast_traverse_no_change_backwards(root, node)
    steps = len(steporder)

    print(steporder)
    model = gp.Model()

    model.Params.OutputFlag = 0

    # contact_points = [(point.X[0], point.X[1]) for point in contacts]

    contact_points = mip.get_footstep_positions(
        model, world, new_startpos, endpos, offset, reachable_distance, steporder, region_id_dict, foot, "compare2.txt")

    if plot:
        for region in world:
            plot_hull(region)
        for idx, point in enumerate(contact_points):
            x, y = point
            if idx <= steps-1:  # plot lines between steps
                plt.plot([step[0] for step in contact_points[idx:idx+2]],
                         [step[1] for step in contact_points[idx:idx+2]], ":", color="black")
            if idx % 2 == 0:  # plot f foot
                if foot == "left":
                    plt.plot(x, y, marker="x", markersize=10,
                             markerfacecolor="blue", markeredgecolor="blue")
                elif foot == "right":
                    plt.plot(x, y, marker="x", markersize=10,
                             markerfacecolor="green", markeredgecolor="green")
            else:  # plot f' foot
                if foot == "right":
                    plt.plot(x, y, marker="x", markersize=10,
                             markerfacecolor="blue", markeredgecolor="blue")
                elif foot == "left":
                    plt.plot(x, y, marker="x", markersize=10,
                             markerfacecolor="green", markeredgecolor="green")
                # plot start and end points
            plt.plot(endpos[0], endpos[1], marker="o", markersize=6, markeredgecolor="red", markerfacecolor="red",
                     alpha=0.5)
            plt.plot(new_startpos[0], new_startpos[1], marker="o", markersize=6, markeredgecolor="green", markerfacecolor="green",
                     alpha=0.5)
        plt.axis('scaled')
        plt.show()
    return steps


def plot_path(steporder):
    for step in steporder:
        for reg in world:
            plot_hull(reg)
        plot_hull(step.hull.parent_hull, color='blue')
        plt.axis('scaled')
        plt.show()


def build_graph(root):

    gr = nx.Graph()
    gr.add_node(root)

    def populate(root):
        if root is None:
            return
        for child in root.children:
            gr.add_node(child)
            gr.add_edge(root, child, color="#FF0000")
            populate(child)

    populate(root)
    return gr


def find_node(pos, foot, gr: nx.Graph) -> HullNode:

    res = []
    for node in gr:
        if node.hull.contains(pos):
            res.append(node)
    correct_foot = [node for node in res if node.hull.foot_in != foot]
    if not correct_foot:
        # print('no directly suitable node found, exiting...')
        raise KeyError
    optimal_region = min([node for node in correct_foot],
                         key=lambda x: x.depth)
    if plot:
        for region in world:
            plot_hull(region)
        plt.plot(pos[0], pos[1], "o", color="green")
        plot_hull(optimal_region.hull, color='green')
        plt.plot(pos[0], pos[1], "o", color="green")
        plt.axis('scaled')
        plt.show()
    return optimal_region


def replan(model, new_pos, foot, end_node, graph):
    new_node = find_node(new_pos, foot, graph)
    if plot:
        draw_graph(root, new_node, True)
    steps = replan_mip_with_graph(
        model, new_node, end_node, foot, new_pos, endpos)
    return steps


model = run_mip_with_graph(root, end)


gr = build_graph(root)


def gen_data_unexpected_in_graph():
    print(f"beginning fig ----- ({constr_time})\n",
          file=open('compare.txt', 'a'))
    for i in range(5, 26):
        try:
            new_idx = (i * 378467) % (len(env)-2)
            new_pos = env[new_idx]

            print(new_pos)

            next_step = "right" if (i * 10039) % 11 >= 6 else "left"
            # print(new_idx, new_pos, endpos, next_step)

            no_steps = replan(model, new_pos, next_step, root, gr)

            new_model = gp.Model()

            biped.get_constraints(
                new_model, world, start=new_pos, end=endpos, no_regions=len(world), steps_taken=no_steps, reachable_distance=reachable_distance, logfile='compare.txt', foot=next_step)

            print("", file=open('compare.txt', 'a'))

        except KeyError:
            continue

    print("end\n", file=open('compare.txt', 'a'))


def results_unexpected_fig6():
    print(f"beginning fi6----- ({constr_time})\n",
          file=open('compare.txt', 'a'))
    for i in range(5, 26):
        try:
            x = random.uniform(-3.5, 3.5)
            y = random.uniform(-6, 14)

            new_pos = np.array([x, y])

            # plt.plot(x, y, 'o', color='green')
            # plt.plot(endpos[0], endpos[1], 'o', color='red')

            # for r in world:
            #     plot_hull(r)
            # plt.axis('scaled')
            # plt.show()

            next_step = "right" if (i * 10039) % 11 >= 6 else "left"

            # print(new_idx, new_pos, endpos, next_step)

            no_steps = replan(model, new_pos, next_step, root, gr)

            new_model = gp.Model()

            biped.get_constraints(
                new_model, world, start=new_pos, end=endpos, no_regions=len(world), steps_taken=no_steps, reachable_distance=reachable_distance, logfile='compare2.txt', foot=next_step)

            print("", file=open('compare2.txt', 'a'))
        except KeyError:
            continue

    print("end\n", file=open('compare2.txt', 'a'))


results_unexpected_fig6()
