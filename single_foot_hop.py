import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from sklearn import cluster

m = 50  # number of wanted regions
n = 2  # dimension of space
steps = 17  # number of steps in solution

rng = np.random.default_rng(seed=70)
# m * 50 random points in n-D

points = 5 * rng.random((10 * m, n))
kmeans = cluster.KMeans(n_clusters=m)
label = kmeans.fit_predict(points)
u_labels = np.unique(label)
while m != len(u_labels):
    kmeans = cluster.KMeans(n_clusters=m)
    label = kmeans.fit_predict(points)
    u_labels = np.unique(label)
# obj_coords = rng.random(n) * 3
obj_coords = points[15]
hulls = [ConvexHull(points[label == i]) for i in u_labels]
M = 1000
reachable_distance = 0.1


def get_constraints(model: gp.Model, vertices):
    plot_hull(vertices)
    all_constrs = np.array(
        [np.array([coords[:n] for coords in hull.equations]) for hull in hulls], dtype=object)
    all_rhs = np.array(
        [np.array([coords[-1] for coords in hull.equations]) for hull in hulls], dtype=object)
    A = all_constrs
    b = all_rhs

    x = model.addMVar((steps, n), lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
    dists = model.addMVar((steps, n), ub=gp.GRB.INFINITY, lb=-gp.GRB.INFINITY)
    active = model.addVars(steps, m, vtype=gp.GRB.BINARY)

    rhs = [[] for _ in range(steps)]
    for i in range(steps):
        for val in b:
            rhs[i].append(model.addMVar(val.shape, lb=-
                                        gp.GRB.INFINITY, ub=gp.GRB.INFINITY))
        model.addConstr(sum([active[i, j] for j in range(m)]) == 1)

    model.addConstr(x[-1] == obj_coords)
    model.addConstr(x[0] == points[0])
    for i in range(steps - 1):
        model.addConstr(dists[i] == (x[i] - x[i+1]))
        model.addConstr(dists[i] @ dists[i] <= reachable_distance)
        for j in range(m):
            for k in range(len(b[j])):
                model.addConstr(
                    rhs[i][j][k] == -(b[j][k]) + ((1 - active[i, j]) * M))
            model.addConstr(A[j] @ x[i] <= rhs[i][j])
    model.optimize()
    plt.plot(obj_coords[0], obj_coords[1], marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red",
             alpha=0.5)
    plt.plot(points[0][0], points[0][1], marker="o", markersize=10, markeredgecolor="green", markerfacecolor="green",
             alpha=0.5)
    print("Start Point:", points[0], "\nEnd Point:", obj_coords)
    try:
        for point in x:
            plt.plot(point.X[0], point.X[1], marker="x", markersize=10,
                     markerfacecolor="blue", markeredgecolor="blue")
            # print("Made step at:", point.X)
        # for i in range(steps):
        #     print("Step region {}: ".format(i),
        #           ([active[i, j].X for j in range(m)]))
    except gp.GurobiError:
        print("Failed to find point")
        pass
    plt.axis('scaled')
    plt.show()
    return None


def plot_hull(hull_vertices):
    plt.axes()
    for idx, hull in enumerate(hulls):
        vertices_ = hull_vertices[label == idx]
        # plt.scatter(vertices_, label=idx)
        for simplex in hull.simplices:
            plt.plot(vertices_[simplex, 0], vertices_[simplex, 1], 'k-')
    return


model = gp.Model('first')

get_constraints(model, points)
