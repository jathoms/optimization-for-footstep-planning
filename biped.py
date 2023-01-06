import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from sklearn import cluster

m = 16  # number of wanted regions
n = 2  # dimension of space
steps = 29  # number of steps in solution
first_foot_forward = 'left'
min_foot_separation_h = 0.1
reachable_distance = 1.7

rng = np.random.default_rng(seed=70)
# m * 50 random points in n-D

points = 5 * rng.random((100 * m, n))
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


def get_constraints(model: gp.Model, all_hulls=hulls, start=points[0], end=obj_coords, no_regions=m):
    all_constrs = np.array(
        [np.array([coords[:n] for coords in hull.equations]) for hull in all_hulls], dtype=object)
    all_rhs = np.array(
        [np.array([coords[-1] for coords in hull.equations]) for hull in all_hulls], dtype=object)
    A = all_constrs
    b = all_rhs
    for hull in all_hulls:
        print(hull.equations)

    x = model.addMVar((steps, n), lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
    dists = model.addMVar((steps, n), ub=gp.GRB.INFINITY, lb=-gp.GRB.INFINITY)
    active = model.addVars(steps, no_regions, vtype=gp.GRB.BINARY)

    rhs = [[] for _ in range(steps)]
    for i in range(steps):
        for val in b:
            rhs[i].append(model.addMVar(val.shape, lb=-
                                        gp.GRB.INFINITY, ub=gp.GRB.INFINITY))
        model.addConstr(sum([active[i, j] for j in range(no_regions)]) == 1)

    model.addConstr(x[-1] == end)
    model.addConstr(x[0] == start)

    # bipedality constraint.
    if first_foot_forward == 'right':
        model.addConstrs(x[i][0] <= x[i+1][0] -
                         min_foot_separation_h for i in range(0, steps-1, 2))
        model.addConstrs(x[i+1][0] >= x[i+2][0] +
                         min_foot_separation_h for i in range(0, steps-2, 2))
    else:
        model.addConstrs(x[i][0] >= x[i+1][0] +
                         min_foot_separation_h for i in range(0, steps-1, 2))
        model.addConstrs(x[i+1][0] <= x[i+2][0] -
                         min_foot_separation_h for i in range(0, steps-2, 2))
    for i in range(steps - 1):
        model.addConstr(dists[i] == (x[i] - x[i+1]))
        model.addConstr(dists[i] @ dists[i] <= reachable_distance**2)
        # model.addConstr(dists[i] @ dists[i] >= foot_size)
        for j in range(no_regions):
            for k in range(len(b[j])):
                model.addConstr(
                    rhs[i][j][k] == -(b[j][k]) + ((1 - active[i, j]) * M))
            model.addConstr(A[j] @ x[i] <= rhs[i][j])
    model.optimize()
    plt.plot(end[0], end[1], marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red",
             alpha=0.5)
    plt.plot(start[0], start[1], marker="o", markersize=10, markeredgecolor="green", markerfacecolor="green",
             alpha=0.5)
    print("Start Point:", start, "\nEnd Point:", end)
    try:
        for idx, point in enumerate(x):
            if idx % 2 == 0:
                plt.plot(point.X[0], point.X[1], marker="x", markersize=10,
                         markerfacecolor="blue", markeredgecolor="blue")
            else:
                plt.plot(point.X[0], point.X[1], marker="x", markersize=10,
                         markerfacecolor="green", markeredgecolor="green")
            # print("Made step at:", point.X)
        # for i in range(steps):
        #     print("Step region {}: ".format(i),
        #           ([active[i, j].X for j in range(m)]))
    except gp.GurobiError:
        print("Failed to find path, problem is infeasible.")
        pass
    return


def plot_hull(hull_vertices):
    plt.axes()
    for idx, hull in enumerate(hulls):
        vertices_ = hull_vertices[label == idx]
        # plt.scatter(vertices_, label=idx)
        for simplex in hull.simplices:
            plt.plot(vertices_[simplex, 0], vertices_[simplex, 1], 'k-')
    return


# plot_hull(points)
# get_constraints(model, hulls)
