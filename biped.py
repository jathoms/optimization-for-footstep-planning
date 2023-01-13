import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from sklearn import cluster
import io
import sys

m = 16  # number of wanted regions (default)
n = 2  # dimension of space
steps = 1  # default number of steps in solution
first_foot_forward = 'left'
min_foot_separation_h = 0.1
default_dist = 0.6

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


def get_constraints(model: gp.Model,
                    all_hulls=hulls,
                    start=points[0],
                    end=obj_coords,
                    no_regions=m,
                    steps_taken=steps,
                    decreasing_steps=False,
                    reachable_distance=default_dist):

    all_constrs = np.array(
        [np.array([coords[:n] for coords in hull.equations]) for hull in all_hulls], dtype=object)
    all_rhs = np.array(
        [np.array([coords[-1] for coords in hull.equations]) for hull in all_hulls], dtype=object)
    A = all_constrs
    b = all_rhs
    print(steps_taken, "STEPS TAKEN")
    x: gp.MVar = model.addMVar((steps_taken, n), lb=-
                               gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
    dists: gp.MVar = model.addMVar(
        (steps_taken, n), ub=gp.GRB.INFINITY, lb=-gp.GRB.INFINITY)
    active = model.addVars(steps_taken, no_regions, vtype=gp.GRB.BINARY)

    rhs = [[] for _ in range(steps_taken)]
    for i in range(steps_taken):
        for val in b:
            rhs[i].append(model.addMVar(val.shape, lb=-
                                        gp.GRB.INFINITY, ub=gp.GRB.INFINITY))
        model.addConstr(sum([active[i, j] for j in range(no_regions)]) == 1)

    model.addConstr(x[-1] == end)
    model.addConstr(x[0] == start)

    # bipedality constraint.
    if first_foot_forward == 'right':
        model.addConstrs(x[i][0] <= x[i+1][0] -
                         min_foot_separation_h for i in range(0, steps_taken-1, 2))
        model.addConstrs(x[i+1][0] >= x[i+2][0] +
                         min_foot_separation_h for i in range(0, steps_taken-2, 2))
    else:
        model.addConstrs(x[i][0] >= x[i+1][0] +
                         min_foot_separation_h for i in range(0, steps_taken-1, 2))
        model.addConstrs(x[i+1][0] <= x[i+2][0] -
                         min_foot_separation_h for i in range(0, steps_taken-2, 2))
    for i in range(steps_taken - 1):
        model.addConstr(dists[i] == (x[i] - x[i+1]))
        model.addConstr(dists[i] @ dists[i] <= reachable_distance**2)
        # model.addConstr(dists[i] @ dists[i] >= foot_size)
        for j in range(no_regions):
            for k in range(len(b[j])):
                model.addConstr(
                    rhs[i][j][k] == -(b[j][k]) + ((1 - active[i, j]) * M))
            model.addConstr(A[j] @ x[i] <= rhs[i][j])
    buffer = io.StringIO()
    # print(f"buffering for {steps_taken} steps")
    sys.stdout = buffer
    model.optimize()
    print(steps_taken, "steps taken.")
    print(model.NodeCount, " nodes traversed.")
    sys.stdout = sys.__stdout__
    output = buffer.getvalue()

    if model.Status == gp.GRB.INFEASIBLE:
        print(f"Problem is infeasible for {steps_taken} steps.")
        if steps_taken > 100:
            return
        if not decreasing_steps:
            get_constraints(gp.Model(f'{steps_taken*2}_steps'), all_hulls, start, end,
                            no_regions, steps_taken*2, reachable_distance=reachable_distance)
            return
        else:
            return model.Status
    decrease_amount = 15/16
    print(
        f"Feasible solution found for {steps_taken} steps, attempting to reduce number of steps (factor {decrease_amount})")
    res = get_constraints(gp.Model(f'{steps_taken*decrease_amount}_steps'), all_hulls, start, end,
                          no_regions, int(steps_taken*(decrease_amount)), decreasing_steps=True, reachable_distance=reachable_distance)
    if res == gp.GRB.INFEASIBLE:
        # print("@@@@@@@@@@@@@@@@@@@\n", output, "\n@@@@@@@@@@@@@@@@@@@@")
        try:
            open("log.txt", "w").writelines(output)
            for idx, point in enumerate(x):
                x, y = point.X[0], point.X[1]

                if idx % 2 == 0:
                    plt.plot(x, y, marker="x", markersize=10,
                             markerfacecolor="blue", markeredgecolor="blue")
                else:
                    plt.plot(x, y, marker="x", markersize=10,
                             markerfacecolor="green", markeredgecolor="green")
                # print("Made step at:", point.X)
            # for i in range(steps):
            #     print("Step region {}: ".format(i),
            #           ([active[i, j].X for j in range(m)]))
            plt.plot(end[0], end[1], marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red",
                     alpha=0.5)
            plt.plot(start[0], start[1], marker="o", markersize=10, markeredgecolor="green", markerfacecolor="green",
                     alpha=0.5)
            print("Start Point:", start, "\nEnd Point:", end)
            print(
                f"Near optimal with {steps_taken} steps (within {int(steps_taken - (steps_taken*decrease_amount))} steps)")
        except gp.GurobiError:
            print(f"Problem is infeasible for {steps_taken} steps.")
            if steps_taken > 100:
                return
            if not decreasing_steps:
                get_constraints(gp.Model(f'{steps_taken*2}_steps'), all_hulls, start, end,
                                no_regions, steps_taken*2, reachable_distance=reachable_distance)
                return
            else:
                return model.Status
    return -1


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
