import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from sklearn import cluster
from graph_construction import linearise_reachable_region
import io
import sys
from time import perf_counter

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
                    reachable_distance=default_dist,
                    logfile="log.txt",
                    foot=first_foot_forward):

    all_constrs = np.array(
        [np.array([coords[:n] for coords in hull.equations]) for hull in all_hulls], dtype=object)
    all_rhs = np.array(
        [np.array([coords[-1] for coords in hull.equations]) for hull in all_hulls], dtype=object)
    A = all_constrs
    b = all_rhs
    print(steps_taken, "STEPS TAKEN")
    contact_points_vector: gp.MVar = model.addMVar((steps_taken, n), lb=-
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

    model.addConstr(contact_points_vector[-1] == end)
    model.addConstr(contact_points_vector[0] == start)

    # bipedality constraint.
    if foot == 'right':
        model.addConstrs(contact_points_vector[i][0] <= contact_points_vector[i+1][0] -
                         min_foot_separation_h for i in range(0, steps_taken-1, 2))
        model.addConstrs(contact_points_vector[i+1][0] >= contact_points_vector[i+2][0] +
                         min_foot_separation_h for i in range(0, steps_taken-2, 2))
    else:
        model.addConstrs(contact_points_vector[i][0] >= contact_points_vector[i+1][0] +
                         min_foot_separation_h for i in range(0, steps_taken-1, 2))
        model.addConstrs(contact_points_vector[i+1][0] <= contact_points_vector[i+2][0] -
                         min_foot_separation_h for i in range(0, steps_taken-2, 2))
    for i in range(steps_taken - 1):
        model.addConstr(dists[i] == (
            contact_points_vector[i] - contact_points_vector[i+1]))
        model.addConstr(dists[i] @ dists[i] <= reachable_distance**2)
        # model.addConstr(dists[i] @ dists[i] >= foot_size)
    for i in range(steps_taken):
        for j in range(no_regions):
            for k in range(len(b[j])):
                model.addConstr(
                    rhs[i][j][k] == -(b[j][k]) + ((1 - active[i, j]) * M))
            model.addConstr(A[j] @ contact_points_vector[i] <= rhs[i][j])
    buffer = io.StringIO()
    # print(f"buffering for {steps_taken} steps")
    sys.stdout = buffer
    t1 = perf_counter()
    model.optimize()
    time_taken = perf_counter() - t1
    print('time taken:',  time_taken, file=open(logfile, "a"))
    return
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
                            no_regions, steps_taken*2, reachable_distance=reachable_distance, logfile=logfile)
            return
        else:
            return model.Status
    decrease_amount = 15/16
    print(
        f"Feasible solution found for {steps_taken} steps, attempting to reduce number of steps (factor {decrease_amount})")
    res = get_constraints(gp.Model(f'{steps_taken*decrease_amount}_steps'), all_hulls, start, end,
                          no_regions, int(steps_taken*(decrease_amount)), decreasing_steps=True, reachable_distance=reachable_distance, logfile=logfile)
    if res == gp.GRB.INFEASIBLE:

        # print("@@@@@@@@@@@@@@@@@@@\n", output, "\n@@@@@@@@@@@@@@@@@@@@")
        try:
            open(logfile, "w").writelines(output)
            for idx, point in enumerate(contact_points_vector):
                x, y = point.X[0], point.X[1]
                # plot footstep positions and linearised version of reachable region from each step
                if idx <= steps_taken-1:
                    pass
                    plt.plot([step.X[0] for step in contact_points_vector[idx:idx+2]],
                             [step.X[1] for step in contact_points_vector[idx:idx+2]], ":", color="black")
                    # print(x, y)
                    # print([step.X[0]
                    #       for step in contact_points_vector[idx:idx+2]])
                    # #   [step.X[1] for step in contact_points_vector[idx:idx+1]])
                if idx % 2 == 0:
                    plt.plot(x, y, marker="x", markersize=10,
                             markerfacecolor="blue", markeredgecolor="blue")
                    linearise_reachable_region(reachable_distance, 10, [
                                               x, y], foot=('right' if foot == 'left' else 'right'), offset=min_foot_separation_h)
                else:
                    plt.plot(x, y, marker="x", markersize=10,
                             markerfacecolor="green", markeredgecolor="green")
                    linearise_reachable_region(reachable_distance, 10, [
                                               x, y], foot=('left' if foot == 'left' else 'right'), offset=min_foot_separation_h)

            plt.plot(end[0], end[1], marker="o", markersize=6, markeredgecolor="red", markerfacecolor="red",
                     alpha=0.5)
            plt.plot(start[0], start[1], marker="o", markersize=6, markeredgecolor="green", markerfacecolor="green",
                     alpha=0.5)
            print("Start Point:", start, "\nEnd Point:", end)
            print(
                f"Near optimal with {steps_taken} steps (within {int(steps_taken - (steps_taken*decrease_amount))} steps)")
            # for i in range(steps_taken):#print matrix
            #     print(i, end="\t")
            #     for j in range(no_regions):
            #         print(int(active[i, j].X), end=' ')
            #     print('\n', end="")

        except gp.GurobiError:
            print(f"Problem is infeasible for {steps_taken} steps.")
            if steps_taken > 100:
                return
            if not decreasing_steps:
                get_constraints(gp.Model(f'{steps_taken*2}_steps'), all_hulls, start, end,
                                no_regions, steps_taken*2, reachable_distance=reachable_distance, logfile=logfile)
                return
            else:
                return model.Status
    return -1


def plot_hull(hull):
    plt.axes()
    for idx, hull in enumerate(hulls):
        vertices_ = hull.points
        for simplex in hull.simplices:
            plt.plot(vertices_[simplex, 0], vertices_[simplex, 1], 'k-')
    return
