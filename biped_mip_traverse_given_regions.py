from graph_construction import linearise_reachable_region
import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from time import perf_counter

n = 2  # dimension
M = 1000  # big number


def create_active_matrix(steporder, regiondict):
    region_id_order = [regiondict[r.hull.parent_hull] for r in steporder]
    steps = len(steporder)
    no_regions = len(regiondict)
    active = np.full((steps, no_regions), 0)
    for i in range(steps):
        active[i, region_id_order[i]] = 1
    # for region in steporder:
    #     for r in world:
    #         plot_hull(r)
    #     plot_hull(region.hull.parent_hull, color="blue")
    #     plt.axis('scaled')
    #     plt.show()
    return active


def get_footstep_positions(model: gp.Model,
                           all_hulls,
                           start,
                           end,
                           min_foot_separation_h,
                           reachable_distance,
                           steporder,
                           regiondict,
                           startfoot,
                           logfile="log.txt"):

    no_regions = len(all_hulls)
    steps_taken = len(steporder)
    active_matrix = create_active_matrix(steporder, regiondict)

    all_constrs = np.array(
        [np.array([coords[:n] for coords in hull.equations]) for hull in all_hulls], dtype=object)
    all_rhs = np.array(
        [np.array([coords[-1] for coords in hull.equations]) for hull in all_hulls], dtype=object)
    A = all_constrs
    b = all_rhs

    contact_points_vector: gp.MVar = model.addMVar((steps_taken, n), lb=-
                                                   gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="contacts")
    dists: gp.MVar = model.addMVar(
        (steps_taken, n), ub=gp.GRB.INFINITY, lb=-gp.GRB.INFINITY)

    rhs = [[] for _ in range(steps_taken)]
    for i in range(steps_taken):
        for val in b:
            rhs[i].append(model.addMVar(val.shape, lb=-
                                        gp.GRB.INFINITY, ub=gp.GRB.INFINITY))
    # model.addConstr(active == active_matrix)

    model.addConstr(contact_points_vector[-1] == end, name="startpointconstr")
    model.addConstr(contact_points_vector[0] == start, name="endpointconstr")

    # bipedality constraint.
    if startfoot == 'right':
        model.addConstrs(contact_points_vector[i][0] <= contact_points_vector[i+1][0] -
                         min_foot_separation_h for i in range(0, steps_taken-1, 2))
        model.addConstrs(contact_points_vector[i+1][0] >= contact_points_vector[i+2][0] +
                         min_foot_separation_h for i in range(0, steps_taken-2, 2))
    else:
        model.addConstrs(contact_points_vector[i][0] >= contact_points_vector[i+1][0] +
                         min_foot_separation_h for i in range(0, steps_taken-1, 2))
        model.addConstrs(contact_points_vector[i+1][0] <= contact_points_vector[i+2][0] -
                         min_foot_separation_h for i in range(0, steps_taken-2, 2))
    # reachability constraint
    for i in range(steps_taken - 1):
        model.addConstr(dists[i] == (
            contact_points_vector[i] - contact_points_vector[i+1]))
        model.addConstr(dists[i] @ dists[i] <= reachable_distance**2)
    # environment constraints
    for i in range(steps_taken):
        for j in range(no_regions):
            for k in range(len(b[j])):
                model.addConstr(
                    rhs[i][j][k] == -(b[j][k]) + ((1 - active_matrix[i, j]) * M))
            model.addConstr(A[j] @ contact_points_vector[i] <= rhs[i][j])

    t1 = perf_counter()
    model.optimize()
    time_taken = perf_counter() - t1
    print("time taken:", time_taken, "steps taken:",
          steps_taken, file=open(logfile, 'a'))
    # print(active_matrix.shape)
    # for i in range(steps_taken):
    #     print(i, end="\t")
    #     for j in range(no_regions):
    #         print(active_matrix[i, j], end=' ')
    #     print('\n', end="")
    return [(point.X[0], point.X[1]) for point in contact_points_vector]
