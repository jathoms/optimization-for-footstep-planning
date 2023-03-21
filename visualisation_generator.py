from OO_graph_objects import *
import matplotlib.patches as mpatches
import graph_construction as g
import numpy as np
import networkx as nx
from hierarchy_tree import hierarchy_pos


def plot_hull(hull, title="", color="black", alpha=1):
    plt.title(title)
    vertices = hull.points
    for simplex in hull.simplices:
        plt.plot(vertices[simplex, 0], vertices[simplex, 1],
                 '-', color=color, alpha=alpha)
    # plt.show()
    return


def plot_hull2(hull, title="", color="black"):
    plt.title(title)
    vertices = hull.points
    for simplex in hull.simplices:
        plt.plot(vertices[simplex, 0], vertices[simplex, 1],
                 '--', color=color)
    # plt.show()
    return


def left_and_right_foot_plot_from_origin(n: int):

    r = g.linearise_reachable_region(1, n, foot='right')
    l = g.linearise_reachable_region(1, n, foot='left')
    print("lin:", r.volume)

    plt.plot(0, 0, marker="o", markersize=10, color="red")

    plot_hull(r, color='blue',
              title="Linearised Reachable Region by Each Foot from the Origin.")
    plot_hull(l, color='green',
              title=f"Linearised Reachable Region by Each Foot from the Origin (n={n}, Δ=0.1, r=1).")
    lc = mpatches.Patch(color='green', label='Left foot')
    rc = mpatches.Patch(color='blue', label='Right foot')

    plt.legend(handles=[rc, lc])


def fig_with_curve_and_linear():
    left_and_right_foot_plot_from_origin(6)

    a = g.linearise_reachable_region(1, 5000)
    b = g.linearise_reachable_region(1, 50, foot='right')

    # plot_hull2(a, color='black')
    # plot_hull2(b, color='black')

    print("curve:", a.volume)


# def just_circle():
#     circ = mpatches.Circle((0.5, 0.5), radius=0.5)
#     ax = plt.figure(figsize=(5, 5))
#     ax.add_artist(circ)


# fig_with_curve_and_linear()


# plt.axis([-1.5, 1.5, -1.5, 1.5])
# plt.axis('equal')
# plt.grid(True, 'major')
# plt.show()

def volume_proportion_check(n):

    full_volume = g.linearise_reachable_region(1, 5000).volume
    plt.grid(True, 'both')
    l = []
    for k in range(2, n):
        a = g.linearise_reachable_region(1, k)
        # c = g.linearise_reachable_region(1, 10000)
        # plt.plot(n, c.volume)
        l.append(a.volume/full_volume)
    plt.plot(range(2, n), l)
    plt.plot(range(2, n), [1]*(n-2), '--')
    plt.ylabel('Proportion of area filled by linearised polygon.')
    plt.xlabel("n")
    plt.show()


def gen_circ():
    n = 10000
    angles = [m*2*math.pi/n for m in range(0, n)]
    xs = [math.cos(th) for th in angles]
    ys = [math.sin(th) for th in angles]
    return ConvexHull(np.array(list(zip(xs, ys))))


def circle_with_points():
    circ = g.linearise_reachable_region(1, 5000)
    # c = gen_circ()  # comment this and next line to not show full circle
    # plot_hull(c)
    plot_hull(circ)
    lin = g.linearise_reachable_region(1, 6)
    points = lin.points
    plt.grid(True)
    plt.plot([point[0] for point in points], [point[1]
             for point in points], "x", markersize=15, color="red", alpha=1)
    plt.plot(0, 0, marker="o", markersize=10, color="red")
    plt.axis('equal')
    plt.show()


# plt.figure(figsize=(5, 5))
# plt.axis('equal')
# fig_with_curve_and_linear()
# plt.grid(True)
# plt.show()
# circle_with_points()

# volume_proportion_check(100)
def draw_equivalent_points():
    phi = [1.8, -0.5]

    point1 = [0, 0]

    point2 = [2, 3]

    point3 = [3, 2]
    a = []
    for point in [point1, point2, point3]:
        a.append(g.linearise_reachable_region(3, 5, centre=point))

    plot_hull(a[0])
    plt.plot(point1[0] + phi[0], point1[1] + phi[1],
             "x", color="black", markersize=12)
    plot_hull(a[1], color="orange")
    plt.plot(point2[0] + phi[0], point2[1] + phi[1],
             "x", color="orange", markersize=12)
    plot_hull(a[2], color="purple")
    plt.plot(point3[0] + phi[0], point3[1] + phi[1],
             "x", color="purple", markersize=12)

    plt.axis("scaled")
    plt.grid(True)
    plt.show()


def plot_convex_hull():

    rng = np.random.default_rng(seed=70)

    points = 5 * rng.random((20, 2))
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    plt.plot(x, y, "o", color='red')
    plt.axis("equal")
    plot_hull(ConvexHull(points))
    plt.show()


def plot_hull_and_lines_and_fill(points, point):
    c = ConvexHull(points)
    # hs = HalfspaceIntersection(c.equations, point)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, aspect='equal')
    xlim, ylim = (-1, 3), (-1, 3)
    fmt = {"color": None, "edgecolor": "b", "alpha": 0.5}
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    x = np.linspace(-1, 3, 100)
    signs = [0, -1, -1]
    for h, sign in zip(c.equations, signs):
        hlist = h.tolist()
        if h[1] == 0:
            ax.axvline(-h[2]/h[0], label='{}x+{}y+{}=0'.format(*hlist), **fmt)
            xi = np.linspace(xlim[sign], -h[2]/h[0], 100)
            # ax.fill_between(xi, ylim[0], ylim[1])
        else:
            ax.plot(x, (-h[2]-h[0]*x)/h[1],
                    label='{}x+{}y+{}=0'.format(*hlist))
            # ax.fill_between(x, (-h[2]-h[0]*x)/h[1], ylim[sign], **fmt)
    hull = ConvexHull(points)
    plt.fill(points[hull.vertices, 0],
             points[hull.vertices, 1], 'k', alpha=0.3)
    plot_hull(hull)


def second_triangle_broken_todo():
    plot_hull_and_lines_and_fill(
        np.array([[0, 0], [1, 0], [0.5, 0.5]]), np.array([0.5, 0.25]))

    plot_hull_and_lines_and_fill(
        np.array([[1, 1], [2, 1], [1.5, 1.5]]), np.array([1.5, 1.25]))

    plt.grid(True)
    plt.show()


def gen_regions():

    rng = np.random.default_rng()
    initial_points = np.array(
        [[0, 0], [0, 1], [1.5, 0.5], [2, 0], [1, 2], [3, 3]])

    for _ in range(20):
        initial_points = 5 * rng.random((20, 2))

        c = ConvexHull(initial_points)
        centroid = np.mean(initial_points[c.vertices, :], axis=0)
        print(initial_points[c.vertices, :])
        print('mean:', np.mean(initial_points[c.vertices, :], axis=0))

        plt.plot(centroid[0], centroid[1], "x", color='red')

        plot_hull(c, color="brown")
        fp = []
        for point in initial_points:
            hull = g.linearise_reachable_region(1, 6, point)
            fp.extend(hull.points)
            # plot_hull(hull, color="blue", alpha=0.5)

        fh = ConvexHull(fp)
        plot_hull(fh, color="blue")

        plt.grid(True)
        plt.axis('equal')
        plt.show()

