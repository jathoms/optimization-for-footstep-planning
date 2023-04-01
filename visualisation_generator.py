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


def graphs_4():

    approach1_time = [0.0017980849952436984, 0.0013708369951928034, 0.0013581180101027712, 0.0041570129978936166, 0.003224414002033882, 0.002580307991593145, 0.0023119919933378696, 0.002026624992140569, 0.005215356999542564, 0.001325662000454031,
                      0.003471089992672205, 0.0031170180009212345, 0.0030547170026693493, 0.002526028998545371, 0.0021401909907581285, 0.001933661988005042, 0.0013087810075376183, 0.0013513300073100254, 0.0035757380101131275, 0.0029236620030133054, 0.0027662419888656586]

    approach1_steps = [11, 10, 9, 27, 22, 19, 19, 14,
                       25, 8, 28, 23, 20, 17, 15, 12, 9, 9, 26, 21, 18]
    approach2_time = [0.013648410007590428, 0.007561539998278022, 0.005211414987570606, 0.6483729450119426, 0.5622939270106144, 0.060368338992702775, 0.10589576800703071, 0.06644841199158691, 0.15123437900911085, 0.003916280998964794,
                      0.36872710700845346, 0.3217708180018235, 0.09200349199818447, 0.040610655007185414, 0.06828647300426383, 0.014194834002410062, 0.005921882999246009, 0.00410555099369958, 0.23020695098966826, 0.0767338750010822, 0.12065448400971945]

    approach_1_steps = [
        25, 20, 5, 9, 14, 19, 27, 22, 8, 12, 15, 18, 25, 21, 6, 9, 13, 18, 27, 22
    ]
    approach_1_times = [
        0.003970313991885632, 0.0024563059996580705, 0.0009860919963102788,
        0.0013283529988257214, 0.0019679560064105317, 0.002135804999852553,
        0.002881250999053009, 0.0027864030125783756, 0.0012412920041242614,
        0.0016319529968313873, 0.0017978959949687123, 0.0022102460061432794,
        0.00275236000015866, 0.0025290140038123354, 0.0010000290058087558,
        0.0013276249956106767, 0.0017701949982438236, 0.0021449449995998293,
        0.002864708993001841, 0.00282281800173223
    ]
    approach_2_times = [
        0.03881304699461907, 0.024297762996866368, 0.0012920689914608374,
        0.0022313049994409084, 0.0025640980020398274, 0.014771164002013393,
        0.0641866520018084, 0.02427998199709691, 0.0016666669980622828,
        0.0023019090003799647, 0.0028770120115950704, 0.005681594004272483,
        0.03617974500230048, 0.026019534008810297, 0.0015421790012624115,
        0.0018350610043853521, 0.0025736949901329353, 0.014507708998280577,
        0.06095206399913877, 0.024221527011832222
    ]
    plt.scatter(approach_1_steps, approach_1_times,
                color='blue', label='Graph-assisted MIP')
    plt.scatter(approach_1_steps, approach_2_times,
                color='red', label='Pure MIP')
    plt.xlabel('Steps Taken')
    plt.ylabel('Time Taken')
    plt.grid(True)
    plt.yscale('log')

    ax = plt.gca()

    ax.set_xticks(range(int(min(approach1_steps)),
                  int(max(approach1_steps)) + 1, 3))
    plt.legend()
    plt.show()


def graphs_5():
    approach1_data = [
        (0.00630649300001096, 11),
        (0.0050072610029019415, 10),
        (0.00804760200844612, 15),
        (0.006503793993033469, 13),
        (0.0029570439946837723, 6),
        (0.004514770000241697, 9),
        (0.0077122150105424225, 15),
        (0.005006942999898456, 10),
        (0.007400945003610104, 15),
        (0.004070121998665854, 8),
        (0.00582166099047754, 12),
        (0.004598156010615639, 9),
        (0.005992840000544675, 12),
        (0.00441604400111828, 9),
        (0.007687933000852354, 15),
        (0.0021537830034503713, 4),
        (0.005772793010692112, 11),
        (0.007686049008043483, 15),
        (0.005954199994448572, 12),
        (0.006417009994038381, 13),
        (0.0041489030118100345, 8),
    ]

    approach2_data = [
        (0.021323116001440212, 11),
        (0.015946491999784485, 10),
        (0.02789114200277254, 15),
        (0.017673148002359085, 13),
        (0.004609180003171787, 6),
        (0.012279447008040734, 9),
        (0.020839727003476582, 15),
        (0.009047883009770885, 10),
        (0.058554268995067105, 15),
        (0.0053724809986306354, 8),
        (0.02674638800090179, 12),
        (0.010887402007938363, 9),
        (0.019803059985861182, 12),
        (0.013154540007235482, 9),
        (0.031934191996697336, 15),
        (0.004030730007798411, 4),
        (0.014375384998857044, 11),
        (0.02033184000174515, 15),
        (0.012974287004908547, 12),
        (0.02001790900249034, 13), (0.008241268005804159, 8),
    ]

    approach1_time, approach1_steps = zip(*approach1_data)
    approach2_time, approach2_steps = zip(*approach2_data)

    plt.scatter(approach1_steps, approach1_time,
                c='blue', label='Graph-assisted MIP')
    plt.scatter(approach2_steps, approach2_time, c='red', label='Pure MIP')
    plt.xlabel('Steps Taken')
    plt.ylabel('Time Taken')
    plt.grid(True)
    plt.yscale('log')

    ax = plt.gca()

    ax.set_xticks(range(int(min(approach1_steps)),
                  int(max(approach2_steps)) + 1, 2))
    plt.legend(loc='upper left')
    plt.ylim(0, 0.2)
    plt.show()


def graphs_3():
    approach1_data = [
        (0.0017980849952436984, 11),
        (0.0013708369951928034, 10),
        (0.0013581180101027712, 9),
        (0.0041570129978936166, 27),
        (0.003224414002033882, 22),
        (0.002580307991593145, 19),
        (0.0023119919933378696, 19),
        (0.002026624992140569, 14),
        (0.005215356999542564, 25),
        (0.001325662000454031, 8),
        (0.003471089992672205, 28),
        (0.0031170180009212345, 23),
        (0.0030547170026693493, 20),
        (0.002526028998545371, 17),
        (0.0021401909907581285, 15),
        (0.001933661988005042, 12),
        (0.0013087810075376183, 9),
        (0.0013513300073100254, 9),
        (0.0035757380101131275, 26),
        (0.0029236620030133054, 21),
        (0.0027662419888656586, 18),
    ]

    approach2_data = [
        (0.013648410007590428, 11),
        (0.007561539998278022, 10),
        (0.005211414987570606, 9),
        (0.6483729450119426, 27),
        (0.5622939270106144, 22),
        (0.060368338992702775, 19),
        (0.10589576800703071, 19),
        (0.06644841199158691, 14),
        (0.15123437900911085, 25),
        (0.003916280998964794, 8),
        (0.36872710700845346, 28),
        (0.3217708180018235, 23),
        (0.09200349199818447, 20),
        (0.040610655007185414, 17),
        (0.06828647300426383, 15),
        (0.014194834002410062, 12),
        (0.005921882999246009, 9),
        (0.00410555099369958, 9),
        (0.23020695098966826, 26),
        (0.0767338750010822, 21),
        (0.12065448400971945, 18),
    ]

    approach1_time, approach1_steps = zip(*approach1_data)
    approach2_time, approach2_steps = zip(*approach2_data)

    plt.scatter(approach1_steps, approach1_time,
                c='blue', label='Graph-assisted MIP')
    plt.scatter(approach2_steps, approach2_time, c='red', label='Pure MIP')
    plt.xlabel('Steps Taken')
    plt.ylabel('Time Taken')
    plt.grid(True)
    plt.yscale('log')

    ax = plt.gca()

    ax.set_xticks(range(int(min(approach1_steps)),
                  int(max(approach2_steps)) + 1, 3))
    plt.legend()
    # plt.ylim(0, 0.2)
    plt.show()


def graphs_1():
    approach_1_times = [
        0.01824564101116266, 0.01378412100893911, 0.00474457599921152,
        0.01321882700722199, 0.015172323008300737, 0.019559616994229145,
        0.012187228989205323, 0.010925498994765803, 0.011465471994597465,
        0.0047283229941967875, 0.017578043989487924, 0.013915818009991199,
        0.0181380449939752, 0.017567393006174825, 0.01736838900251314
    ]

    approach_1_steps = [
        43, 32, 11, 32, 35, 45, 24, 26, 28, 12, 43, 34, 43, 43, 42
    ]

    approach_2_times = [
        0.2727088760002516, 0.133427787994151, 0.009732539998367429,
        0.16440911800600588, 0.21875037800054997, 0.2613523990003159,
        0.06107317800342571, 0.07084370400116313, 0.09704320099262986, 0.011794179998105392,
        0.4050804370053811, 0.16900627400900703, 0.5479234780068509,
        0.39330620400141925, 0.30291061601019464
    ]

    plt.scatter(approach_1_steps, approach_1_times,
                color='blue', label='Graph-assisted MIP')
    plt.scatter(approach_1_steps, approach_2_times,
                color='red', label='Pure MIP')
    plt.xlabel('Steps Taken')
    plt.ylabel('Time Taken')
    plt.grid(True)
    plt.yscale('log')

    ax = plt.gca()

    ax.set_xticks(range(int(min(approach_1_steps)),
                  int(max(approach_1_steps)) + 1, 3))
    plt.legend()
    plt.show()


def graphs_2():
    approach_1_steps = [
        21, 28, 5, 21, 24, 7, 21, 22, 9, 18, 20, 11, 14, 17, 13, 16, 15, 15, 14, 13
    ]
    approach_1_times = [
        0.00969420000910759, 0.012934559999848716, 0.002117484007612802,
        0.008871932994225062, 0.010277018998749554, 0.0027864809962920845,
        0.009239882012479939, 0.009279706995585002, 0.003849425003863871,
        0.007816821991582401, 0.010202946999925189, 0.00464867299888283,
        0.005776000994956121, 0.007513982011005282, 0.00535295601002872,
        0.006530996994115412, 0.006142880010884255, 0.006799780006986111, 0.006401685997843742,
        0.005510707997018471
    ]
    approach_2_times = [
        0.08898907400725875, 0.9465635230008047, 0.005832661991007626,
        0.07046299701323733, 0.3686280020046979, 0.010752286994829774,
        0.05517133399553131, 0.16830886500247288, 0.016303821990732104,
        0.03385537899157498, 0.0738292449968867, 0.015395303998957388,
        0.016570231993682683, 0.0622919519955758, 0.03328937399783172,
        0.05637851399660576, 0.042280239998945035, 0.036018072001752444,
        0.03828308200172614, 0.025773086003027856
    ]

    plt.scatter(approach_1_steps, approach_1_times,
                color='blue', label='Graph-assisted MIP')
    plt.scatter(approach_1_steps, approach_2_times,
                color='red', label='Pure MIP')
    plt.xlabel('Steps Taken')
    plt.ylabel('Time Taken')
    plt.grid(True)
    plt.yscale('log')

    ax = plt.gca()

    ax.set_xticks(range(int(min(approach_1_steps)),
                  int(max(approach_1_steps)) + 1, 3))
    plt.legend()
    plt.show()


def graphs_6():
    approach_1_steps = [
        5, 4, 3, 5, 6, 5, 5, 2, 3, 4, 6, 5, 6, 6, 5, 4
    ]
    approach_1_times = [
        0.000452345993835479, 0.0004683290026150644, 0.0004238090186845511,
        0.0004244599840603769, 0.0004658630059566349, 0.00044704799074679613,
        0.0004445189842954278, 0.00023554699146188796, 0.0003953029809053987,
        0.00044924599933438003, 0.0004675719828810543, 0.00043618999188765883,
        0.0005317519826348871, 0.0006138739991001785, 0.00046349799958989024,
        0.00040796902612783015
    ]
    approach_2_times = [
        0.0025720029952935874, 0.0019494339940138161, 0.0005308199906721711,
        0.0022413000115193427, 0.0022776029945816845, 0.0018169730028603226,
        0.0018317640060558915, 0.0004621779953595251,
        0.002003363973926753, 0.0019563919922802597, 0.002450571017106995,
        0.002014205005252734, 0.0020057199872098863, 0.002155229012714699,
        0.0041903970122803, 0.0017715130234137177
    ]
    plt.scatter(approach_1_steps, approach_1_times,
                color='blue', label='Graph-assisted MIP')
    plt.scatter(approach_1_steps, approach_2_times,
                color='red', label='Pure MIP')
    plt.xlabel('Steps Taken')
    plt.ylabel('Time Taken')
    plt.grid(True)
    plt.yscale('log')

    ax = plt.gca()

    ax.set_xticks(range(int(min(approach_1_steps)),
                  int(max(approach_1_steps)) + 1))
    plt.legend()
    plt.show()


graphs_1()
graphs_2()
graphs_3()
graphs_4()
graphs_5()
graphs_6()
