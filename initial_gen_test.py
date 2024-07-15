from timeit import default_timer
from skopt import Optimizer, sampler
from matplotlib import pyplot as plt


def test():
    sob = sampler.Sobol()
    lhs = sampler.Lhs()
    sob_points = take_sample(sob)
    lhs_points = take_sample(lhs)


def take_sample(model, num=2, initial_points=10):
    points = []
    n = 0
    while n < initial_points:
        x = model.generate([(1,10), (1,10)], num)
        for p in x:
            points.append(p)
            n+=1
            if n == initial_points:
                break
            # if p[0]<p[1]:
            #     points.append(p)
            #     n+=1
            #     if n == initial_points:
            #         break
    return points

if __name__ == "__main__":
    # take samples
    sob = sampler.Sobol()
    lhs = sampler.Lhs()
    sob_points = take_sample(sob, 2, 16)
    lhs_points = take_sample(lhs, 2, 16)

    sob_points2 = take_sample(sob, 16, 16)
    lhs_points2 = take_sample(lhs, 16, 16)

    # plot 
    fig = plt.figure()
    fig.set_size_inches(10,10)
    ax = fig.add_subplot(221)
    ax.scatter([i[0] for i in sob_points], [j[1] for j in sob_points])
    ax.set_title("SOBOL_Limited")
    ax2 = fig.add_subplot(222)
    ax2.scatter([i[0] for i in lhs_points], [j[1] for j in lhs_points])
    ax2.set_title("LHS_Limited")
    ax3 = fig.add_subplot(223)
    ax3.scatter([i[0] for i in sob_points2], [j[1] for j in sob_points2])
    ax3.set_title("SOBOL")
    ax4 = fig.add_subplot(224)
    ax4.scatter([i[0] for i in lhs_points2], [j[1] for j in lhs_points2])
    ax4.set_title("LHS")
    plt.savefig("sobol vs lhs")
    plt.show()
    # print(points)