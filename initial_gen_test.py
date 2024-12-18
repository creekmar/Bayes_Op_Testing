"""
File: initial_gen_test.py
Author: Ming Creekmore
Purpose: To show how evenly distributed the sample points from LHS
         and sobol are, including for when samples are take a few 
         points at a time vs when taken at one time
"""

from timeit import default_timer
from skopt import Optimizer, sampler
from matplotlib import pyplot as plt


def take_sample(model, num=1, initial_points=10):
    """
    Take a specified number of samples given
    @param: model: the sampler to use
    @param: num: the number of samples to take per round
    @param: initial_points: the number of total samples to take
    """
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
    sob_not_random = sampler.Sobol(randomize=False)
    lhs = sampler.Lhs()
    lhs_center = sampler.Lhs("centered")
    sob_points = take_sample(sob, 2, 16)
    sob_not_random_points = take_sample(sob_not_random, 2, 16)
    lhs_points = take_sample(lhs, 2, 16)
    lhs_center_points = take_sample(lhs_center, 2, 16)

    sob_points2 = take_sample(sob, 16, 16)
    sob_not_random_points2 = take_sample(sob_not_random, 16, 16)
    lhs_points2 = take_sample(lhs, 16, 16)
    lhs_center_points2 = take_sample(lhs_center, 16, 16)

    # plot LIMITED
    fig = plt.figure()
    fig.set_size_inches(20,10)
    ax = fig.add_subplot(241)
    ax.grid()
    ax.scatter([i[0] for i in sob_points], [j[1] for j in sob_points])
    ax.set_title("SOBOL_Limited")

    ax = fig.add_subplot(242)
    ax.grid()
    ax.scatter([i[0] for i in sob_not_random_points], [j[1] for j in sob_not_random_points])
    ax.set_title("SOBOL_Not_Random_Limited")

    ax2 = fig.add_subplot(243)
    ax2.grid()
    ax2.scatter([i[0] for i in lhs_points], [j[1] for j in lhs_points])
    ax2.set_title("LHS_Limited")

    ax2 = fig.add_subplot(244)
    ax2.grid()
    ax2.scatter([i[0] for i in lhs_center_points], [j[1] for j in lhs_center_points])
    ax2.set_title("LHS_Center_Limited")

    # PLOT REGULAR

    ax3 = fig.add_subplot(245)
    ax3.grid()
    ax3.scatter([i[0] for i in sob_points2], [j[1] for j in sob_points2])
    ax3.set_title("SOBOL")

    ax3 = fig.add_subplot(246)
    ax3.grid()
    ax3.scatter([i[0] for i in sob_not_random_points2], [j[1] for j in sob_not_random_points2])
    ax3.set_title("SOBOL_Not_Random")

    ax4 = fig.add_subplot(247)
    ax4.grid()
    ax4.scatter([i[0] for i in lhs_points2], [j[1] for j in lhs_points2])
    ax4.set_title("LHS")

    ax4 = fig.add_subplot(248)
    ax4.grid()
    ax4.scatter([i[0] for i in lhs_center_points2], [j[1] for j in lhs_center_points2])
    ax4.set_title("LHS_Center")

    plt.savefig("sobol vs lhs")
    plt.show()
    # print(points)