import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",   # This is needed only in the slides
    "font.sans-serif": ["Helvetica Neue"],   # This is needed only in the slides
    "font.size": 12,   # In the paper you can put 11 or 12
    })


def comp_time():
    N = np.arange(1, 11)
    base_perf_time = np.array([.133, .249, .222, .268, .359, .414, .415, .593, .579, .636])

    fig, ax = plt.subplots(figsize=(6, 4))

    plt.plot(N, base_perf_time, 'go', label='SDP relaxation')

    plt.title('Computation time of SDP relaxation')
    plt.xlabel('$N$')
    plt.ylabel('time (seconds)')

    plt.legend()

    # plt.show()
    fname = 'images/NNLS_times.pdf'
    plt.savefig(fname)


def plot_NNLS():
    N = np.arange(1, 11)
    print(N)
    base_perf = np.array([1140.396, 560.393, 232.06, 102.35, 49.311, 26.680, 16.587, 11.705, 9.205, 7.84])
    base_perf_time = np.array([.133, .249, .222, .268, .359, .414, .415, .593, .579, .636])

    # warm_start_perf = np.array([14.294, 11.188, 8.779, 7.528, 6.798, 6.357, 6.06, 5.84, 5.70, 5.60])
    warm_start_perf = np.array([.344, 1.13, 1.47, 1.78, 2.07, 2.315, 2.511, 2.656, 2.768, 2.847])

    gurobi_perf = np.array([525.749, 43.0074, 11.6696, 1.0766, .84451])  # 6: .37339 # 7: .10546, #8: .04806, #10: .01085
    N_g = np.arange(1, 6)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(N, base_perf, 'g-', label='sdp relaxation')
    # plt.plot(N, warm_start_perf, 'ro', label='warm-started sdp')
    ax.plot(N_g, gurobi_perf, 'b-', label='globally optimal solution')

    plt.title('Worst-case performance, SDP vs globally optimal')
    plt.xlabel('$N$')
    plt.ylabel('maximum $|| x^N - x^0 ||_2^2$')
    plt.yscale('log')

    plt.legend()

    plt.show()
    # fname = 'images/NNLS_values.pdf'
    # plt.savefig(fname)


def warm_start():
    N = np.arange(1, 11)
    print(N)
    base_perf = np.array([1140.396, 560.393, 232.06, 102.35, 49.311, 26.680, 16.587, 11.705, 9.205, 7.84])

    # warm_start_perf = np.array([14.294, 11.188, 8.779, 7.528, 6.798, 6.357, 6.06, 5.84, 5.70, 5.60])
    warm_start_perf = np.array([.344, 1.13, 1.47, 1.78, 2.07, 2.315, 2.511, 2.656, 2.768, 2.847])

    fig, ax = plt.subplots(figsize=(6, 4))

    plt.plot(N, base_perf, 'g', label='original sdp relaxation')
    # plt.plot(N, warm_start_perf, 'ro', label='warm-started sdp')
    plt.plot(N, warm_start_perf, 'r', label='sdp relaxation for warm-starting')

    plt.title('Worst-case performance, original vs warm-started')
    plt.xlabel('$N$')
    plt.ylabel('maximum $|| x^N - x^{N-1} ||_2^2$')
    plt.yscale('log')

    plt.legend()

    # plt.show()
    fname = 'images/NNLS_warmstart.pdf'
    plt.savefig(fname)


def plot_NNLS_smaller():
    N = np.arange(1, 11)
    print(N)
    base_perf = np.array([359.399, 288.485, 224.303, 176.235, 140.051, 112.606, 91.703, 75.666, 63.398, 53.955])

    gurobi_perf = np.array(
        [114.54, 6.845, 1.7877, 1.0561, 0.626, 0.381, 0.280, 0.207, 0.154, .114])
    N_g = np.arange(1, 11)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(N, base_perf, 'g-', label='sdp relaxation')
    # plt.plot(N, warm_start_perf, 'ro', label='warm-started sdp')
    ax.plot(N_g, gurobi_perf, 'b-', label='globally optimal solution')

    plt.title('Worst-case performance, SDP vs globally optimal, m=5, n=3, r=10')
    plt.xlabel('$N$')
    plt.ylabel('maximum $|| x^N - x^{N-1} ||_2^2$')
    plt.yscale('log')

    plt.legend()

    # plt.show()
    fname = 'images/NNLS_values_smallerprob.pdf'
    plt.savefig(fname)


def main():
    # comp_time()
    # plot_NNLS()
    # plot_NNLS_smaller()
    warm_start()


if __name__ == '__main__':
    main()
