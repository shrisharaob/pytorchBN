import numpy as np


def relu(x):
    return np.maximum(0, x)


def one_pop_network(N, K, JII, JI0=1.0, iters=100):
    # Generate adjacency matrix for Erdos-Renyi network
    adjacency_matrix = np.random.binomial(1, K / N, size=(N, N))
    # np.fill_diagonal(adjacency_matrix, 0)  # No self-connections
    # Compute weights
    weights = JII / np.sqrt(K) * adjacency_matrix
    ff_vector = JI0 * np.sqrt(K) * np.ones(N)
    rates = []
    old_rates = np.zeros(N)
    for i in range(iters):
        new_rates = relu(np.dot(weights, old_rates) + ff_vector)
        rates.append(new_rates)
        old_rates = new_rates
    return np.stack(rates)


def main(iters=100):
    N = 1000  # Number of units
    K = 100  # Average number of inputs per unit
    JII = -0.01  # Weight parameter
    JI0 = 1.0
    #
    rates = one_pop_network(N, K, JII=JII, JI0=JI0, iters=iters)
    print(rates.shape)
    print(f'JI0={JI0} JII={JII}')
    #
    import matplotlib.pyplot as plt
    plt.plot(rates.mean(axis=1))
    plt.title('network avg rate')
    # plt.hist(rates)
    plt.figure()
    plt.hist(rates[-1, :])
    plt.title('rate distr')
    plt.show()


if __name__ == '__main__':
    main()
