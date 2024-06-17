import torch
import torch.nn as nn
import torch.nn.functional as F


class BalRNN(nn.Module):
    '''one pop bal network'''

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 batch_first=True,
                 K=10,
                 JI0=0.4,
                 JII=-0.1):
        super(BalRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.JI0 = JI0
        self.JII = JII
        #
        self.history = []
        #
        self.weight_ih = nn.ParameterList([
            nn.Parameter(
                torch.randn(hidden_size, input_size) *
                (JI0 / torch.sqrt(torch.tensor(K, dtype=torch.float32))))
            for _ in range(num_layers)
        ])

        self.weight_hh = nn.ParameterList([
            nn.Parameter(self.initialize_sparse_weight(hidden_size, K, JII))
            for _ in range(num_layers)
        ])

    def initialize_ff_weight(self, input_size, hidden_size, K, JI0):
        indices = []
        values = []
        for i in range(hidden_size):
            selected_indices = torch.randperm(input_size)[:K]
            indices.extend([[i, j] for j in selected_indices])
            values.extend(
                [self.JI0 / torch.sqrt(torch.tensor(K, dtype=torch.float32))] *
                K)  # list mult

        indices = torch.tensor(indices).t()
        values = torch.tensor(values, dtype=torch.float32)
        ff_sparse_weight = torch.sparse.FloatTensor(
            indices, values, torch.Size([hidden_size, input_size]))
        return ff_sparse_weight

    def initialize_sparse_weight(self, hidden_size, K, J):
        indices = []
        values = []
        for i in range(hidden_size):
            selected_indices = torch.randperm(hidden_size)[:K]
            indices.extend([[i, j] for j in selected_indices])
            values.extend(
                [self.JII / torch.sqrt(torch.tensor(K, dtype=torch.float32))] *
                K)

        indices = torch.tensor(indices).t()
        values = torch.tensor(values, dtype=torch.float32)
        sparse_weight = torch.sparse.FloatTensor(
            indices, values, torch.Size([hidden_size, hidden_size]))
        return sparse_weight

    def transfer_function(self, total_input):
        return F.relu(total_input)  # should we care about positive rates?
        # return torch.ones(x.shape) +
        # return torch.tanh(total_input)

    def forward(self, x, h_0=None):
        if self.batch_first:
            x = x.transpose(0, 1)  # Ensure (seq_len, batch_size, input_size)
        seq_len, batch_size, _ = x.size()
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        h_t_minus_1 = h_0
        h_t = h_0.clone()
        output = []
        ff_input = torch.ones(self.hidden_size) * torch.sqrt(torch.tensor(K))
        print('ffshape:', ff_input.shape)
        tmp = torch.sparse.mm(self.weight_hh[0], h_t_minus_1[0].T).T
        print(tmp.shape)
        tmp2 = ff_input + tmp
        print('tmp2shape:', tmp2.shape)
        for t in range(seq_len):
            for layer in range(self.num_layers):
                if layer == 0:  # x[t] @ self.weight_ih[layer].T
                    h_t[layer] = self.transfer_function(
                        ff_input + torch.sparse.mm(self.weight_hh[layer],
                                                   h_t_minus_1[layer].T).T)
                    # h_t[layer] = self.transfer_function(
                    #     torch.sparse.mm(self.weight_ih[layer], x[t].T).T +
                    #     torch.sparse.mm(self.weight_hh[layer],
                    #                     h_t_minus_1[layer].T).T)
                else:
                    h_t[layer] = self.transfer_function(
                        torch.sparse.mm(self.weight_hh[layer], \
                                        h_t[layer - 1].T).T +
                        torch.sparse.mm(self.weight_hh[layer],
                                        h_t_minus_1[layer].T).T)
            output.append(h_t[-1])
            h_t_minus_1 = h_t.clone()
        output = torch.stack(output)
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, h_t


if __name__ == '__main__':
    input_size = 250  # num of ff neurons
    hidden_size = 1000  # num of neurons in the balanced network
    num_layers = 1  # 1 layer of balanced network
    batch_size = 1
    seq_len = 100  # simulation time
    K = 100
    JI0 = 1.0
    JII = -0.01
    rnn = BalRNN(input_size,
                 hidden_size,
                 num_layers,
                 batch_first=True,
                 K=K,
                 JI0=JI0,
                 JII=JII)
    # (batch_size, seq_len, input_size)
    x = 1.0 * torch.ones(batch_size, seq_len, input_size)
    # total_input will be: O(K * JI0 / sqrt(K)) = O(sqrt(K))
    # randn(3, 5, input_size)
    h_0 = torch.zeros(num_layers, batch_size,
                      hidden_size)  # (num_layers, batch_size, hidden_size)
    output, h_n = rnn(x, h_0)
    print(output.shape)
    print(h_n.shape)
    print(f'JI0={JI0} JII={JII}')
    import matplotlib.pyplot as plt
    import numpy as np

    # plt.hist(h_n.detach().numpy().squeeze(), 25)

    rates = output.detach().numpy().squeeze()
    #.mean(axis=1)
    # plt.plot(rates[:, :10])
    fig, ax = plt.subplots()
    ri = rates.mean(axis=0)
    print(ri.min(), ri.max(), sum(ri <= 0))
    ax.hist(ri, 25)
    ax.set_xscale('symlog')
    # ax.set_xticks(np.linspace(1e-5, 1e1, 1e-1))
    plt.title('rate distr: is this balanced??')
    plt.figure()
    plt.plot(rates.mean(axis=1))
    plt.title('network avg rate')
    print(sum(rates.mean(axis=1) <= 0))
    plt.show()
